"""A fake V4.1 forward for the serving tests: the ``protocol.Forward`` rules with toy arithmetic.

Its state lives where the real forward's does, so the serving layer's bugs show up as different logits:

- pool rows (``Pool.view``: GLM ``Paged`` over the shared physical tensors): every row writes layer 20's ratio-1
  ``comp.20`` / ``index_k.20`` row at its position, and a closing ratio-2 group writes ``comp.2`` from the carry;
- ``state.SlotState``: an encoder ring (layer 0) and a decoder ring (layer 25), the ratio-2 carry, the Engram
  lookback, the DSpark taps; snapshots / restores go through ``SlotState.snapshot`` / ``restore``;
- CED: in ``replay`` mode prefill writes no decoder rows; ``finish_prompt`` rebuilds the decoder ring from the tail.

A row's logits are a hash of everything it can see (every pool row at or before it, the ring rows in its 128-row
window, the carry, the lookback, the tap), so a wrong page, ring row, carry or lookback after a restore changes
them. Rows are independent of their window (row invariance by construction). ``draft`` is an oracle drafter (the
greedy continuation, with some drafts deliberately wrong) to exercise acceptance."""

from __future__ import annotations

import numpy as np
import torch

from engine.serving.protocol import Bounded, Candidates
from engine.serving.state import RING, TAP_WINDOW, SlotState

M64 = (1 << 64) - 1
ENC, DEC = 0, 25


def _mix(x: int) -> int:
    x &= M64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & M64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def _h(*vals: int) -> int:
    x = 0x9E3779B97F4A7C15
    for v in vals:
        x = _mix(x ^ (int(v) & M64))
    return x


def _wsum(arr: np.ndarray) -> int:
    """Order-sensitive digest of uint64 rows."""

    if arr.size == 0:
        return 0
    w = np.arange(1, arr.size + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return int(np.sum(arr.astype(np.uint64) * w, dtype=np.uint64))


class FakeForward:
    def __init__(self, topo, pool, n_slots: int, vocab: int = 4096, *, eos: int = 1, eos_bias: float = 0.0,
                 wrong_every: int = 3, seed: int = 0) -> None:
        self.topo, self.pool, self.vocab = topo, pool, int(vocab)
        self.eos, self.eos_bias = eos, eos_bias
        self.wrong_every = wrong_every
        self.seed = seed
        self.st = [SlotState(topo, "cpu") for _ in range(n_slots)]
        self.slots = None                          # the executor's pool.Slot list (bound by ``bind``)
        self.calls: list[tuple] = []               # what was asked, for rank agreement checks
        self.win_carry: dict[int, list[tuple[float, list[int]]]] = {}
        self.win_start: dict[int, tuple[int, list[int]]] = {}
        self.mode = "full"

    def bind(self, slots) -> None:
        self.slots = slots
        for s, st in zip(slots, self.st):
            st.pages = s

    # -- one row ---------------------------------------------------------------------------------------------------
    def _row(self, slot: int, t: int, p: int, *, decoder: bool, logits: bool):
        st = self.st[slot]
        sp = self.slots[slot]
        hv = _h(t, p, 11, *st.lookback)              # Engram: the n-gram lookback shapes every later KV row
        b = np.frombuffer(np.uint64(hv).tobytes(), dtype=np.uint8)
        comp20 = self.pool.view(sp, "comp.20")
        comp20[p] = torch.from_numpy(np.pad(b, (0, comp20.shape[1] - 8)))
        idx20 = self.pool.view(sp, "index_k.20")
        row = torch.zeros(idx20.shape[1], dtype=torch.bfloat16)
        row[0] = float(t % 251)
        idx20[p] = row
        if p % 2 == 0:
            st.carry[0, 0] = float(t)
        else:
            c = int(st.carry[0, 0].item())
            b2 = np.frombuffer(np.uint64(_h(c, t, 2)).tobytes(), dtype=np.uint8)
            comp2 = self.pool.view(sp, "comp.2")
            comp2[p // 2] = torch.from_numpy(np.pad(b2, (0, comp2.shape[1] - 8)))
        st.swa_values[ENC, p % RING, :8] = torch.from_numpy(b.copy())
        if decoder:
            b3 = np.frombuffer(np.uint64(_h(t, p, 25)).tobytes(), dtype=np.uint8)
            st.swa_values[DEC, p % RING, :8] = torch.from_numpy(b3.copy())
        st.push([t])
        st.taps[p % TAP_WINDOW, 0, 0] = float(t % 97)
        st.pos = p + 1
        if not logits:
            return None
        return self._logits(slot, p)

    def _ring(self, st: SlotState, layer: int, p: int) -> int:
        lo = max(0, p - st.window + 1)
        idx = [q % RING for q in range(lo, p + 1)]
        raw = st.swa_values[layer, idx, :8].contiguous().numpy().view(np.uint64).reshape(-1)
        return _wsum(raw)

    def _logits(self, slot: int, p: int) -> np.ndarray:
        st, sp = self.st[slot], self.slots[slot]
        c20 = self.pool.view(sp, "comp.20").gather(0, p + 1)[:, :8].contiguous().numpy().view(np.uint64).reshape(-1)
        i20 = self.pool.view(sp, "index_k.20").gather(0, p + 1)[:, 0].float().numpy().astype(np.uint64)
        n2 = (p + 1) // 2
        c2 = self.pool.view(sp, "comp.2").gather(0, n2)[:, :8].contiguous().numpy().view(np.uint64).reshape(-1) \
            if n2 else np.zeros(0, np.uint64)
        d = _h(_wsum(c20), _wsum(i20), _wsum(c2), self._ring(st, ENC, p), self._ring(st, DEC, p),
               int(st.carry[0, 0].item()) if p % 2 == 0 else -1, *st.lookback,
               int(st.taps[(p - 5) % TAP_WINDOW, 0, 0].item()) if p >= 5 else -1, self.seed)
        rng = np.random.default_rng(d)
        v = rng.standard_normal(self.vocab).astype(np.float32)
        v[self.eos] += self.eos_bias
        return v

    # -- the protocol ----------------------------------------------------------------------------------------------
    def reset(self, slot: int) -> None:
        self.calls.append(("reset", slot))
        self.st[slot].reset()

    def prefill(self, pieces, *, mode: str) -> None:
        self.calls.append(("prefill", tuple((p.slot, p.start, p.ids) for p in pieces), mode))
        for pc in pieces:
            if self.st[pc.slot].pos != pc.start:
                raise AssertionError(f"slot {pc.slot}: piece at {pc.start}, state at {self.st[pc.slot].pos}")
            for i, t in enumerate(pc.ids):
                self._row(pc.slot, t, pc.start + i, decoder=mode == "full", logits=False)

    def finish_prompt(self, slot: int, end: int, tail, *, mode: str) -> None:
        self.calls.append(("finish", slot, end, tuple(tail), mode))
        st = self.st[slot]
        assert st.pos == end, (st.pos, end)
        if mode == "replay":
            st.swa_values[DEC].zero_()
            for i, t in enumerate(tail):
                q = end - len(tail) + i
                b3 = np.frombuffer(np.uint64(_h(t, q, 25)).tobytes(), dtype=np.uint8)
                st.swa_values[DEC, q % RING, :8] = torch.from_numpy(b3.copy())

    def window(self, windows, *, count: int, masks=None) -> Candidates:
        self.calls.append(("window", tuple((w.slot, w.start, w.tokens) for w in windows), count))
        ids, vals = [], []
        for wi, w in enumerate(windows):
            st = self.st[w.slot]
            assert st.pos == w.start, (w.slot, st.pos, w.start)
            look0 = list(st.lookback)
            carries = []
            for i, t in enumerate(w.tokens):
                v = self._row(w.slot, t, w.start + i, decoder=True, logits=True)
                carries.append(float(st.carry[0, 0].item()))
                if masks is not None and masks[wi] is not None:
                    bits = masks[wi][i]
                    allowed = ((bits[np.arange(self.vocab) // 32].astype(np.int64) >> (np.arange(self.vocab) % 32))
                               & 1).astype(bool)
                    v = np.where(allowed, v, -np.inf).astype(np.float32)
                order = np.lexsort((np.arange(self.vocab), -v))[:count]
                ids.append(order.astype(np.int64))
                vals.append(v[order])
            self.win_carry[w.slot] = carries
            self.win_start[w.slot] = (w.start, look0, list(w.tokens))
        return Candidates(np.stack(ids), np.stack(vals))

    def commit(self, slot: int, accepted: int) -> None:
        self.calls.append(("commit", slot, accepted))
        st = self.st[slot]
        start, look0, toks = self.win_start.pop(slot)
        st.carry[0, 0] = self.win_carry.pop(slot)[accepted]
        st.lookback = (look0 + toks[:accepted + 1])[-3:]
        st.pos = start + accepted + 1

    def snapshot(self, slot: int) -> Bounded:
        self.calls.append(("snapshot", slot))
        return self.st[slot].snapshot("replay" if self.mode == "replay" else "full")

    def restore(self, slot: int, snap: Bounded) -> None:
        self.calls.append(("restore", slot, snap.pos))
        self.st[slot].restore(snap)

    def draft(self, slots, pendings, starts, depths) -> list[list[int]]:
        """The greedy continuation (an oracle), every ``wrong_every``-th draft replaced by a wrong token."""

        self.calls.append(("draft", tuple(slots), tuple(pendings), tuple(starts), tuple(depths)))
        out = []
        for slot, t, p, k in zip(slots, pendings, starts, depths):
            st = self.st[slot]
            saved = (float(st.carry[0, 0].item()), list(st.lookback), st.pos)
            got, cur = [], t
            for i in range(k):
                v = self._row(slot, cur, p + i, decoder=True, logits=True)
                nxt = int(np.lexsort((np.arange(self.vocab), -v))[0])
                if self.wrong_every and (p + i) % self.wrong_every == 0:
                    nxt = (nxt + 7) % self.vocab
                got.append(nxt)
                cur = nxt
            st.carry[0, 0], st.lookback, st.pos = saved[0], saved[1], saved[2]
            out.append(got)
        return out

    def digest(self) -> int:
        """Everything the ranks must agree on (tests of rank agreement)."""

        h = _h(len(self.calls))
        for st in self.st:
            h = _h(h, st.pos, _wsum(st.swa_values[:, :, :8].contiguous().numpy().view(np.uint64).reshape(-1)))
        for t in self.pool.phys.values():
            h = _h(h, _wsum(t.contiguous().view(torch.uint8).numpy()[:, :8].copy().view(np.uint64).reshape(-1)))
        return h
