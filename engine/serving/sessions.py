# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""Sessions for DeepSeek-V4.1-Flash on our GLM session design (0110 RAM / 0250 NVMe / 0310 prefix share / 0540
replay; ``glm5_next/spark/sessions.py``'s prefix digests are reused as is): what a snapshot holds, how it is keyed,
how the RAM tier shares pool pages, and what parking a 300K session costs.

A snapshot at position S is:

- the pool pages below S of every family (``pool.families``: compressed rows + index keys of layers 2, 8, 14, 20),
  kept **mapped in the pool** (RAM tier: the entry holds them; a resume adopts its full pages, copies the partial
  last one) or written to NVMe (``sessdisk``: parked);
- the slot's bounded state (``protocol.Bounded``, ``state.SlotState.snapshot``): the rings' last 128 rows, the
  Engram lookback, the DSpark taps, the carries when S is odd.

When snapshots are taken (``batch.Batcher``): at a prompt's replay point S = ``snapshot_point(n)`` (the last
16-token grid point strictly before its end, GLM 0540: an identical resend resumes there and prefills < 16 tokens)
and at a turn's end (every processed token: the next turn's prompt usually extends it). A prompt resumes from the
longest stored entry whose ids are a strict prefix of it, with the same tag.

Keys: the token ids (256-token page chains, GLM ``sessions.chain``), image spans by their content-hash virtual ids
(GLM 0500), and an arithmetic tag: (prefill kernels: exact | fast grid G) x (CED: full | replay) x KV format.
Entries of different tags never resume each other.

Exactness, ``resumed == fresh`` in both CED modes:

- full mode: the snapshot is exactly the state a fresh prefill reaches at S (row-invariant kernels, the grid rule);
- replay mode: a prompt of n tokens runs the encoder over [S, n) and the decoder over [max(0, n - 128), n) with
  its SWA windows starting at that segment; the decoder reads layer 20's compressed KV of every earlier position
  (written by the encoder pass) and nothing else from before the segment, so decoder rings in the snapshot are
  never read: a resumed prompt computes what a fresh one does.

The NVMe tier (``sessdisk``) writes entries O_DIRECT with per-chunk SHA-256 and a compat ident (image id, knobs,
layout, tag); a parked 300K session is ~0.6 GiB and resumes at NVMe speed (~0.1 s at 6-9 GB/s) plus the CED decoder
replay of 128 rows (tens of ms) in replay mode.
"""

from __future__ import annotations

import hashlib
from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from tensorfold.families.glm5_next.spark import sessions as glm_sessions

from . import protocol as RW  # the FP8 row record's sizes (csa2.rows: 584 = 576 + 8)
from .pool import GRID, PAGE, Pool, Slot, bytes_per_token
from .protocol import Bounded
from .state import LOOKBACK, TAP_WINDOW
from .topology import Topology

MODES = ("full", "replay")
chain = glm_sessions.chain                     # chain[j] determines ids[:256 (j + 1)] (GLM: blake2b-16, chained)
common_prefix = glm_sessions.common_prefix
assert glm_sessions.PAGE == PAGE


@dataclass(frozen=True)
class Tag:
    prefill: str = "exact"          # exact | fast (grid G)
    grid: int = 0
    ced: str = "full"               # full | replay
    kv: str = "fp8"

    def code(self) -> int:
        """The int the round header and the session keys carry."""

        p = 0 if self.prefill == "exact" else self.grid
        return (p << 8) | (MODES.index(self.ced) << 4) | {"fp8": 0, "fp4": 1}[self.kv]


def snapshot_point(n: int, grid: int = GRID) -> int:
    """GLM 0540: the last grid point strictly before a prompt's end (an identical resend resumes there)."""

    return (n - 1) // grid * grid if n > 0 else 0


def snapshot_bytes(topo: Topology, tokens: int, ced: str = "full", index_kv: str = "bf16") -> int:
    pages = int(bytes_per_token(topo, index_kv) * tokens)
    enc = sum(1 for ly in topo.layers if ly.encoder)
    rings = enc if ced == "replay" else len(topo.layers) + len(topo.dspark)
    state = rings * topo.window * RW.ROW_BYTES + LOOKBACK * 4 + len(topo.taps) * topo.hidden * 2 * TAP_WINDOW
    return pages + state


def compat_extra(tag: Tag, topo: Topology) -> dict:
    """What the NVMe tier's compat ident adds for this family (GLM ``sessdisk.compat_ident(extra=...)``)."""

    return {"family": "deepseek_v41", "tag": tag.code(), "row": RW.ROW_BYTES, "window": topo.window,
            "kv_sources": topo.kv_sources, "grid": GRID}


def entry_key(tag: int, ids: Sequence[int]) -> str:
    """An entry's identity (hex): two entries with the same key hold the same state."""

    h = hashlib.blake2b(digest_size=16)
    h.update(b"dsv41-entry")
    h.update(int(tag).to_bytes(4, "little"))
    h.update(len(ids).to_bytes(8, "little"))
    h.update(array("i", [int(t) for t in ids]).tobytes())
    return h.hexdigest()


@dataclass
class Entry:
    """A stored snapshot. ``pages`` (RAM tier: pool pages covering [0, len(ids)), held by the entry) or ``disk``
    (NVMe: the tier's key); ``bounded`` in RAM, None when only on disk."""

    id: int
    key: str
    tag: int
    ids: list[int]
    chain: list[bytes]
    kind: str = "turn"                       # prompt (replay point) | turn (a reply's end)
    pages: list[int] | None = None
    bounded: Bounded | None = None
    disk: bool = False
    used: int = 0

    @property
    def pos(self) -> int:
        return len(self.ids)

    @property
    def in_ram(self) -> bool:
        return self.pages is not None


def is_prefix(e_ids: Sequence[int], e_chain: list[bytes], prompt: Sequence[int], p_chain: list[bytes]) -> bool:
    """``e_ids`` is a strict prefix of ``prompt`` (whole blocks by digest, then the tail's tokens)."""

    n = len(e_ids)
    if n >= len(prompt) or n == 0:
        return False
    full = n // PAGE
    if full and (len(p_chain) < full or e_chain[full - 1] != p_chain[full - 1]):
        return False
    return list(e_ids[full * PAGE:]) == list(prompt[full * PAGE:n])


@dataclass
class Store:
    """The RAM tier: entries holding pool pages + bounded state, LRU within ``ram_bytes`` of bounded state (the
    pages are the pool's: ``spill`` frees them). ``disk``: the NVMe tier (``sessdisk.DiskTier``) or None.

    Both ranks hold a store and make the same calls in the same order (the round plan names entries by key), so
    entry ids, pool pages and LRU order agree."""

    pool: Pool
    ram_bytes: int = 512 << 20
    disk: Any = None
    entries: dict[str, Entry] = field(default_factory=dict)
    clock: int = 0
    next_id: int = 0
    stats: dict[str, int] = field(default_factory=lambda: {"saves": 0, "hits_ram": 0, "hits_disk": 0, "parks": 0,
                                                          "drops": 0, "dups": 0})

    # lookup
    def find(self, prompt: Sequence[int], tag: int) -> tuple[str, str] | None:
        """(tier, key) of the longest entry whose ids strictly prefix ``prompt`` (RAM before disk on a tie)."""

        p_chain = chain(prompt)
        best: tuple[int, int, str, str] | None = None
        for e in self.entries.values():
            if e.tag == tag and e.in_ram and is_prefix(e.ids, e.chain, prompt, p_chain):
                cand = (e.pos, 1, "ram", e.key)
                best = cand if best is None or cand[:2] > best[:2] else best
        if self.disk is not None:
            d = self.disk.find(prompt, tag, p_chain)
            if d is not None and (best is None or d[0] > best[0]):
                best = (d[0], 0, "disk", d[1])
        return None if best is None else (best[2], best[3])

    def length(self, tier: str, key: str) -> int:
        if tier == "ram":
            return self.entries[key].pos
        return self.disk.length(key)

    # saving
    def bounded_bytes(self) -> int:
        return sum(e.bounded.nbytes() for e in self.entries.values() if e.bounded is not None)

    def save(self, slot: Slot, bounded: Bounded, ids: Sequence[int], tag: int, kind: str = "turn") -> Entry | None:
        """An entry for ``slot``'s state at ``bounded.pos`` = len(ids): shares of the slot's pages below it."""

        ids = [int(t) for t in ids]
        if bounded.pos != len(ids) or not ids:
            raise ValueError(f"snapshot at {bounded.pos} for {len(ids)} ids")
        key = entry_key(tag, ids)
        self.clock += 1
        old = self.entries.get(key)
        if old is not None and old.in_ram:
            old.used = self.clock
            self.stats["dups"] += 1
            return None
        need = -(-len(ids) // PAGE)
        if len(slot.pages) < need:
            raise ValueError(f"save: the slot maps {len(slot.pages)} pages, the entry needs {need}")
        pages = list(slot.pages[:need])
        self.pool.share(pages)
        e = Entry(self.next_id, key, tag, ids, chain(ids), kind, pages, bounded, bool(old and old.disk), self.clock)
        self.next_id += 1
        self.entries[key] = e
        self.stats["saves"] += 1
        self.trim()
        return e

    def trim(self) -> list[str]:
        """Evict (park when a disk tier takes it, else drop) the least recently used entries past ``ram_bytes``."""

        done = []
        while self.bounded_bytes() > self.ram_bytes:
            victims = sorted((e for e in self.entries.values() if e.in_ram), key=lambda e: e.used)
            if len(victims) <= 1:
                break
            done.append(self.evict(victims[0].key))
        return done

    def evict(self, key: str) -> str:
        e = self.entries[key]
        if self.disk is not None and self.disk.accepts(e.pos):
            self.park(key)
            return key
        self.drop(key)
        return key

    def park(self, key: str) -> None:
        """Write a RAM entry to NVMe and let go of its pages and bounded state."""

        e = self.entries[key]
        if not e.in_ram:
            return
        if not e.disk:
            data = self.pool.read_pages(e.pages)
            self.disk.write(e.key, e.tag, e.ids, e.kind, data, e.bounded, self.pool.page)
        self.pool.drop(e.pages)
        del self.entries[key]
        self.stats["parks"] += 1

    def drop(self, key: str) -> None:
        e = self.entries.pop(key)
        if e.pages is not None:
            self.pool.drop(e.pages)
        self.stats["drops"] += 1

    def spill_order(self) -> list[str]:
        """RAM entries, coldest first (what ``spill`` would free)."""

        return [e.key for e in sorted((e for e in self.entries.values() if e.in_ram), key=lambda e: e.used)]

    def plan_spill(self, need: int, keep: Sequence[str] = ()) -> list[str] | None:
        """Which entries to let go of (coldest first, never ``keep``) so ``need`` pool pages are available to an
        admission: [] when they are already, None when even all of them are not enough. Decides only (rank 0 puts
        the keys in the round plan; both ranks ``evict`` them)."""

        avail = self.pool.available()
        if avail >= need:
            return []
        holders: dict[int, int] = {}
        for s in self.pool.slots:
            for p in s.pages:
                holders[p] = holders.get(p, 0) + 1
        for e in self.entries.values():
            if e.in_ram:
                for p in e.pages:
                    holders[p] = holders.get(p, 0) + 1
        chosen = []
        for k in self.spill_order():
            if avail >= need:
                break
            if k in keep:
                continue
            chosen.append(k)
            for p in self.entries[k].pages:
                holders[p] -= 1
                if holders[p] == 0:
                    avail += 1
        return chosen if avail >= need else None

    def spill(self, need: int, keep: Sequence[str] = ()) -> list[str] | None:
        """``plan_spill``, done here (one rank)."""

        chosen = self.plan_spill(need, keep)
        for k in chosen or []:
            self.evict(k)
        return chosen

    # resuming
    def restore(self, tier: str, key: str, slot: Slot) -> Bounded:
        """Map an entry's rows into an empty ``slot`` and return its bounded state (the forward ``restore``s it)."""

        if tier == "ram":
            e = self.entries[key]
            self.clock += 1
            e.used = self.clock
            full = e.pos // PAGE
            slot.adopt(e.pages[:full])
            if e.pos % PAGE:
                new = self.pool.take(1)
                self.pool.copy_page(e.pages[full], new[0])
                slot.pages.extend(new)
                slot._write(full, new)
            self.stats["hits_ram"] += 1
            return e.bounded
        ids, data, bounded = self.disk.read(key)
        slot.ensure(len(ids))
        need = -(-len(ids) // PAGE)
        self.pool.write_pages(slot.pages[:need], data)
        self.stats["hits_disk"] += 1
        return bounded

    def describe(self) -> str:
        ram = [e for e in self.entries.values() if e.in_ram]
        pages = len({p for e in ram for p in e.pages})
        s = f"{len(ram)} RAM entries ({pages} pool pages, {self.bounded_bytes() / 2**20:.1f} MiB bounded state)"
        if self.disk is not None:
            s += f"; {self.disk.describe()}"
        return s
