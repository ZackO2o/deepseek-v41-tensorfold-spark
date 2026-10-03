# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""A request slot's bounded state for DeepSeek-V4.1-Flash (the analogue of the GLM Spark engine's ``forward.State``,
without KDA): what lives outside the paged pool, what a snapshot stores, and what it costs.

Per slot:

- ``swa``: one ring a layer (40 + 3 DSpark blocks) of ``RING`` FP8 rows (``csa2.rows``), position p at p % RING.
  RING = 256 >= 127 + the widest verify window (16 rows), so a window's rows are written before any of them attends
  and never overwrite a row another window row still reads;
- ``carry``: the last fp32 [kv | score] projection row (1,024 wide) of each ratio-2 kv source (layers 2, 8, 14): a
  group closing at the window's first position pools it (``csa2.compress.pool_norm``);
- ``engram``: the last 3 token ids (n-gram orders 2-4 look back 3), as the hashes see them;
- ``taps``: DSpark's context, the mHC outputs of layers 37-39 for the last 128 positions (bf16, 3 x 5,120 a
  position after the stream mean: vLLM ``nvidia/dspark.py``), its own SWA rings are in ``swa``;
- ``pages``: the slot's ``pool.Slot`` in the KV pool.

Snapshots (``Bounded``, ``protocol.py``) hold the rings' last ``window`` rows in position order (every ring in
``full`` mode; the encoder rings only for a ``replay``-mode prompt snapshot: the decoder rings are rebuilt from the
prompt's tail, ENGINE-PLAN section 8), the carries when the position is odd (a turn's end; prompt snapshots sit on
the 16-token grid, where no ratio-2 group is open), the lookback and the taps.

``SlotState`` is the reference implementation the forward may use as is (torch tensors on the rank's device; CPU in
tests): ``write_ring`` / ``push`` / ``snapshot`` / ``restore`` / ``reset``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import protocol as RW  # the FP8 row record's sizes (csa2.rows: 584 = 576 + 8)
from .protocol import Bounded
from .topology import Topology

RING = 256
CARRY_WIDTH = 1024          # fp32 [kv 512 | score 512]
LOOKBACK = 3
TAP_WINDOW = 128


@dataclass(frozen=True)
class SlotBytes:
    swa: int
    carry: int
    engram: int
    taps: int

    @property
    def total(self) -> int:
        return self.swa + self.carry + self.engram + self.taps


def slot_bytes(topo: Topology) -> SlotBytes:
    rings = len(topo.layers) + len(topo.dspark)
    carries = sum(1 for ly in topo.layers if ly.role == "full" and ly.ratio == 2)
    taps = len(topo.taps) * topo.hidden * 2 * TAP_WINDOW
    return SlotBytes(rings * RING * RW.ROW_BYTES, carries * CARRY_WIDTH * 4, LOOKBACK * 4, taps)


def snapshot_state_bytes(topo: Topology) -> int:
    """Bounded state a snapshot stores besides its pool pages: the window's last 128 rows of every ring, the
    lookback and the taps (no carries: snapshots sit on even positions)."""

    rings = len(topo.layers) + len(topo.dspark)
    return rings * topo.window * RW.ROW_BYTES + LOOKBACK * 4 + len(topo.taps) * topo.hidden * 2 * TAP_WINDOW


def ring_layers(topo: Topology, ced: str) -> list[int]:
    """Ring indices a snapshot keeps: every ring (``full``), or the encoder layers' (``replay``)."""

    n = len(topo.layers) + len(topo.dspark)
    if ced == "replay":
        return [ly.index for ly in topo.layers if ly.encoder]
    return list(range(n))


class SlotState:
    """One slot's bounded state (torch tensors on ``device``)."""

    def __init__(self, topo: Topology, device="cpu", pages=None, *, window: int | None = None) -> None:
        import torch

        n = len(topo.layers) + len(topo.dspark)
        self.topo = topo
        self.window = int(window or topo.window)
        if self.window + 16 > RING:
            raise ValueError(f"SWA window {self.window} + a 16-row verify window does not fit the {RING}-row ring")
        self.pos = 0
        self.swa_values = torch.zeros((n, RING, RW.VB), dtype=torch.uint8, device=device)
        self.swa_scales = torch.zeros((n, RING, RW.SB), dtype=torch.uint8, device=device)
        self.carry_layers = [ly.index for ly in topo.layers if ly.role == "full" and ly.ratio == 2]
        self.carry = torch.zeros((len(self.carry_layers), CARRY_WIDTH), dtype=torch.float32, device=device)
        self.lookback = [-1] * LOOKBACK
        self.taps = torch.zeros((TAP_WINDOW, len(topo.taps), topo.hidden), dtype=torch.bfloat16, device=device)
        self.pages = pages                       # pool.Slot (GLM 0290 page table), None without a pool
        self._torch = torch

    def ring(self, layer: int):
        """(values, scales) of a layer's ring: the ``swa`` argument of ``csa2.attn.attention``."""

        return self.swa_values[layer], self.swa_scales[layer]

    def reset(self) -> None:
        self.pos = 0
        self.swa_values.zero_()
        self.swa_scales.zero_()
        self.carry.zero_()
        self.taps.zero_()
        self.lookback = [-1] * LOOKBACK

    def push(self, tokens) -> None:
        """The Engram lookback after ``tokens`` (committed, in order)."""

        lb = (self.lookback + [int(t) for t in tokens])[-LOOKBACK:]
        self.lookback = lb

    def _positions(self, pos: int, n: int) -> list[int]:
        return list(range(max(0, pos - n), pos))

    def snapshot(self, ced: str = "full") -> Bounded:
        """The bounded state at ``self.pos`` (host copies)."""

        torch = self._torch
        pos = self.pos
        keep = ring_layers(self.topo, ced)
        w = self.window
        idx = [p % RING for p in self._positions(pos, w)]
        pad = w - len(idx)
        sel = torch.tensor(idx, dtype=torch.long, device=self.swa_values.device)
        vals = self.swa_values[keep].index_select(1, sel).cpu().numpy()
        scs = self.swa_scales[keep].index_select(1, sel).cpu().numpy()
        if pad:
            vals = np.concatenate([np.zeros((len(keep), pad, RW.VB), np.uint8), vals], axis=1)
            scs = np.concatenate([np.zeros((len(keep), pad, RW.SB), np.uint8), scs], axis=1)
        tidx = [p % TAP_WINDOW for p in self._positions(pos, TAP_WINDOW)]
        taps = self.taps.index_select(0, torch.tensor(tidx, dtype=torch.long, device=self.taps.device))
        taps = taps.view(torch.int16).cpu().numpy()
        arrays = {"swa_values": vals, "swa_scales": scs, "taps": taps,
                  "lookback": np.asarray(self.lookback, dtype=np.int64),
                  "rings": np.asarray(keep, dtype=np.int64)}
        if pos % 2:
            arrays["carry"] = self.carry.cpu().numpy().copy()
        return Bounded(pos, arrays, {"ced": ced, "window": w})

    def restore(self, snap: Bounded) -> None:
        """Load ``snap`` (rings it lacks are left zero: the replay rule rebuilds them)."""

        torch = self._torch
        a = snap.arrays
        if int(snap.meta.get("window", self.window)) != self.window:
            raise ValueError("snapshot of another SWA window")
        self.reset()
        pos = int(snap.pos)
        self.pos = pos
        dev = self.swa_values.device
        pos_list = self._positions(pos, self.window)
        k = len(pos_list)
        if k:
            sel = torch.tensor([p % RING for p in pos_list], dtype=torch.long, device=dev)
            for j, layer in enumerate(a["rings"].tolist()):
                v = torch.from_numpy(np.ascontiguousarray(a["swa_values"][j, self.window - k:])).to(dev)
                s = torch.from_numpy(np.ascontiguousarray(a["swa_scales"][j, self.window - k:])).to(dev)
                self.swa_values[layer].index_copy_(0, sel, v)
                self.swa_scales[layer].index_copy_(0, sel, s)
        tpos = self._positions(pos, TAP_WINDOW)
        if tpos:
            taps = torch.from_numpy(np.ascontiguousarray(a["taps"])).view(torch.bfloat16).to(self.taps.device)
            self.taps.index_copy_(0, torch.tensor([p % TAP_WINDOW for p in tpos], dtype=torch.long,
                                                  device=self.taps.device), taps)
        if "carry" in a:
            self.carry.copy_(torch.from_numpy(a["carry"]).to(self.carry.device))
        self.lookback = [int(t) for t in a["lookback"].tolist()]
