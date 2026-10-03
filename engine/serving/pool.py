# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""The KV pool's cache families for DeepSeek-V4.1-Flash, on our GLM pool (patch 0290: ``glm5_next/spark/kvpool.py``,
``PoolBook`` / ``SlotPages`` / ``Paged``; upstream PR 3's ``PagedPool``).

What is paged (it grows with the context; shared by the request slots, admitted by pages):

| family          | layers          | rows a token | row bytes                         |
| --- | --- | --- | --- |
| ``comp``        | kv sources 2/8/14 (ratio 2), 20 (ratio 1) | 1 / ratio | 576 values + 8 scales (``csa2.rows``) |
| ``index_k``     | the same four   | 1 / ratio    | 256 (bf16 x 128) or 132 (``TF_DSV41_INDEX_KV=fp8``, later) |

A page holds ``page`` tokens of every family (rows per page = page / ratio), so one page table a slot serves every
kernel (``rows.prow``: PT / PSH per family, PSH = log2 rows a page). Pages are aligned to 16 tokens: the snapshot
grid (ratio 2 x candidate block 8), so a page boundary never splits a compressed group or a candidate block.

What is not paged (bounded, per slot, ``state.SlotState``): the 40 + 3 SWA rings, the compressor carries, the Engram
lookback, the DSpark tap window.

**Shared pages** (the session store's RAM tier, GLM 0310's prefix share without copies): a session entry keeps the
pages below its length mapped; a request resuming it adopts the entry's *full* pages (read only: a slot writes only
positions >= its resume point, which lie past them) and gets a copy of the partial last page (copy on write). A page
returns to the free list when its last holder (slot or entry) lets go. Parking an entry to NVMe (``sessdisk``)
writes its pages and lets go of them: that is how the pool makes room (``batch.Batcher``'s spills).

Allocation is deterministic (lowest free page first, GLM ``PageAlloc``): both ranks make the same calls in the same
order (the round plan), so their pools stay identical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tensorfold.families.glm5_next.spark import kvpool

from . import protocol as RW  # the FP8 row record's sizes (csa2.rows: 584 = 576 + 8)
from .topology import Topology

PAGE = 256                 # tokens a page (GLM's default): 128 rows of a ratio-2 family
GRID = 16                  # snapshot / page alignment in tokens
INDEX_ROW = {"bf16": 256, "fp8": 132}
SLACK = kvpool.DEFAULT_SLACK            # tokens past prompt + max_tokens a request may write (verify windows)
POOL_ENV = "TF_DSV41_POOL_TOKENS"


@dataclass(frozen=True)
class Family:
    name: str              # "comp.2", "index_k.20", ...
    layer: int
    ratio: int
    row_bytes: int

    def rows(self, tokens: int) -> int:
        return tokens // self.ratio

    def page_rows(self, page: int = PAGE) -> int:
        return page // self.ratio

    def bytes_per_token(self) -> float:
        return self.row_bytes / self.ratio


def families(topo: Topology, index_kv: str = "bf16") -> list[Family]:
    out = []
    for ly in topo.layers:
        if ly.role != "full":
            continue
        out.append(Family(f"comp.{ly.index}", ly.index, ly.ratio, RW.ROW_BYTES))
        out.append(Family(f"index_k.{ly.index}", ly.index, ly.ratio, INDEX_ROW[index_kv]))
    return out


def bytes_per_token(topo: Topology, index_kv: str = "bf16") -> float:
    """Pool bytes one context token costs on each rank (the single KV head: every rank holds whole rows)."""

    return sum(f.bytes_per_token() for f in families(topo, index_kv))


def pool_tokens(streams: int, context: int, slack: int = SLACK, page: int = PAGE) -> int:
    """Tokens the pool must hold for ``streams`` active streams at ``context`` each (+ verify / draft slack),
    rounded up to whole pages a stream."""

    per = -(-(context + slack) // page) * page
    return streams * per


def pool_bytes(topo: Topology, streams: int, context: int, index_kv: str = "bf16", page: int = PAGE) -> int:
    toks = pool_tokens(streams, context, page=page)
    return int(sum(f.row_bytes * (toks // f.ratio) for f in families(topo, index_kv)))


def settings(streams: int, context: int) -> int:
    """TF_DSV41_POOL_TOKENS (default: ``streams`` x ``context`` + slack, whole pages a stream), a page multiple."""

    raw = os.environ.get(POOL_ENV, "").strip()
    tokens = int(raw) if raw else pool_tokens(streams, context)
    if tokens <= 0:
        raise ValueError(f"{POOL_ENV}={raw!r}: expected a positive token count")
    return -(-tokens // PAGE) * PAGE


# -- slots with shared pages ------------------------------------------------------------------------------------------
class Slot(kvpool.SlotPages):
    """A slot's page table (GLM ``SlotPages``) whose leading pages may be shared with session entries."""

    pool: Pool

    def adopt(self, pages: list[int]) -> None:
        """Map shared ``pages`` as the slot's first pages (an empty slot; the store's resume)."""

        if self.pages:
            raise ValueError("adopt: the slot already maps pages")
        if len(pages) > self.max_pages:
            raise ValueError("adopt: past the slot's capacity")
        self.pool.share(pages)
        self.pages.extend(int(p) for p in pages)
        self._write(0, list(self.pages))

    def truncate(self, keep: int) -> int:
        k = kvpool.pages_for(keep, self.page)
        if k >= len(self.pages):
            return 0
        drop = self.pages[k:]
        del self.pages[k:]
        self.pool.drop(drop)
        self._write(k, [self.pool.null] * len(drop))
        return len(drop)

    def release(self) -> int:
        self.quota = None
        self.over = 0
        return self.truncate(0)


class Pool(kvpool.PoolBook):
    """The V4.1 pool: one physical tensor a family ([(npages + 1) x page rows, row bytes]; the last page is GLM's null
    page), shared pages with holder counts, the slots' tables. ``device="meta"`` sizes it without memory (tests,
    the load-time budget)."""

    def __init__(self, topo: Topology, tokens: int, page: int = PAGE, *, index_kv: str = "bf16",
                 device: Any = "cpu", slack: int = SLACK) -> None:
        import torch

        if tokens % page:
            raise ValueError(f"pool of {tokens} tokens: not a multiple of the {page}-token page")
        super().__init__(tokens // page, page)
        self.topo = topo
        self.slack = int(slack)
        self.device = torch.device(device)
        self.fams = families(topo, index_kv)
        self.extra: dict[int, int] = {}           # page -> holders past the first
        self.phys: dict[str, Any] = {}
        for f in self.fams:
            rows = (self.npages + 1) * f.page_rows(page)
            if f.name.startswith("index_k") and index_kv == "bf16":
                self.phys[f.name] = torch.zeros((rows, f.row_bytes // 2), dtype=torch.bfloat16, device=self.device)
            else:
                self.phys[f.name] = torch.zeros((rows, f.row_bytes), dtype=torch.uint8, device=self.device)
        self._torch = torch

    # holders
    def share(self, pages) -> None:
        for p in pages:
            self.extra[int(p)] = self.extra.get(int(p), 0) + 1

    def drop(self, pages) -> None:
        back = []
        for p in pages:
            p = int(p)
            n = self.extra.get(p, 0)
            if n:
                if n == 1:
                    del self.extra[p]
                else:
                    self.extra[p] = n - 1
            else:
                back.append(p)
        if back:
            self.alloc.give(back)

    def holders(self, page: int) -> int:
        return 1 + self.extra.get(int(page), 0)

    # slots and views
    def new_slot(self, capacity: int) -> Slot:
        table = None
        if self.device.type != "meta":
            mp = kvpool.pages_for(capacity, self.page)
            table = self._torch.full((mp,), self.null, dtype=self._torch.int32, device=self.device)
        sp = Slot(self, capacity, table)
        self.slots.append(sp)
        return sp

    def family(self, name: str) -> Family:
        for f in self.fams:
            if f.name == name:
                return f
        raise KeyError(name)

    def view(self, slot: Slot, name: str) -> kvpool.Paged:
        """A family's rows as the slot sees them (GLM ``Paged``: logical row r -> its page's physical row)."""

        f = self.family(name)
        return kvpool.Paged(self.phys[name], slot, f.page_rows(self.page), slot.capacity // f.ratio)

    def take(self, k: int) -> list[int]:
        return self.alloc.take(k)

    def copy_page(self, src: int, dst: int) -> None:
        for f in self.fams:
            per = f.page_rows(self.page)
            t = self.phys[f.name]
            t[dst * per:(dst + 1) * per].copy_(t[src * per:(src + 1) * per])

    def read_pages(self, pages: list[int]) -> dict[str, Any]:
        """{family: host uint8 [len(pages) x page rows, row bytes]} (the NVMe tier's park)."""

        out = {}
        for f in self.fams:
            per = f.page_rows(self.page)
            t = self.phys[f.name]
            idx = self._torch.tensor([p * per + r for p in pages for r in range(per)], dtype=self._torch.long,
                                     device=self.device)
            rows = t.index_select(0, idx).contiguous().cpu()
            out[f.name] = rows.view(self._torch.uint8).numpy().reshape(len(idx), f.row_bytes)
        return out

    def write_pages(self, pages: list[int], data: dict[str, Any]) -> None:
        """``read_pages``' output back into ``pages`` (the NVMe tier's resume)."""

        torch = self._torch
        for f in self.fams:
            per = f.page_rows(self.page)
            t = self.phys[f.name]
            src = torch.from_numpy(data[f.name].reshape(len(pages) * per, f.row_bytes).copy()).to(self.device)
            src = src.view(t.dtype).reshape(len(pages) * per, *t.shape[1:])
            idx = torch.tensor([p * per + r for p in pages for r in range(per)], dtype=torch.long, device=self.device)
            t.index_copy_(0, idx, src)

    # accounting
    def nbytes(self) -> int:
        return int(sum(t.numel() * t.element_size() for t in self.phys.values()))

    def page_bytes(self) -> int:
        return int(sum(f.row_bytes * f.page_rows(self.page) for f in self.fams))

    def need_pages(self, prompt: int, max_tokens: int, capacity: int) -> int:
        return kvpool.pages_for(kvpool.need_tokens(prompt, max_tokens, capacity, self.slack), self.page)

    def entry_pages(self) -> int:
        """Pages held by nothing but session entries (what parking can free)."""

        mapped = {p for s in self.slots for p in s.pages}
        return self.npages - self.alloc.n_free - len(mapped)

    def null_clean(self) -> bool:
        for f in self.fams:
            per = f.page_rows(self.page)
            if bool(self.phys[f.name][self.null * per:(self.null + 1) * per].ne(0).any()):
                return False
        return True
