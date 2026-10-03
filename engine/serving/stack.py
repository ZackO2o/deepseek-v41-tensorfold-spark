# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""The serving stack of a V4.1 rank in one call (M1's ``Dsv41Engine.__init__`` uses it): the KV pool, the session
store (RAM tier + NVMe tier), the grammar host, the memory floor and the batcher, sized from the engine's options and
the ``TF_DSV41_*`` knobs, the same way on both ranks.

    from .stack import build
    self.batch, self.grammar_host = build(fwd, topo, rank=rank, link=link, model_dir=model_dir, eos=self.eos,
                                          slots=parallel, context=self.limit, device=dev, image=code_digest)
    # rank 0: generate() -> self.batch.generate_request(prompt, max_tokens, sampling, on_tokens, draft,
    #                                                    request=self.request, host=self.grammar_host)
    # rank 1: follow() -> self.batch.follow()

Knobs: TF_DSV41_POOL_TOKENS (default slots x (context + 64), whole pages), TF_DSV41_INDEX_KV (bf16 | fp8),
TF_DSV41_PREFILL (full | replay), TF_DSV41_SESSIONS (1; 0: no session store), TF_DSV41_SESSION_RAM_MIB (256: the RAM
tier's bounded state; its pages live in the pool), TF_DSV41_SESSION_DISK / _GIB / _MIN (the NVMe tier),
TF_DSV41_FLOOR_GIB / _HARD_GIB (5 / 4), TF_DSV41_PREFILL_ROWS (2048), TF_DSV41_DRAFT_DEPTH (3), TF_DSV41_PACK (mia29:
the load-time budget's weights), TF_DSV41_GRAMMAR / _TOOL_GRAMMAR.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from . import memory, sessdisk
from . import pool as pool_mod
from .batch import Batcher
from .sessions import Store, Tag, compat_extra
from .structured import Host

BYTE_KNOBS = ("TF_DSV41_PREFILL", "TF_DSV41_INDEX_KV")      # knobs that change what an NVMe entry's bytes mean


def mode_from_env() -> str:
    mode = os.environ.get("TF_DSV41_PREFILL", "full").strip() or "full"
    if mode not in ("full", "replay"):
        raise ValueError(f"TF_DSV41_PREFILL={mode!r}: expected full or replay")
    return mode


def build(fwd, topo, *, rank: int, link, model_dir, eos: Sequence[int], slots: int, context: int,
          device: Any = "cuda", image: str = "", quiet: bool = False) -> tuple[Batcher, Host]:
    mode = mode_from_env()
    index_kv = os.environ.get("TF_DSV41_INDEX_KV", "bf16").strip() or "bf16"
    try:
        memory.load_check(topo, os.environ.get("TF_DSV41_PACK", "mia29"), rank=rank, streams=slots, context=context,
                          index_kv=index_kv,
                          session_ram_gib=int(os.environ.get("TF_DSV41_SESSION_RAM_MIB", "256") or 256) / 1024)
    except KeyError:
        pass                                            # an unknown pack name: no budget to check against
    pool = pool_mod.Pool(topo, pool_mod.settings(slots, context), index_kv=index_kv, device=device)
    tag = Tag(ced=mode, kv="fp8")
    store = None
    if (os.environ.get("TF_DSV41_SESSIONS", "1").strip() or "1") != "0":
        ident = sessdisk.compat_ident(image=image, knobs={k: os.environ.get(k, "") for k in BYTE_KNOBS},
                                      layout=compat_extra(tag, topo), extra={"page": pool.page})
        disk = sessdisk.from_env(rank, ident)
        store = Store(pool, ram_bytes=int(os.environ.get("TF_DSV41_SESSION_RAM_MIB", "256") or 256) << 20, disk=disk)
    host = Host(model_dir, topo.vocab, eos, quiet=quiet or rank != 0)
    floor = memory.Floor.from_env() if rank == 0 else None
    batch = Batcher(fwd, pool, n_slots=slots, capacity=context + pool.slack, eos=eos, store=store, link=link,
                    rank=rank, grammars=host.grammars, floor=floor, mode=mode, tag=tag.code(), start=rank == 0)
    if not quiet and rank == 0:
        print(f"[tensorfold] serving: {batch.describe()}; prefill {mode}; floor {floor.target_gib if floor else '-'}"
              f" GiB", flush=True)
    return batch, host
