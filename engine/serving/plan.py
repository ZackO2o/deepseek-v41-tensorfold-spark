# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""A round plan of the V4.1 batcher: what rank 0 decided for one round and both ranks execute in the same order
(GLM ``batchplan``'s protocol: rank 0 plans, shares, both execute; the plan is the only thing rank 1 learns).

Execution order (``rounds.Executor.run``), every step on both ranks:

1. ``commits``  the previous round's windows: (slot, accepted, bonus) -> ``Forward.commit``; the slot's history
                grows by the kept rows; rank 1's grammar follows the chosen tokens;
2. ``finishes`` (slot, save, cancelled): a turn-end snapshot into the session store when ``save``, then the slot's
                pages are released;
3. ``spills``   session entries let go of (parked to NVMe or dropped) so admissions find their pages;
4. ``admits``   a request in a slot: page reservation, ``Forward.reset``, a session restore (RAM or NVMe) when
                ``tier`` is set (``cached`` prompt tokens skipped), the request's grammar;
5. ``pieces``   prefill pieces (slot, start, end) of several slots in one ``Forward.prefill`` (GLM 0560);
6. ``saves``    prompt snapshots at the replay point (the slot sits at it after its piece);
7. ``finals``   ``Forward.finish_prompt`` for slots whose prompt is done (CED replay);
8. ``windows``  (slot, start, pending, depth, drafts): drafts from ``Forward.draft`` when depth > 0 (DSpark, both
                ranks) or the plan's own (host lookup drafts), cut by the slot's grammar, masks filled, one
                ``Forward.window``.

Encoding: a list of ints for the communicator (GLM ``_share``): [MAGIC, json bytes, *json words (3 bytes an int),
prompt count, (slot, n, *ids) ...]. Prompts travel once, at admission, as raw ids.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .protocol import pack_bytes, unpack_bytes

MAGIC = 0x0D541
VERSION = 1


@dataclass
class Admit:
    slot: int
    prompt: list[int]
    max_tokens: int
    tier: str = ""                  # "" | ram | disk
    key: str = ""
    cached: int = 0
    quota: int = 0                  # pool pages reserved for the request
    grammar: list[int] | None = None   # GLM ``grammar.pack(bound)`` (rank 1 compiles it)
    job: int = 0


@dataclass
class WindowSpec:
    slot: int
    start: int
    pending: int
    depth: int = 0                  # DSpark depth (Forward.draft on both ranks)
    drafts: list[int] = field(default_factory=list)    # host drafts (lookup), used when depth == 0


@dataclass
class Plan:
    round: int = 0
    mode: str = "full"
    count: int = 1                  # candidates a verify row
    commits: list[tuple[int, int, int]] = field(default_factory=list)
    finishes: list[tuple[int, bool, bool]] = field(default_factory=list)
    spills: list[str] = field(default_factory=list)
    admits: list[Admit] = field(default_factory=list)
    pieces: list[tuple[int, int, int]] = field(default_factory=list)
    saves: list[int] = field(default_factory=list)
    finals: list[int] = field(default_factory=list)
    windows: list[WindowSpec] = field(default_factory=list)
    stop: bool = False

    def empty(self) -> bool:
        return not (self.commits or self.finishes or self.spills or self.admits or self.pieces or self.saves
                    or self.finals or self.windows or self.stop)

    def encode(self) -> list[int]:
        d = asdict(self)
        prompts = [(a["slot"], a.pop("prompt")) for a in d["admits"]]
        data = json.dumps(d, separators=(",", ":")).encode()
        words = pack_bytes(data)
        out = [MAGIC, VERSION, len(data), len(words), *words, len(prompts)]
        for slot, ids in prompts:
            out += [int(slot), len(ids), *(int(t) for t in ids)]
        return out

    @classmethod
    def decode(cls, ints: list[int]) -> Plan:
        ints = [int(v) for v in ints]
        if len(ints) < 5 or ints[0] != MAGIC or ints[1] != VERSION:
            raise ValueError("not a V4.1 round plan (or another version)")
        n, nw = ints[2], ints[3]
        d = json.loads(unpack_bytes(ints[4:4 + nw], n))
        i = 4 + nw
        count = ints[i]
        i += 1
        prompts = {}
        for _ in range(count):
            slot, k = ints[i], ints[i + 1]
            prompts[slot] = ints[i + 2:i + 2 + k]
            i += 2 + k
        if i != len(ints):
            raise ValueError("round plan: trailing ints")
        admits = [Admit(prompt=prompts[a["slot"]], **a) for a in d.pop("admits")]
        windows = [WindowSpec(**w) for w in d.pop("windows")]
        p = cls(**d)
        p.admits, p.windows = admits, windows
        p.commits = [tuple(c) for c in p.commits]
        p.finishes = [tuple(f) for f in p.finishes]
        p.pieces = [tuple(x) for x in p.pieces]
        return p


def piece_end(done: int, target: int, budget: int, grid: int) -> int:
    """The end of a slot's next prefill piece from ``done`` toward ``target`` with ``budget`` rows: ``target`` when
    it fits, else the last grid point within the budget (at least one grid step: a piece is never empty)."""

    if target - done <= budget:
        return target
    end = (done + budget) // grid * grid
    if end <= done:
        end = min(target, (done // grid + 1) * grid)
    return end
