"""DSpark with exact verification, on our GLM drafting interfaces (``depth.py`` cost-derived depth, ``lookup.py``
suffix drafts, ``exact_sampling`` keyed sampling; UPSTREAM-CONTRIB-PLAN I4's ``Drafter`` protocol).

DSpark (the checkpoint's ``mtp.0-2``): 3 transformer blocks (SWA 128, 128 experts top-3, 4-bit) reading the target's
streams after layers 37-39 (vLLM ``nvidia/dspark.py``: the mHC outputs, mean over the 4 streams), one pass for a
block of 5 positions seeded with the noise token 128,799, a rank-256 Markov head, and a confidence head (per-position
acceptance).

Exactness (the rule our GLM drafters keep):

- the target decides every token: row i of a verify window is the serial step at its position (row-invariant
  kernels), and its token is ``exact_sampling.choose_rows`` with the request's seed at that absolute position;
- a draft is kept only up to the first position where it differs from that keyed choice, so drafted == serial for
  any drafter, any depth, any acceptance; drafts change speed, never a reply;
- DSpark's own sampling (a T > 0 request's draft tokens) uses the same keyed noise as the target at each position,
  which raises acceptance at temperature without touching the reply (GLM's DFlash2 rule).

What vLLM / the kit do instead: rejection sampling against the draft distribution, so a T > 0 reply depends on the
drafts and on the batch. Ours is the first exact DSpark serving.

Depth: the confidence head's per-position acceptance q_i feeds ``depth.best_k(qs, verify, rate, ...)`` with the
measured verify costs C(R) (calibrated at load, cached per image); suffix lookup (0020) can extend copy-heavy rounds
past DSpark's 5 positions up to the deep-verify width (16 rows, 0380).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

BLOCK = 5
NOISE_TOKEN = 128799
MARKOV_RANK = 256
TAPS = (37, 38, 39)
MAX_ROWS = 16


@dataclass
class Proposal:
    tokens: list[int]                    # drafted tokens after the pending one
    parents: list[int]                   # tree parents (-1 = the pending token); a chain is [-1, 0, 1, ...]
    confidence: list[float] = field(default_factory=list)    # DSpark's per-position acceptance estimate
    source: str = "dspark"               # dspark | lookup


class Drafter(Protocol):
    name: str

    def propose(self, slot: int, pending: int, rows: int, sampling) -> Proposal: ...

    def cost_ms(self, rows: int) -> float: ...


def accepted(window_tokens: Sequence[int], chosen: Sequence[int]) -> int:
    """Drafts kept from a chain window: ``window_tokens`` = [pending, d1, d2, ...], ``chosen`` = the target's keyed
    choice after each row. Keep d_i while d_i == chosen[i - 1]; the round emits accepted + 1 tokens."""

    n = 0
    for d, c in zip(window_tokens[1:], chosen):
        if d != c:
            break
        n += 1
    return n


class DSpark:
    """Interface stub of the DSpark drafter on the target's ranks (TP=2 like the target: heads and experts split)."""

    name = "dspark"

    def __init__(self, engine, *, block: int = BLOCK) -> None:
        self.engine = engine
        self.block = block

    def propose(self, slot: int, pending: int, rows: int, sampling) -> Proposal:
        raise NotImplementedError("M2: one DSpark pass over the slot's taps; Markov head; keyed draft noise")

    def cost_ms(self, rows: int) -> float:
        raise NotImplementedError("M2: from the load-time calibration")
