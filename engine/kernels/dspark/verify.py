"""Exact keyed verification of a DSpark round (ENGINE-PLAN 7): why drafted == serial, and the host step that keeps it.

Serial decoding at position p: the target's logits row for the context t_0 .. t_{p-1} gives the token
t_p = choose(row, seed, p) (``exact_sampling.choose_rows``: keyed Gumbel over (seed, position, id); greedy = the first
of (value desc, id asc)). A round with pending token t_P (at position P) and drafts d_1 .. d_k verifies the window
w = [t_P, d_1, .., d_k] in one forward at positions P .. P + k:

1. row i of the window is computed from the context t_0 .. t_P, d_1 .. d_i only (causal attention, and every kernel
   row-invariant: a row's bits do not depend on the window's other rows or their count: csa2, mhc, engram, router,
   experts, the EXL3 linears). So whenever d_1 .. d_i equal the serial tokens t_{P+1} .. t_{P+i}, row i IS the serial
   step at position P + i, bit for bit;
2. its choice c_i = choose(row_i, seed, P + i + 1) is then the serial token t_{P+i+1};
3. ``accept`` keeps d_{i+1} only while d_{i+1} == c_i, and emits c_0 .. c_a (a = the drafts kept): by induction on i
   every emitted token is the serial token at its position, and every later row (after the first mismatch) is
   discarded, whatever it computed.

Nothing about the drafts enters the emitted tokens: not the drafter's numerics, its candidates, its noise, its depth
or its acceptance. So DSpark's draft side (``chain``) is free to approximate (candidate-only Markov bias, libdevice
log); the keyed draft noise (the same uniform at the same position as the target's choice) only raises acceptance.
The Engram rows of rejected drafts are read and discarded (rows are keyed by token ids, never by state), and the
DSpark context rings are refreshed from the committed rows' taps only.
"""

from __future__ import annotations

from typing import Sequence


def accept(window: Sequence[int], chosen: Sequence[int]) -> tuple[int, list[int]]:
    """window = [pending, d_1 .. d_k], chosen = the target's keyed choice after each window row (k + 1 of them) ->
    (drafts kept a, the a + 1 tokens the round emits)."""

    a = 0
    for d, c in zip(window[1:], chosen):
        if int(d) != int(c):
            break
        a += 1
    return a, [int(c) for c in chosen[:a + 1]]


def positions(p: int, k: int) -> list[int]:
    """The keyed positions of a window's choices: row i chooses the token at P + i + 1."""

    return [p + 1 + i for i in range(k + 1)]
