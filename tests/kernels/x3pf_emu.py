"""Schedule emulator of x3pf.cu's pf_kernel against upstream's grouped_kernel (experts_grouped.cuh), no GPU.

Each kernel is transcribed at the level that decides the bits of a Z element: which program, warp and loop iteration
issues each mma into it, with which A row (the member's pair row, or 0 for a dead row) and which B tile (k tile,
column tile, n8 half), from which accumulator start, and how the warps' partials are summed into which Z address.
``trace_*`` return {Z address: (A row, ((warp, [(k tile, column tile, half), ...]), ...))} for every element a kernel
stores; equal dicts mean every stored element is the same mma chain and the same warp sum (upstream's bits).
"""

from __future__ import annotations

W = 4


def _rows(members: list[list[int]], u: int, base: int, count: int, maxm: int, slots: int) -> list[int]:
    out = []
    for i in range(count):
        m = base + i
        code = members[u][m] if m < maxm else -1
        out.append((code >> 5) * slots + (code & 31) if code >= 0 else -1)
    return out


def _chain(split: int, warp: int, KT: int, SK: int, nt0: int, i: int, h: int) -> list[tuple[int, int, int]]:
    per_split = KT // SK
    per_warp = per_split // W
    kt0 = split * per_split + warp * per_warp
    return [(kt0 + it, nt0 + i, h) for it in range(per_warp)]


def _store(z: dict, mat: int, split: int, P: int, N: int, r: int, column: int, chains: tuple, SK: int) -> None:
    addr = ((mat * SK + split) * P + r) * N + column
    assert addr not in z, ("stored twice", addr)
    z[addr] = (r, chains)


def trace_grouped(members, ucount, K, N, P, SK, slots, mats, NT) -> dict:
    """grouped_kernel<CB, NT, 4, PF, LO, HI>: program (u, n block, (mat * SK + split) * MT + member tile)."""

    maxm = len(members[0])
    MT = (maxm + 15) // 16
    KT, z = K // 16, {}
    for u in range(len(members)):
        if u >= ucount:
            continue
        for by in range(N // (16 * NT)):
            for bz in range(mats * SK * MT):
                mtile, split, mat = bz % MT, (bz // MT) % SK, bz // MT // SK
                rows = _rows(members, u, mtile * 16, 16, maxm, slots)
                if rows[0] < 0:
                    continue
                nt0 = by * NT
                for idx in range(16 * NT * 16):
                    row, col = idx // (NT * 16), idx % (NT * 16)
                    r = rows[row]
                    if r < 0:
                        continue
                    i, h = col // 16, (col % 16) // 8
                    chains = tuple((w, tuple(_chain(split, w, KT, SK, nt0, i, h))) for w in range(W))
                    _store(z, mat, split, P, N, r, nt0 * 16 + col, chains, SK)
    return z


def trace_pf(members, ucount, K, N, P, SK, slots, mats, NT, MTL) -> dict:
    """pf_kernel<CB, NT, MTL, LO, HI>: program (u, n block, (mat * SK + split) * MG + member group)."""

    maxm = len(members[0])
    MG = (maxm + 16 * MTL - 1) // (16 * MTL)
    KT, z = K // 16, {}
    for u in range(len(members)):
        if u >= ucount:
            continue
        for by in range(N // (16 * NT)):
            for bz in range(mats * SK * MG):
                mg, split, mat = bz % MG, (bz // MG) % SK, bz // MG // SK
                rows = _rows(members, u, mg * MTL * 16, MTL * 16, maxm, slots)
                if rows[0] < 0:
                    continue
                live = 0
                for mt in range(MTL):
                    if rows[mt * 16] >= 0:
                        live = mt + 1
                nt0 = by * NT
                for mt in range(MTL):
                    if mt >= live:
                        break
                    for idx in range(16 * NT * 16):
                        row, col = idx // (NT * 16), idx % (NT * 16)
                        r = rows[mt * 16 + row]
                        if r < 0:
                            continue
                        i, h = col // 16, (col % 16) // 8
                        # pf_tiles: every live tile gets the mma of (k tile, column tile i, half h) in k order
                        chains = tuple((w, tuple(_chain(split, w, KT, SK, nt0, i, h))) for w in range(W))
                        _store(z, mat, split, P, N, r, nt0 * 16 + col, chains, SK)
    return z
