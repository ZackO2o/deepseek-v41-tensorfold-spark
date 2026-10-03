#!/usr/bin/env python3
"""Compile every CSA2 / router Triton kernel for sm_121 (GB10) WITHOUT a GPU, report registers / spills (ptxas -v
on the PTX, Triton's bundled ptxas), and print a hash of each kernel's PTX (debug lines stripped): the GLM repo's
``tests/kvpool_ptx.py`` pattern. Run it before and after a change that must not move a kernel's bits:

    python tests/kernels/csa2_ptx.py > /tmp/ptx-before.json
    python tests/kernels/csa2_ptx.py --against /tmp/ptx-before.json

``test_csa2_compile.py`` runs the same compiles under pytest with the structural checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

if __name__ == "__main__":
    os.environ.pop("TRITON_INTERPRET", None)      # compile, not interpret

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

from engine.kernels import router  # noqa: E402
from engine.kernels.csa2 import attn, compress, index  # noqa: E402

TGT = GPUTarget("cuda", 121, 32)


def strip(ptx: str) -> str:
    out, skip = [], False
    for ln in ptx.splitlines():
        s = ln.strip()
        if s.startswith(".section") and "debug" in s:
            skip = True
        if skip:
            if s == "}":
                skip = False
            continue
        if s.startswith((".loc", ".file", "//", "$L__tmp")) or not s:      # debug lines and their labels
            continue
        out.append(ln)
    return "\n".join(out)


def comp(fn, sig: dict, cst: dict, warps: int = 4, stages: int = 1) -> str:
    sig = dict(sig)
    cst = dict(cst)
    names = [p.name for p in fn.params]
    if "PT" in names and "PT" not in sig:
        sig["PT"] = "constexpr"
        cst.setdefault("PT", None)
        cst.setdefault("PSH", 0)
    for k in cst:
        sig[k] = "constexpr"
    src = ASTSource(fn=fn, signature=sig, constexprs=cst)
    k = triton.compile(src, target=TGT, options={"num_warps": warps, "num_stages": stages})
    return k.asm["ptx"]


def cases() -> dict:
    c = {}
    for ratio in (1, 2):
        c[f"pool_norm_r{ratio}"] = (compress._pool_norm, {"BUF": "*fp32", "b_stride": "i32", "W": "*bf16",
                                                          "LAT": "*bf16", "POS": "*i32", "n": "i32"},
                                    {"EPS": compress.EPS, "RATIO": ratio}, 4)
    for ratio in (0, 1, 2):
        for paged in ((False, True) if ratio else (False,)):
            sig = {"LAT": "*bf16", "l_stride": "i32", "CS": "*fp32", "cs_stride": "i32", "V": "*u8", "S": "*u8",
                   "POS": "*i32"}
            cst = {"RATIO": ratio, "RING": 256}
            if paged:
                sig["PT"] = "*i32"
                cst["PSH"] = 6
            c[f"kv_store_r{ratio}{'_paged' if paged else ''}"] = (compress._kv_store, sig, cst, 4)
    c["index_k_r2"] = (compress._index_k, {"KP": "*bf16", "k_stride": "i32", "W": "*bf16", "CS": "*fp32",
                                           "cs_stride": "i32", "IK": "*bf16", "POS": "*i32"},
                       {"EPS": compress.EPS, "RATIO": 2}, 1)
    for gather in (False, True):
        for paged in (False, True):
            sig = {"QI": "*bf16", "W": "*fp32", "w_stride": "i32", "IK": "*bf16", "OUT": "*fp32", "POS": "*i32",
                   "KEYS": "*i32", "k_stride": "i32", "NK": "i32", "o_stride": "i32"}
            cst = {"RATIO": 1, "H": 32, "D": 128, "BP": index.BP, "WS": 32 ** -0.5, "SCALE": 128 ** -0.5,
                   "GATHER": gather}
            if paged:
                sig["PT"] = "*i32"
                cst["PSH"] = 8
            c[f"scores{'_gather' if gather else ''}{'_paged' if paged else ''}"] = (index._scores, sig, cst,
                                                                                    index.SCORE_WARPS)
    c["keys"] = (index._keys, {"S": "*fp32", "s_stride": "i32", "K": "*i64", "k_stride": "i32", "POSN": "*i32",
                               "p_stride": "i32",
                               "NK": "i32"}, {"BLOCK": 1024, "GATHER": True}, 4)
    c["block_keys"] = (index._block_keys, {"S": "*fp32", "s_stride": "i32", "K": "*i64", "k_stride": "i32",
                                           "POS": "*i32", "NB": "i32"},
                       {"RATIO": 1, "BS": 8, "TB": 256}, 4)
    for paged in (False, True):
        for hi in (False, True):
            sig = {"Q": "*bf16", "CV": "*u8", "CSC": "*u8", "TOK": "*i32", "t_stride": "i32", "CNT": "*i32",
                   "SV": "*u8", "SSC": "*u8", "LO": "*i32", "HI": "*i32", "POS": "*i32", "PO": "*fp32",
                   "PM": "*fp32", "PL": "*fp32", "R": "i32"}
            cst = {"H": 32, "CH": attn.CH, "NCOMP": attn.NCOMP, "KT": attn.KT, "BMQ": attn.BMQ, "SCALE": 512 ** -0.5,
                   "RING": 256, "WINDOW": attn.WINDOW, "HAS_HI": hi}
            if paged:
                sig["PT"] = "*i32"
                cst["PSH"] = 6
            c[f"attn_chunks{'_hi' if hi else ''}{'_paged' if paged else ''}"] = (attn._chunks, sig, cst, attn.WARPS)
    c["attn_merge"] = (attn._merge, {"PO": "*fp32", "PM": "*fp32", "PL": "*fp32", "SINK": "*fp32", "CS": "*fp32",
                                     "cs_stride": "i32", "POS": "*i32", "OUT": "*bf16", "R": "i32"},
                       {"H": 32, "NCH": attn.NCOMP + 1}, 4)
    for shared in (1, 0):
        c[f"router_s{shared}"] = (router._route, {"X": "*bf16", "x_stride": "i32", "WG": "*fp16", "BIAS": "*fp16",
                                                  "PICK": "*i32", "WTS": "*fp32", "R": "i32"},
                                  {"D": router.DIMS, "E": router.EXPERTS, "K": router.TOPK, "SLOTS": router.TOPK + shared,
                                   "SCALE": router.SCALE, "BE": router.BE, "BK": router.BK}, router.WARPS)
    return c


def ptxas_info(ptx: str) -> dict:
    """Registers / spills / stack per entry from Triton's bundled ptxas (sm_121)."""

    from triton.backends.nvidia import compiler as nvc

    ptxas = nvc.get_ptxas(TGT.arch).path if hasattr(nvc, "get_ptxas") else nvc._path_to_binary("ptxas")[0]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "k.ptx"
        p.write_text(ptx)
        r = subprocess.run([ptxas, "-arch=sm_121a", "-v", str(p), "-o", str(Path(td) / "k.cubin")],
                           capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    m_regs = re.search(r"Used (\d+) registers", r.stderr)
    m_sp = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", r.stderr)
    return {"regs": int(m_regs.group(1)) if m_regs else -1,
            "spill": (int(m_sp.group(1)) + int(m_sp.group(2))) if m_sp else 0}


def run() -> dict:
    out = {}
    for name, (fn, sig, cst, warps) in cases().items():
        ptx = comp(fn, sig, cst, warps)
        info = ptxas_info(ptx)
        out[name] = {"ptx": hashlib.sha256(strip(ptx).encode()).hexdigest()[:16], **info}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--against", type=Path)
    args = ap.parse_args()
    got = run()
    if args.against:
        base = json.loads(args.against.read_text())
        bad = [k for k in got if k in base and got[k]["ptx"] != base[k]["ptx"]]
        for k in sorted(got):
            print(f"{k:28s} {got[k]['ptx']} regs {got[k]['regs']:4d} spill {got[k]['spill']:4d} "
                  f"{'CHANGED' if k in bad else ('same' if k in base else 'new')}")
        return 1 if bad else 0
    print(json.dumps(got, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
