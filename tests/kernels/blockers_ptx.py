#!/usr/bin/env python3
"""Compile the mHC / Engram / DSpark / streaming top-k kernels for sm_121a (GB10) WITHOUT a GPU (Triton's bundled
ptxas), report registers / spills, and print each kernel's PTX hash (``csa2_ptx.py``'s pattern and helpers):

    python tests/kernels/blockers_ptx.py > /tmp/ptx-before.json
    python tests/kernels/blockers_ptx.py --against /tmp/ptx-before.json

Every kernel compiles with the launch options its wrapper uses (``enable_fp_fusion=False`` where the wrapper says so).
``test_blockers_compile.py`` runs the same compiles under pytest with the structural checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    os.environ.pop("TRITON_INTERPRET", None)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import triton  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

import csa2_ptx  # noqa: E402
from engine.kernels.csa2 import index, stream_topk  # noqa: E402,F401  (import the kernels before any
from engine.kernels.dspark import kernels as DK  # noqa: E402,F401   interpreter test module sets TRITON_INTERPRET)
from engine.kernels.engram import kernels as EK  # noqa: E402,F401
from engine.kernels.mhc import kernels as MK  # noqa: E402

TGT = csa2_ptx.TGT


def comp(fn, sig: dict, cst: dict, warps: int = 4, fusion: bool = True, stages: int = 1) -> str:
    sig = dict(sig)
    for k in cst:
        sig[k] = "constexpr"
    src = ASTSource(fn=fn, signature=sig, constexprs=dict(cst))
    k = triton.compile(src, target=TGT, options={"num_warps": warps, "num_stages": stages,
                                                 "enable_fp_fusion": fusion})
    return k.asm["ptx"]


def _mhc_cases() -> dict:
    c = {}
    sig = {"X": "*bf16", "x_stride": "i32", "XOUT": "*bf16", "G": "*fp32", "g_rank": "i32", "POST": "*fp32",
           "COMB": "*fp32", "PRE": "*fp32", "FN": "*fp32", "PART": "*fp32", "C": "*bf16", "TAP": "*bf16",
           "tap_stride": "i32", "R": "i32"}
    for name, post, coll, mix, tap in (("boundary", True, 2, True, False), ("boundary_tap", True, 2, True, True),
                                       ("site_entry", False, 1, True, False), ("site", False, 2, True, False),
                                       ("post_only", True, 0, False, True), ("final", True, 2, False, False)):
        cst = {"D": 5120, "NB": MK.NB, "BM": MK.BM, "BK": MK.BK, "WORLD": 2, "POST_ON": post, "COLLAPSE": coll,
               "MIX": mix, "TAP_ON": tap}
        c[f"mhc_{name}"] = (MK._site, sig, cst, MK.WARPS, False)
    fsig = {"PART": "*fp32", "BASE": "*fp32", "SCALE": "*fp32", "PRE": "*fp32", "POST": "*fp32", "COMB": "*fp32",
            "C": "*bf16", "NW": "*bf16", "OUT": "*bf16", "eps": "fp32", "hc_eps": "fp32", "post_alpha": "fp32"}
    for coef in (True, False):
        c[f"mhc_finish{'' if coef else '_norm'}"] = (MK._finish, fsig, {"D": 5120, "NB": MK.NB, "ITERS": 20,
                                                                        "COEF": coef, "BLOCK": 1024}, 4, False)
    return c


def _engram_cases() -> dict:
    from engine.kernels.engram import kernels as EK

    c = {"engram_dequant": (EK._dequant, {"RAW": "*u8", "raw_stride": "i32", "OUT": "*bf16", "o_stride": "i32",
                                          "col0": "i32"}, {"H": 12, "HP": 16}, 4, False)}
    sig = {"X": "*bf16", "x_stride": "i32", "KV": "*bf16", "kv_stride": "i32", "QK": "*fp32", "KEEP": "*fp32",
           "GATE": "*fp32", "eps": "fp32", "clamp": "fp32", "sqrt_d": "fp32"}
    c["engram_fuse"] = (EK._fuse, sig, {"D": 5120, "BK": EK.BK, "HAS_KEEP": True, "HAS_GATE": False},
                        EK.FUSE_WARPS, False)
    return c


def _dspark_cases() -> dict:
    from engine.kernels.dspark import kernels as DK

    sig = {"CAND": "*i32", "CVAL": "*fp32", "c_stride": "i32", "n_cand": "i32", "W1": "*bf16", "W2": "*bf16",
           "HID": "*bf16", "CW": "*fp32", "ANCHOR": "*i32", "IPAR": "*i64", "FPAR": "*fp32", "DRAFT": "*i32",
           "CONF": "*fp32", "SCR": "*fp64"}
    return {f"dspark_chain{'' if conf else '_noconf'}": (
        DK._chain, sig, {"D": 5120, "N": 5, "KP": DK.KP, "CH": DK.CH, "R": DK.RANK, "RB": DK.RB, "HB": 1024,
                         "HAS_CONF": conf},
        DK.WARPS, True) for conf in (True, False)}


def _topk_cases() -> dict:
    from engine.kernels.csa2 import index
    from engine.kernels.csa2 import stream_topk as ST

    c = {}
    sig = {"QI": "*bf16", "W": "*fp32", "w_stride": "i32", "IK": "*bf16", "POS": "*i32", "KEYS": "*i32",
           "k_stride": "i32", "NK": "i32", "BUF": "*i64", "nsplit": "i32"}
    for mode, name, k, split in ((0, "select", 512, ST.SPLIT), (1, "reindex", 512, ST.SPLIT),
                                 (2, "blocks", 2048, ST.SPLIT_BLOCKS)):
        for paged in (False, True):
            sg = dict(sig)
            cst = {"RATIO": 1, "H": 32, "D": 128, "BP": index.BP, "WS": 32 ** -0.5, "SCALE": 128 ** -0.5,
                   "MODE": mode, "SPLIT": split, "K": k, "CAP": 2 * k, "BS": 8}
            if paged:
                sg["PT"] = "*i32"
                cst["PSH"] = 8
            else:
                cst["PT"] = None
                cst["PSH"] = 0
            c[f"stream_{name}{'_paged' if paged else ''}"] = (ST._stream, sg, cst, index.SCORE_WARPS, True)
    return c


def cases() -> dict:
    c = {}
    c.update(_topk_cases())
    c.update(_dspark_cases())
    c.update(_mhc_cases())
    c.update(_engram_cases())
    return c


def run() -> dict:
    out = {}
    for name, (fn, sig, cst, warps, fusion) in cases().items():
        ptx = comp(fn, sig, cst, warps, fusion)
        info = csa2_ptx.ptxas_info(ptx)
        out[name] = {"ptx": hashlib.sha256(csa2_ptx.strip(ptx).encode()).hexdigest()[:16], **info}
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
