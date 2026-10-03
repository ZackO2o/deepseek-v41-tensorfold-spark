#!/usr/bin/env python3
"""One line a router of router_check's report (TF_DSV41_ROUTER_CHECK): G6.sh router gate."""
import json
import sys

try:
    d = json.load(open(sys.argv[1]))["by_e"]
except (OSError, ValueError, KeyError) as e:
    print(f"no router check report ({e})")
    sys.exit(0)
for e, b in sorted(d.items()):
    m = sorted(b.get("diff_f64_margins", []))
    print(f"E{e}: {b['calls']} calls, {b['rows']} rows, order {b['agree_order']:.6f}, set {b['agree_set']:.6f}; "
          f"{b['diff_rows']} differ (fused == f64 {b['diff_fused_eq_f64']}, split == f64 {b['diff_split_eq_f64']}, "
          f"neither {b['diff_neither_eq_f64']}; f64 margins min / median {m[0] if m else '-'} / "
          f"{m[len(m) // 2] if m else '-'}); max |logit diff| {b['max_logit_diff']:.2e}, max |w diff| "
          f"{b['max_w_diff_same_set']:.2e}", end="; ")
print()
