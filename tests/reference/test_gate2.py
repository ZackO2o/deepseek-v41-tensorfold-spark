"""gate2.py: the kit's greedy capture parsed from token-id logprobs, turned into an oracle file, and scored with a
verdict at the first divergence (identical / a kit tie / real)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate2  # noqa: E402


def _cap():
    # prompt 0: identical; prompt 1: diverges at 1 on a kit tie (ours = kit #2, margin 0.0625);
    # prompt 2: diverges at 0 where ours is the kit's #3 (real)
    tops = [[[5, -0.5], [372, -0.5625], [9, -3.0]], [[1, -0.1], [0, -2.0]], [[0, -0.2], [7, -1.0]]]
    return {"meta": {}, "prompts": [
        {"prompt": [0, 11, 12], "reply": [5, 1, 0], "top": tops},
        {"prompt": [0, 13], "reply": [5, 1, 0], "top": tops},
        {"prompt": [0, 14], "reply": [5, 1, 0], "top": tops},
    ]}


def test_parse_choice_token_ids():
    choice = {"logprobs": {"tokens": ["token_id:5", "token_id:1"],
                           "top_logprobs": [{"token_id:372": -0.6, "token_id:5": -0.5}, {"token_id:1": -0.1}]}}
    reply, tops = gate2.parse_choice(choice)
    assert reply == [5, 1] and tops[0] == [[5, -0.5], [372, -0.6]] and tops[1] == [[1, -0.1]]


def test_oracle_positions_line_up_with_the_reply():
    o = gate2.to_oracle(_cap())
    p = o["prompts"][0]
    assert p["ids"] == [0, 11, 12, 5, 1, 0]
    assert p["positions"][:3] == [None, None, None]
    assert p["positions"][3]["top"][0][0] == 5 and p["positions"][4]["token"] == 1
    assert p["positions"][5]["token_logprob"] == -0.2


def test_verdicts():
    cap = _cap()
    assert gate2.verdict(cap["prompts"][0], [5, 1, 0])["verdict"] == "identical"
    r = gate2.verdict(cap["prompts"][1], [5, 0, 0])
    assert r["verdict"] == "real" and r["first_divergence"] == 1 and r["ours_kit_rank"] == 2   # margin 1.9: no tie
    r = gate2.verdict(cap["prompts"][1], [372, 1, 0])
    assert r["verdict"] == "tie" and r["kit_margin"] == 0.0625 and r["ours_kit_gap"] == 0.0625
    r = gate2.verdict(cap["prompts"][2], [9, 1, 0])
    assert r["verdict"] == "real" and r["ours_kit_rank"] == 3


def test_score_and_forced(tmp_path):
    cap = tmp_path / "cap.json"
    cap.write_text(json.dumps(_cap()))
    ours = tmp_path / "ours.json"
    ours.write_text(json.dumps({"prompts": [{"greedy_reply": [5, 1, 0]}, {"greedy_reply": [372, 1, 0]},
                                            {"greedy_reply": [9, 1, 0]}]}))
    forced = tmp_path / "forced.json"
    forced.write_text(json.dumps({"prompts": [{"misses": []}, {"misses": [[2, 5, 372, 0.0625, -0.6]]},
                                              {"misses": [[1, 13, 4, 2.0, -3.0], [2, 5, 9, 2.5, -3.0]]}]}))
    out = tmp_path / "score.json"
    rc = gate2.main(["score", "--capture", str(cap), "--ours", str(ours), "--forced", str(forced), "--need", "1",
                     "--out", str(out)])
    s = json.loads(out.read_text())
    assert rc == 0 and s["identical"] == 1 and s["identical_or_tie"] == 2
    assert [r["misses"] for r in s["forced"]] == [0, 1, 1]          # prompt-position misses are not counted
    assert s["forced_top1"] == round(1 - 2 / 9, 4) and s["forced_top1_without_ties"] == round(1 - 1 / 9, 4)
