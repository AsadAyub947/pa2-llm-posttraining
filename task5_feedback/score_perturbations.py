"""Task 5 Step 2 - reward sensitivity / robustness on the Controlled Reward Diagnostic Set.

Controlled pairs per problem (better response first):
  reasoning   : clean_correct  vs corrupt_reasoning_correct_final  (same correct final; S_reason)
  outcome     : clean_correct  vs good_reasoning_wrong_final       (reasoning ~fixed, final changed; S_outcome)
  distractor  : clean_correct  vs gold_distractor_wrong_final      (gold number present but wrong final)
  filler      : clean_correct  vs persuasive_filler_correct        (same answer + irrelevant persuasion)
For each mechanism (exact verifier, pairwise AI judge) and pair type we report better-response
rate, tie rate and wrong-preference rate. S_reason = better rate on `reasoning`;
S_outcome = better rate on outcome-changing pairs (outcome + distractor; also each separately).

Extra diagnostics:
  * conflict pair corrupt_reasoning_correct_final vs good_reasoning_wrong_final (which signal wins:
    outcome or reasoning?),
  * pointwise reward per variant: verifier reward, and the direct-RLAIF group reward (normalised
    pairwise win rate) from a 5-way round robin over the five variants - covers all 5 categories,
  * judge order-consistency: each pair is also judged in swapped order (the released judge picks
    a hash-dependent orientation, so this measures position/presentation noise),
  * verifier check against the dataset's expected_exact_reward.
"""
from __future__ import annotations

import argparse
import itertools
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
VARIANT_ORDER = ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
                 "persuasive_filler_correct", "gold_distractor_wrong_final"]
PAIRS = {
    "reasoning": ("clean_correct", "corrupt_reasoning_correct_final"),
    "outcome": ("clean_correct", "good_reasoning_wrong_final"),
    "distractor": ("clean_correct", "gold_distractor_wrong_final"),
    "filler": ("clean_correct", "persuasive_filler_correct"),
}
CONFLICT = ("corrupt_reasoning_correct_final", "good_reasoning_wrong_final")


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def verifier_pref(a, b, gold):
    ra, rb = exact_reward(a, gold), exact_reward(b, gold)
    return "A" if ra > rb else "B" if rb > ra else "TIE"


def rates(prefs):
    p = np.array(prefs)
    return {"n": int(p.size), "better": float(np.mean(p == "A")), "tie": float(np.mean(p == "TIE")),
            "wrong": float(np.mean(p == "B"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--no-swap-check", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    judge = None if args.skip_judge else PairwiseAIJudge(cfg, repo_path(cfg["results_dir"]) / "task5_feedback" / "judge_cache.json")
    pair_recs, point_recs = [], []
    n_calls, judge_time = 0, 0.0
    verifier_mismatch = []
    for pid, v in sorted(groups.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        gold = str(v["clean_correct"]["gold_final"])
        q = v["clean_correct"]["question"]
        for name in VARIANT_ORDER:
            r = exact_reward(v[name]["response"], gold)
            exp = v[name].get("expected_exact_reward")
            if exp is not None and float(exp) != r:
                verifier_mismatch.append({"problem_id": pid, "variant": name, "expected": exp, "verifier": r})
        for ptype, (a, b) in list(PAIRS.items()) + [("conflict", CONFLICT)]:
            ya, yb = v[a]["response"], v[b]["response"]
            rec = {"problem_id": pid, "pair_type": ptype, "a": a, "b": b, "verifier": verifier_pref(ya, yb, gold),
                   "words_a": len(ya.split()), "words_b": len(yb.split())}
            if judge is not None:
                t0 = time.perf_counter()
                rec["judge"] = judge.compare(q, ya, yb)
                if not args.no_swap_check:
                    sw = judge.compare(q, yb, ya)
                    rec["judge_swapped"] = {"A": "B", "B": "A", "TIE": "TIE"}[sw]
                    rec["judge_order_consistent"] = rec["judge"] == rec["judge_swapped"]
                judge_time += time.perf_counter() - t0
                n_calls += 2 if not args.no_swap_check else 1
            pair_recs.append(rec)
        # pointwise: verifier reward + direct-RLAIF group reward over the 5 variants
        resp = [v[n]["response"] for n in VARIANT_ORDER]
        grp = judge.group_rewards(q, resp) if judge is not None else [float("nan")] * 5
        for n, y, gr in zip(VARIANT_ORDER, resp, grp):
            point_recs.append({"problem_id": pid, "variant": n, "verifier_reward": exact_reward(y, gold),
                               "rlaif_group_reward": gr, "words": len(y.split())})

    write_jsonl(outdir / "pair_judgements.jsonl", pair_recs)
    write_jsonl(outdir / "pointwise_rewards.jsonl", point_recs)
    df = pd.DataFrame(pair_recs)
    table = []
    for mech in ["verifier"] + (["judge"] if judge is not None else []):
        for ptype in list(PAIRS) + ["conflict"]:
            sub = df[df.pair_type == ptype][mech].tolist()
            table.append({"mechanism": mech, "pair_type": ptype, **rates(sub)})
    tdf = pd.DataFrame(table)
    tdf.to_csv(outdir / "pair_rates.csv", index=False)

    def better(mech, types):
        sub = df[df.pair_type.isin(types)][mech]
        return float((sub == "A").mean())

    sens = {}
    for mech in ["verifier"] + (["judge"] if judge is not None else []):
        sens[mech] = {
            "S_reason": better(mech, ["reasoning"]),
            "S_outcome": better(mech, ["outcome", "distractor"]),
            "S_outcome_good_reasoning_wrong_final": better(mech, ["outcome"]),
            "S_outcome_gold_distractor": better(mech, ["distractor"]),
            "filler_resistance": better(mech, ["filler"]),
            "filler_preferred_rate": float((df[df.pair_type == "filler"][mech] == "B").mean()),
            "conflict_prefers_correct_final": float((df[df.pair_type == "conflict"][mech] == "A").mean()),
            "conflict_prefers_good_reasoning": float((df[df.pair_type == "conflict"][mech] == "B").mean()),
        }
    pdf = pd.DataFrame(point_recs)
    pointwise = pdf.groupby("variant")[["verifier_reward", "rlaif_group_reward", "words"]].mean().reindex(VARIANT_ORDER)
    pointwise.to_csv(outdir / "pointwise_by_variant.csv")
    res = {
        "n_problems": len(groups), "pair_rates": table, "sensitivity": sens,
        "pointwise_by_variant": pointwise.reset_index().to_dict(orient="records"),
        "verifier_vs_expected_mismatches": verifier_mismatch,
        "judge_calls": n_calls, "judge_seconds_total": judge_time,
        "judge_seconds_per_call": judge_time / n_calls if n_calls else None,
    }
    if judge is not None and not args.no_swap_check:
        res["judge_order_consistency"] = float(df["judge_order_consistent"].mean())
        res["judge_order_consistency_by_pair"] = df.groupby("pair_type")["judge_order_consistent"].mean().to_dict()
    save_json(outdir / "summary.json", res)
    print(tdf.to_string(index=False))
    print("Sensitivity:", sens)
    print(pointwise.to_string())
    if verifier_mismatch:
        print(f"WARNING: verifier disagrees with expected_exact_reward on {len(verifier_mismatch)} rows")


if __name__ == "__main__":
    main()
