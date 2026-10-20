"""Task 5 - combine in-domain, diagnostic and transfer results into the final comparison tables.

Reads results/task5_feedback/{gsm,transfer,diagnostics}/summary.json and writes
results/task5_feedback/feedback_comparison.{json,md}: accuracy / win-rate / format / length per
policy and dataset, in-domain -> transfer drop, verifier-judge agreement, sensitivity table,
failure-type shifts, and the quantities used for the coverage/noise/exploitability/cost
discussion (no new model calls).
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import save_json


def _load(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    base = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm, tr, diag = _load(base / "gsm" / "summary.json"), _load(base / "transfer" / "summary.json"), _load(base / "diagnostics" / "summary.json")
    if gsm is None:
        raise SystemExit("Run: python -m task5_feedback.evaluate_math --dataset gsm")

    rows = []
    for ds_name, ds in [("GSM8K (in-domain)", gsm), ("SVAMP (transfer)", tr)]:
        if ds is None:
            continue
        for p, s in ds["policies"].items():
            pw = (ds.get("pairwise") or {}).get(f"{p}_vs_sft", {})
            rows.append({"dataset": ds_name, "policy": p, "exact_acc": s["exact_accuracy"],
                         "format": s["format_compliance"], "len_mean": s["length"]["mean"], "len_std": s["length"]["std"],
                         "trunc": s["truncation_rate"], "win_vs_sft": pw.get("win_rate"), "ties_vs_sft": pw.get("ties"),
                         "verifier_judge_agree": pw.get("agreement_all"), "judge_acc_on_decisive": pw.get("judge_agrees_on_decisive")})
    perf = pd.DataFrame(rows)

    drops = []
    if tr is not None:
        for p in gsm["policies"]:
            g, t = gsm["policies"][p], tr["policies"][p]
            d = {"policy": p, "acc_gsm": g["exact_accuracy"], "acc_svamp": t["exact_accuracy"],
                 "acc_drop": g["exact_accuracy"] - t["exact_accuracy"],
                 "format_drop": g["format_compliance"] - t["format_compliance"],
                 "len_change": t["length"]["mean"] - g["length"]["mean"]}
            if p != "sft" and gsm.get("pairwise") and tr.get("pairwise"):
                d["win_gsm"] = gsm["pairwise"][f"{p}_vs_sft"]["win_rate"]
                d["win_svamp"] = tr["pairwise"][f"{p}_vs_sft"]["win_rate"]
                d["win_drop"] = d["win_gsm"] - d["win_svamp"]
            # gain over SFT on each set (does the RL gain transfer?)
            d["gain_over_sft_gsm"] = g["exact_accuracy"] - gsm["policies"]["sft"]["exact_accuracy"]
            d["gain_over_sft_svamp"] = t["exact_accuracy"] - tr["policies"]["sft"]["exact_accuracy"]
            drops.append(d)
    drops = pd.DataFrame(drops)

    failures = []
    for ds_name, ds in [("gsm", gsm), ("transfer", tr)]:
        if ds is None:
            continue
        for p, s in ds["policies"].items():
            n = sum(s["failure_types"].values())
            failures.append({"dataset": ds_name, "policy": p, **{k: v / n for k, v in s["failure_types"].items()}})
    failures = pd.DataFrame(failures).fillna(0.0)

    discussion = {}
    if diag is not None:
        sens = diag["sensitivity"]
        discussion = {
            "coverage": {
                "verifier_tie_rate_reasoning_pairs": next((r["tie"] for r in diag["pair_rates"] if r["mechanism"] == "verifier" and r["pair_type"] == "reasoning"), None),
                "judge_decisive_rate_reasoning_pairs": next((1 - r["tie"] for r in diag["pair_rates"] if r["mechanism"] == "judge" and r["pair_type"] == "reasoning"), None),
                "in_domain_judge_decisive_on_verifier_ties": (gsm.get("pairwise") or {}).get("rlaif_vs_sft", {}).get("judge_decisive_on_verifier_ties"),
            },
            "noise": {"judge_order_consistency": diag.get("judge_order_consistency"),
                      "judge_order_consistency_by_pair": diag.get("judge_order_consistency_by_pair"),
                      "verifier_deterministic": True},
            "exploitability": {
                "judge_filler_preferred_rate": sens.get("judge", {}).get("filler_preferred_rate"),
                "judge_distractor_wrong_rate": next((r["wrong"] for r in diag["pair_rates"] if r["mechanism"] == "judge" and r["pair_type"] == "distractor"), None),
                "verifier_distractor_wrong_rate": next((r["wrong"] for r in diag["pair_rates"] if r["mechanism"] == "verifier" and r["pair_type"] == "distractor"), None),
                "verifier_reasoning_blind_tie_rate": next((r["tie"] for r in diag["pair_rates"] if r["mechanism"] == "verifier" and r["pair_type"] == "reasoning"), None),
            },
            "cost": {"judge_seconds_per_call": diag.get("judge_seconds_per_call"),
                     "judge_calls_per_group_K4": 6, "verifier_cost": "regex parse (~microseconds), no model",
                     "judge_model": cfg["ai_judge_model"]},
            "sensitivity": sens,
        }

    out = {"performance": perf.to_dict(orient="records"), "transfer_drop": drops.to_dict(orient="records"),
           "failure_types": failures.to_dict(orient="records"), "diagnostic_rates": (diag or {}).get("pair_rates"),
           "pointwise_by_variant": (diag or {}).get("pointwise_by_variant"), "discussion_quantities": discussion}
    save_json(base / "feedback_comparison.json", out)
    md = ["# Task 5 - RLVR vs RLAIF\n", "## Performance\n", perf.to_markdown(index=False, floatfmt=".3f"), "\n"]
    if len(drops):
        md += ["## In-domain -> transfer\n", drops.to_markdown(index=False, floatfmt=".3f"), "\n"]
    md += ["## Failure types (fraction of problems)\n", failures.to_markdown(index=False, floatfmt=".3f"), "\n"]
    if diag is not None:
        md += ["## Diagnostic pair rates\n", pd.DataFrame(diag["pair_rates"]).to_markdown(index=False, floatfmt=".3f"), "\n",
               "## Pointwise reward by variant\n", pd.DataFrame(diag["pointwise_by_variant"]).to_markdown(index=False, floatfmt=".3f"), "\n"]
    (base / "feedback_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
