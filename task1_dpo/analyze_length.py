"""Task 1 Step 3 - length-confounding study.

1. Dataset diagnostics (property of the *data*, no model): token-length difference between
   chosen and rejected responses in the standard train set, the balanced train set and the
   stratified eval set (fraction preferred-longer, mean/median difference).
2. Train one DPO model from the original initialisation on the supplied length-balanced subset
   (one epoch, same optimiser/LoRA/seed/beta as the standard run).
3. Evaluate SFT, standard DPO and length-balanced DPO with the common protocol, including the
   length-stratified held-out pairs (preferred_longer / length_matched / rejected_longer),
   generated response length, and word-limit compliance (property of the *policy*).

Requires the standard run (python -m task1_dpo.train --run-name standard) to exist.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.eval_utils import reusable_eval, training_done
from common.logging_utils import save_json
from common.models import load_tokenizer
from task1_dpo.evaluate import evaluate_policy
from task1_dpo.train import run_training


def dataset_length_profile(tokenizer, rows, name):
    diffs, strata = [], []
    for r in rows:
        if "chosen_tokens" in r and "rejected_tokens" in r:
            c, j = int(r["chosen_tokens"]), int(r["rejected_tokens"])
        else:
            yc, yr = preference_responses(r)
            c = len(tokenizer(yc, add_special_tokens=False)["input_ids"])
            j = len(tokenizer(yr, add_special_tokens=False)["input_ids"])
        diffs.append(c - j)
        strata.append(r.get("length_stratum"))
    d = np.asarray(diffs, dtype=float)
    out = {
        "dataset": name, "n": int(d.size),
        "frac_chosen_longer": float(np.mean(d > 0)), "frac_rejected_longer": float(np.mean(d < 0)),
        "frac_equal": float(np.mean(d == 0)),
        "len_diff_mean": float(d.mean()), "len_diff_median": float(np.median(d)), "len_diff_std": float(d.std()),
    }
    if any(s is not None for s in strata):
        out["stratum_counts"] = pd.Series([s for s in strata if s]).value_counts().to_dict()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-sft", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain and re-evaluate even if results exist")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"])

    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))

    tok = load_tokenizer(cfg["base_model"])
    profiles = [
        dataset_length_profile(tok, read_jsonl(cfg["paths"]["dpo_standard_train"]), "standard_train"),
        dataset_length_profile(tok, read_jsonl(cfg["paths"]["dpo_standard_eval"]), "standard_eval"),
        dataset_length_profile(tok, balanced, "length_balanced_train"),
        dataset_length_profile(tok, stratified, "length_stratified_eval"),
    ]
    save_json(outdir / "length_dataset_profile.json", profiles)
    print(pd.DataFrame(profiles).drop(columns=["stratum_counts"], errors="ignore").to_string(index=False))

    lb_summary = outdir / "length_balanced" / "train_summary.json"
    if not args.skip_train and (args.force or not training_done(cfg["length_output"], lb_summary)):
        run_training(args.config, run_name="length_balanced", dataset_path=cfg["paths"]["dpo_length_train"],
                     output_path=cfg["length_output"], max_examples=cfg.get("length_balanced_max_examples"))
    elif not args.skip_train:
        print("[resume] length_balanced already trained; reusing (use --force to retrain)")

    extra = {"length_stratified": cfg["paths"]["dpo_length_eval"]}
    conditions = [("standard", cfg["standard_output"]), ("length_balanced", cfg["length_output"])]
    if not args.skip_sft:
        conditions.insert(0, ("sft", None))
    rows = []
    for name, adapter in conditions:
        if name == "standard" and not repo_path(adapter).exists():
            raise FileNotFoundError("Train the standard DPO model first: python -m task1_dpo.train --run-name standard")
        ev_name = f"{name}_lengthstudy" if name != "standard" else "standard"
        tr_summary = outdir / name / "train_summary.json" if name != "sft" else None
        ev = None if args.force else reusable_eval(outdir / "eval" / ev_name / "summary.json", tr_summary,
                                                   ["pairs_length_stratified", "generation", "word_limit"])
        if ev is None:
            ev = evaluate_policy(args.config, adapter, ev_name, extra_pairs_files=extra)
        strata = ev["pairs_length_stratified"].get("by_stratum", {})
        base = {"condition": name}
        for s in ["preferred_longer", "length_matched", "rejected_longer"]:
            st = strata.get(s, {})
            base[f"acc_{s}"] = st.get("preference_accuracy")
            base[f"margin_{s}"] = st.get("margin_mean")
            base[f"n_{s}"] = st.get("n")
        base["acc_stratified_overall"] = ev["pairs_length_stratified"]["overall"].get("preference_accuracy")
        base["corr_margin_lendiff"] = ev["pairs_length_stratified"]["overall"].get("corr_margin_vs_length_diff")
        g, w = ev["generation"], ev["word_limit"]
        base.update({
            "rm_score": g["reward"]["mean"], "kl_token_mean": g["kl_token_mean"],
            "gen_len_mean": g["length"]["mean"], "gen_len_std": g["length"]["std"], "gen_len_iqr": g["length"]["iqr"],
            "wl_compliance_greedy": w["greedy"]["compliance_rate"], "wl_compliance_sampled": w["sampled"]["compliance_rate"],
            "wl_words_mean_sampled": w["sampled"]["words"]["mean"], "wl_words_std_sampled": w["sampled"]["words"]["std"],
            "wl_ratio_to_limit": w["sampled"]["ratio_words_to_limit_mean"],
        })
        rows.append(base)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "length_study_summary.csv", index=False)
    save_json(outdir / "length_study_summary.json", rows)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
