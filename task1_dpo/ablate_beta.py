"""Task 1 Step 2 - regularisation-strength study.

For each beta in cfg["betas"]: train from the *original* initialisation (fresh LoRA on the base
model, same seed) on the fixed first `short_ablation_examples` rows of the standard training
file, with identical optimiser/LoRA/seed; then evaluate every fork with the common protocol.
Only beta changes between forks.

Outputs: outputs/task1_dpo/beta_<b>/ (adapters), results/task1_dpo/beta_<b>/ (train logs),
results/task1_dpo/eval/beta_<b>/ (eval), results/task1_dpo/beta_sweep_summary.{json,csv}.
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from common.eval_utils import reusable_eval, training_done
from task1_dpo.evaluate import evaluate_policy
from task1_dpo.train import run_training


def beta_tag(b: float) -> str:
    return f"beta_{b:.2f}".replace(".", "p")


def flatten(name, beta, ev, tr):
    p = ev["pairs_standard_eval"]["overall"]
    g = ev.get("generation", {})
    w = ev.get("word_limit", {})
    return {
        "condition": name, "beta": beta, "train_examples": tr.get("n_examples") if tr else None,
        "optimizer_steps": tr.get("optimizer_steps") if tr else None,
        "heldout_dpo_loss": p.get("dpo_loss"), "heldout_pref_acc": p.get("preference_accuracy"),
        "heldout_margin": p.get("margin_mean"),
        "chosen_reward": p.get("chosen_reward_mean"), "rejected_reward": p.get("rejected_reward_mean"),
        "kl_token_mean": g.get("kl_token_mean"), "kl_seq_mean": g.get("kl_seq_mean"),
        "entropy": g.get("entropy_token_mean"),
        "rm_score": (g.get("reward") or {}).get("mean"), "rm_score_sem": (g.get("reward") or {}).get("sem"),
        "len_mean": (g.get("length") or {}).get("mean"), "len_std": (g.get("length") or {}).get("std"),
        "len_iqr": (g.get("length") or {}).get("iqr"), "truncation_rate": g.get("truncation_rate"),
        "word_limit_compliance_greedy": (w.get("greedy") or {}).get("compliance_rate"),
        "word_limit_compliance_sampled": (w.get("sampled") or {}).get("compliance_rate"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="re-evaluate existing forks only")
    ap.add_argument("--force", action="store_true", help="retrain and re-evaluate even if results exist")
    ap.add_argument("--include-standard", action="store_true",
                    help="also add the standard one-epoch run (must already be trained) to the table")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = [float(b) for b in cfg["betas"]]
    n = int(cfg["short_ablation_examples"])
    print("Required beta values:", betas, "| short-run examples per condition:", n)

    rows = []
    import json
    for b in betas:
        tag = beta_tag(b)
        out = f"outputs/task1_dpo/{tag}"
        tr_path = repo_path(cfg["results_dir"]) / tag / "train_summary.json"
        if args.skip_train or (not args.force and training_done(out, tr_path)):
            if not args.skip_train:
                print(f"[resume] {tag} already trained; reusing (use --force to retrain)")
            tr = json.loads(tr_path.read_text()) if tr_path.exists() else None
        else:
            tr = run_training(args.config, run_name=tag, output_path=out, beta=b, max_examples=n)
        ev = None if args.force else reusable_eval(repo_path(cfg["results_dir"]) / "eval" / tag / "summary.json", tr_path,
                                                   ["pairs_standard_eval", "generation", "word_limit"])
        if ev is None:
            ev = evaluate_policy(args.config, out, tag, beta=b)
        rows.append(flatten(f"short β={b:g}", b, ev, tr))

    if args.include_standard:
        import json
        ev_p = repo_path(cfg["results_dir"]) / "eval" / "standard" / "summary.json"
        tr_p = repo_path(cfg["results_dir"]) / "standard" / "train_summary.json"
        if ev_p.exists():
            rows.insert(0, flatten("standard (1 epoch)", float(cfg["beta"]), json.loads(ev_p.read_text()),
                                   json.loads(tr_p.read_text()) if tr_p.exists() else None))

    df = pd.DataFrame(rows)
    outdir = repo_path(cfg["results_dir"])
    df.to_csv(outdir / "beta_sweep_summary.csv", index=False)
    save_json(outdir / "beta_sweep_summary.json", rows)
    print(df.to_string(index=False))
    # monotonicity check over the tested range (short forks only)
    short = df[df["condition"].str.startswith("short")].sort_values("beta")
    mono = {}
    for col in ["heldout_pref_acc", "kl_token_mean", "rm_score", "len_mean"]:
        d = short[col].diff().dropna()
        mono[col] = "increasing" if (d > 0).all() else "decreasing" if (d < 0).all() else "non-monotonic"
    print("Monotonicity in beta:", mono)
    save_json(outdir / "beta_sweep_monotonicity.json", mono)


if __name__ == "__main__":
    main()
