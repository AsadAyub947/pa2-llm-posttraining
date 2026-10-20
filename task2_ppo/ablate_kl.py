"""Task 2 Step 3 - reward-overoptimisation / KL-pressure study.

From the identical supplied midpoint, run `fork_updates` PPO updates for each beta_KL in
cfg["kl_values"] (same prompts, seeds, epsilon, budgets), evaluate each fork with the common
held-out protocol, and analyse which observable departs first when KL pressure is removed.

Outputs: results/task2_ppo/kl_forks_summary.{csv,json}, kl_trajectories.csv,
kl_first_divergence.json, kl_reward_vs_quality_candidates.jsonl (for manual inspection).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task2_ppo.fork_utils import eval_row, first_divergence, load_train_log, stability_stats, train_or_reuse_and_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="forks already trained; only re-evaluate/analyse")
    ap.add_argument("--force", action="store_true", help="retrain and re-evaluate forks even if results exist")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    from task2_ppo.continue_train import run_ppo
    from task2_ppo.evaluate import evaluate

    betas = [float(b) for b in cfg["kl_values"]]
    print("KL beta conditions:", betas, "| fork update budget:", cfg["fork_updates"])
    rdir = repo_path(cfg["results_dir"])
    rows, logs, traj = [], {}, []
    for b in betas:
        name = f"kl_{b:g}"
        out = f"outputs/task2_ppo/{name}"
        ev = train_or_reuse_and_eval(
            cfg, args.config, name, out,
            lambda: run_ppo(args.config, output=out, updates=int(cfg["fork_updates"]), kl_beta=b, run_name=name),
            lambda: evaluate(args.config, out, name), skip_train=args.skip_train, force=args.force)
        logs[b] = load_train_log(cfg, name)
        rows.append({"kl_beta": b, **eval_row(ev), **stability_stats(logs[b])})
        for r in logs[b]:
            traj.append({"kl_beta": b, **{k: r[k] for k in ["update", "rm_score", "task_reward", "kl_token_mean",
                                                             "kl_seq_mean", "entropy", "response_length",
                                                             "policy_loss", "value_loss", "clip_fraction",
                                                             "policy_grad_norm", "approx_kl_old_new"]}})

    df = pd.DataFrame(rows)
    df.to_csv(rdir / "kl_forks_summary.csv", index=False)
    save_json(rdir / "kl_forks_summary.json", rows)
    pd.DataFrame(traj).to_csv(rdir / "kl_trajectories.csv", index=False)
    print(df.T.to_string())

    # Which observable departs first when KL pressure is weakened? compare weakest vs strongest.
    weak, strong = min(betas), max(betas)
    keys = ["rm_score", "kl_token_mean", "entropy", "response_length"]
    div = {"weak_beta": weak, "strong_beta": strong, "z": 1.0,
           "first_divergence_update": first_divergence(logs[weak], logs[strong], keys)}
    save_json(rdir / "kl_first_divergence.json", div)
    print("First divergence (weak vs strong KL):", div["first_divergence_update"])

    # Reward-vs-quality candidates: same held-out prompt, beta=0 fork vs strongest-beta fork.
    a = {r["id"]: r for r in read_jsonl(rdir / "eval" / f"kl_{weak:g}" / "generations.jsonl")}
    bb = {r["id"]: r for r in read_jsonl(rdir / "eval" / f"kl_{strong:g}" / "generations.jsonl")}
    cands = []
    for pid in a:
        if pid not in bb:
            continue
        ra, rb = a[pid], bb[pid]
        cands.append({"id": pid, "prompt": ra["prompt"], "reward_gain": ra["reward"] - rb["reward"],
                      "len_ratio": (ra["response_tokens"] + 1) / (rb["response_tokens"] + 1),
                      f"reward_kl{weak:g}": ra["reward"], f"reward_kl{strong:g}": rb["reward"],
                      f"response_kl{weak:g}": ra["response"], f"response_kl{strong:g}": rb["response"]})
    cands.sort(key=lambda c: -c["reward_gain"])
    write_jsonl(rdir / "kl_reward_vs_quality_candidates.jsonl", cands)
    if cands:
        print("Reward gain vs length-ratio corr:",
              float(np.corrcoef([c["reward_gain"] for c in cands], [np.log(c["len_ratio"]) for c in cands])[0, 1]))


if __name__ == "__main__":
    main()
