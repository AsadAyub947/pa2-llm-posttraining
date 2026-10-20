"""Task 3 Step 3 - canonical GRPO (1/T_k) vs Dr.-GRPO-style (1/L_max) sequence normalisation.

Both forks start from the identical supplied midpoint and use the same prompts, seeds, K,
reward, beta, epsilon, generation settings and update count (`fork_updates`). Only `loss_type`
changes. Group-relative advantages are the same (within-group mean/std) in both forks so that
the comparison isolates the length normalisation term.

Length-conditioned statistics (from per-completion training logs, micro_batch_size = 1):
  * exact gradient norm of each completion's policy term vs its length T
    (Spearman correlation, mean grad norm for short vs long completions split at the pooled
    median length, share of total gradient norm carried by long completions),
  * the analytic per-token weight |A|/T (grpo) vs |A|/L_max (dr_grpo), separately for positive
    and negative advantages (the Liu et al. 2025 length-bias argument),
  * response length trajectory during training.
Held-out: reward, KL, entropy, response length with the common protocol.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task2_ppo.fork_utils import train_or_reuse_and_eval


def length_conditioned(log: list[dict], median_len: float) -> dict:
    comps = [c for r in log for c in r["per_completion"] if not c.get("masked")]
    if not comps:
        return {}
    L = np.array([c["length"] for c in comps], dtype=float)
    G = np.array([c.get("policy_grad_norm", np.nan) for c in comps], dtype=float)
    A = np.array([c["advantage"] for c in comps], dtype=float)
    W = np.array([c["token_weight"] for c in comps], dtype=float)
    long_ = L > median_len
    ok = np.isfinite(G) & (np.abs(A) > 0)

    def spearman(x, y):
        if len(x) < 3:
            return float("nan")
        return float(pd.Series(x).rank().corr(pd.Series(y).rank()))

    out = {
        "n_completions": int(len(comps)),
        "spearman_gradnorm_vs_length": spearman(L[ok], G[ok]),
        "gradnorm_short_mean": float(np.nanmean(G[ok & ~long_])) if (ok & ~long_).any() else float("nan"),
        "gradnorm_long_mean": float(np.nanmean(G[ok & long_])) if (ok & long_).any() else float("nan"),
        "gradnorm_share_long": float(np.nansum(G[ok & long_]) / np.nansum(G[ok])) if ok.any() else float("nan"),
        "frac_long": float(long_.mean()),
        # gradient norm per unit |advantage| removes the advantage-magnitude confound
        "gradnorm_per_absadv_short": float(np.nanmean(G[ok & ~long_] / np.abs(A[ok & ~long_]))) if (ok & ~long_).any() else float("nan"),
        "gradnorm_per_absadv_long": float(np.nanmean(G[ok & long_] / np.abs(A[ok & long_]))) if (ok & long_).any() else float("nan"),
        "token_weight_pos_short": float(np.mean(W[(A > 0) & ~long_])) if ((A > 0) & ~long_).any() else float("nan"),
        "token_weight_pos_long": float(np.mean(W[(A > 0) & long_])) if ((A > 0) & long_).any() else float("nan"),
        "token_weight_neg_short": float(np.mean(W[(A < 0) & ~long_])) if ((A < 0) & ~long_).any() else float("nan"),
        "token_weight_neg_long": float(np.mean(W[(A < 0) & long_])) if ((A < 0) & long_).any() else float("nan"),
        "len_mean_pos_adv": float(L[A > 0].mean()) if (A > 0).any() else float("nan"),
        "len_mean_neg_adv": float(L[A < 0].mean()) if (A < 0).any() else float("nan"),
        "corr_reward_length": float(np.corrcoef(L, [c["reward"] for c in comps])[0, 1]) if L.std() > 0 else float("nan"),
    }
    first, last = log[: max(1, len(log) // 2)], log[len(log) // 2:]
    out["train_len_first_half"] = float(np.mean([r["response_length"] for r in first]))
    out["train_len_second_half"] = float(np.mean([r["response_length"] for r in last]))
    out["train_reward_mean"] = float(np.mean([r["reward_mean"] for r in log]))
    out["train_kl_final"] = float(log[-1]["kl_token_mean"])
    out["grad_norm_mean"] = float(np.mean([r["grad_norm"] for r in log]))
    out["generated_tokens"] = int(log[-1]["cumulative_generated_tokens"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain and re-evaluate forks even if results exist")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    from task3_grpo.continue_train import run_grpo
    from task3_grpo.evaluate import evaluate

    print("Fork updates:", cfg["fork_updates"])
    rdir = repo_path(cfg["results_dir"])
    logs, evs = {}, {}
    for lt in ["grpo", "dr_grpo"]:
        name = f"norm_{lt}"
        out = f"outputs/task3_grpo/{name}"
        evs[lt] = train_or_reuse_and_eval(
            cfg, args.config, name, out,
            lambda: run_grpo(args.config, output=out, updates=int(cfg["fork_updates"]), loss_type=lt, run_name=name),
            lambda: evaluate(args.config, out, name), skip_train=args.skip_train, force=args.force)
        logs[lt] = read_jsonl(rdir / name / "train_log.jsonl")

    pooled = [c["length"] for lt in logs for r in logs[lt] for c in r["per_completion"] if not c.get("masked")]
    med = float(np.median(pooled)) if pooled else 0.0
    rows = []
    for lt in ["grpo", "dr_grpo"]:
        ev = evs[lt]
        rows.append({
            "loss_type": lt, "median_split_len": med,
            "heldout_rm": ev["reward"]["mean"], "heldout_rm_sem": ev["reward"]["sem"],
            "heldout_kl_tok": ev["kl_token_mean"], "heldout_entropy": ev["entropy_token_mean"],
            "heldout_len_mean": ev["length"]["mean"], "heldout_len_std": ev["length"]["std"],
            "heldout_len_iqr": ev["length"]["iqr"], "heldout_trunc": ev["truncation_rate"],
            **length_conditioned(logs[lt], med),
        })
    df = pd.DataFrame(rows)
    df.to_csv(rdir / "normalization_summary.csv", index=False)
    save_json(rdir / "normalization_summary.json", rows)
    per = []
    for lt in logs:
        for r in logs[lt]:
            for c in r["per_completion"]:
                per.append({"loss_type": lt, "update": r["update"], **c})
    pd.DataFrame(per).to_csv(rdir / "normalization_per_completion.csv", index=False)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
