"""Shared fork orchestration + summary statistics for the PPO clipping and KL studies."""
from __future__ import annotations

import json

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path


def train_or_reuse_and_eval(cfg, config_path, name, out, train_fn, eval_fn, skip_train=False, force=False):
    """Train a fork unless it already finished (resume after a Colab disconnect), then evaluate
    unless a newer saved evaluation exists. Returns the eval summary."""
    from common.eval_utils import reusable_eval, training_done

    tr = repo_path(cfg["results_dir"]) / name / "train_summary.json"
    if not skip_train:
        if force or not training_done(out, tr):
            train_fn()
        else:
            print(f"[resume] {name} already trained; reusing (use --force to retrain)")
    ev = None if force else reusable_eval(repo_path(cfg["results_dir"]) / "eval" / name / "summary.json", tr,
                                          ["reward", "kl_token_mean", "length"])
    return ev if ev is not None else eval_fn()


def load_train_log(cfg, run_name):
    return read_jsonl(repo_path(cfg["results_dir"]) / run_name / "train_log.jsonl")


def stability_stats(log: list[dict]) -> dict:
    """Clearly defined stability statistics over a fork's updates.

    - update_size_mean / max: per-update approx KL(old || new) on the update's own batch,
      E[(rho - 1) - log rho] after all PPO epochs. Larger = bigger single-update policy moves.
    - policy_loss_std: std of the policy loss across updates.
    - grad_norm_max / grad_norm_cv: max and coefficient of variation of the pre-clip grad norm.
    - kl_ref_slope: least-squares slope of KL(pi || ref) vs update index (drift rate).
    - reward_slope: same for the learned reward.
    """
    if not log:
        return {}
    u = np.array([r["update"] for r in log], dtype=float)

    def slope(key):
        y = np.array([r[key] for r in log], dtype=float)
        return float(np.polyfit(u, y, 1)[0]) if len(u) > 1 else float("nan")

    gn = np.array([r["policy_grad_norm"] for r in log], dtype=float)
    uk = np.array([r.get("approx_kl_old_new", np.nan) for r in log], dtype=float)
    return {
        "update_size_mean": float(np.nanmean(uk)), "update_size_max": float(np.nanmax(uk)),
        "policy_loss_std": float(np.std([r["policy_loss"] for r in log])),
        "grad_norm_mean": float(gn.mean()), "grad_norm_max": float(gn.max()),
        "grad_norm_cv": float(gn.std() / gn.mean()) if gn.mean() > 0 else float("nan"),
        "clip_fraction_mean": float(np.mean([r["clip_fraction"] for r in log])),
        "affected_fraction_mean": float(np.mean([r["affected_fraction"] for r in log])),
        "post_update_clip_fraction_mean": float(np.mean([r.get("post_update_clip_fraction", np.nan) for r in log])),
        "kl_ref_slope": slope("kl_token_mean"), "reward_slope": slope("rm_score"),
        "train_kl_final": float(log[-1]["kl_token_mean"]), "train_reward_mean": float(np.mean([r["rm_score"] for r in log])),
        "train_entropy_mean": float(np.mean([r["entropy"] for r in log])),
        "value_loss_mean": float(np.mean([r["value_loss"] for r in log])),
        "explained_variance_mean": float(np.nanmean([r.get("explained_variance", np.nan) for r in log])),
        "generated_tokens": int(log[-1]["cumulative_generated_tokens"]),
        "updates": len(log),
    }


def eval_row(ev: dict) -> dict:
    return {
        "heldout_rm": ev["reward"]["mean"], "heldout_rm_sem": ev["reward"]["sem"],
        "heldout_rm_eos_pen": ev.get("reward_with_eos_penalty_mean"),
        "heldout_kl_tok": ev["kl_token_mean"], "heldout_kl_seq": ev["kl_seq_mean"],
        "heldout_entropy": ev["entropy_token_mean"],
        "heldout_len_mean": ev["length"]["mean"], "heldout_len_std": ev["length"]["std"],
        "heldout_len_iqr": ev["length"]["iqr"], "heldout_trunc": ev["truncation_rate"],
    }


def load_eval(cfg, name):
    p = repo_path(cfg["results_dir"]) / "eval" / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def first_divergence(log_a: list[dict], log_b: list[dict], keys: list[str], z: float = 1.0) -> dict:
    """First update at which fork A departs from fork B by more than z pooled std (per metric).

    Used to answer "which observable changes first when KL pressure is weakened". The pooled std
    is the std of that metric across all updates of both forks, so metrics on different scales
    are comparable. Returns None if they never diverge.
    """
    out = {}
    n = min(len(log_a), len(log_b))
    for k in keys:
        a = np.array([r[k] for r in log_a[:n]], dtype=float)
        b = np.array([r[k] for r in log_b[:n]], dtype=float)
        sd = np.concatenate([a, b]).std()
        if not np.isfinite(sd) or sd == 0:
            out[k] = None
            continue
        # running mean of the difference smooths single noisy rollouts
        d = np.cumsum(a - b) / np.arange(1, n + 1)
        hit = np.where(np.abs(d) > z * sd)[0]
        out[k] = int(hit[0] + 1) if hit.size else None
    return out
