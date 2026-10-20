"""Task 6 - cross-task synthesis: figures + compact tables from saved results only (no training).

Every figure/table is built from the machine-readable files written by Tasks 1-5, so each
number in the report traces to a file under results/. Missing inputs are skipped with a note.

Outputs:
  report/figures/*.pdf|png
  results/synthesis/cross_task_policies.csv      (reward / KL / entropy / length per evaluated condition)
  results/synthesis/safety_vs_reward.csv         (Task 4 rates joined with the standard policies' metrics)
  results/synthesis/synthesis_tables.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import read_jsonl, repo_path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colour-blind-safe categorical palette (Okabe-Ito)
C = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.bbox": "tight",
                     "figure.constrained_layout.use": True, "axes.formatter.useoffset": False})

R = repo_path("results")
FIG = repo_path("report/figures")
OUT = R / "synthesis"


def _json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("  figure:", FIG / f"{name}.pdf")


# ------------------------------------------------------------------------------ Task 1
def fig_dpo():
    p = R / "task1_dpo" / "beta_sweep_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        short = df[df["condition"].str.startswith("short")].sort_values("beta")
        fig, ax = plt.subplots(1, 4, figsize=(8, 1.9))
        for a, (col, lab) in zip(ax, [("heldout_pref_acc", "held-out pref. acc."), ("kl_token_mean", "KL to ref (token mean)"),
                                      ("rm_score", "RM score"), ("len_mean", "response tokens")]):
            a.plot(short["beta"], short[col], "o-", color=C[0], label="short forks (600 ex.)")
            std = df[df["condition"].str.startswith("standard")]
            if len(std):
                a.axhline(float(std[col].iloc[0]), color=C[1], ls="--", lw=1, label="standard (1 epoch, β=0.1)")
            a.set_xscale("log")
            a.set_xticks(short["beta"])
            a.set_xticklabels([f"{b:g}" for b in short["beta"]])
            a.set_xlabel("β")
            a.set_title(lab)
        ax[0].legend(fontsize=6, frameon=False)
        save(fig, "t1_beta_sweep")
    p = R / "task1_dpo" / "length_study_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        strata = ["preferred_longer", "length_matched", "rejected_longer"]
        fig, ax = plt.subplots(1, 3, figsize=(8, 2.0), gridspec_kw={"width_ratios": [2, 1, 1]})
        w = 0.8 / len(df)
        for i, (_, r) in enumerate(df.iterrows()):
            ax[0].bar(np.arange(3) + i * w, [r[f"acc_{s}"] for s in strata], w, color=C[i], label=r["condition"])
        ax[0].set_xticks(np.arange(3) + w * (len(df) - 1) / 2)
        ax[0].set_xticklabels(["preferred longer", "length matched", "rejected longer"])
        ax[0].axhline(0.5, color="grey", lw=0.8, ls=":")
        ax[0].set_ylabel("preference accuracy")
        ax[0].legend(fontsize=6, frameon=False)
        ax[1].bar(df["condition"], df["gen_len_mean"], yerr=df["gen_len_std"], color=C[: len(df)])
        ax[1].set_title("generated tokens")
        ax[2].bar(df["condition"], df["wl_compliance_sampled"], color=C[: len(df)])
        ax[2].set_title("word-limit compliance")
        for a in ax[1:]:
            a.tick_params(axis="x", rotation=30)
        save(fig, "t1_length_confounding")


# ------------------------------------------------------------------------------ Task 2/3 trajectories
def fig_traj(task, run, keys, name):
    p = R / task / run / "train_log.jsonl"
    if not p.exists():
        return
    log = read_jsonl(p)
    u = [r["update"] for r in log]
    n = len(keys)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(8, 2.0 * rows))
    ax = np.atleast_1d(ax).ravel()
    for a, (k, lab) in zip(ax, keys):
        y = [r.get(k, np.nan) for r in log]
        a.plot(u, y, "-", color=C[0], lw=1)
        a.plot(u, pd.Series(y).rolling(5, min_periods=1).mean(), color=C[3], lw=1.2)
        a.set_title(lab)
        a.set_xlabel("update")
    for a in ax[n:]:
        a.axis("off")
    save(fig, name)


def fig_ppo_studies():
    p = R / "task2_ppo" / "clipping_cached.csv"
    if p.exists():
        df = pd.read_csv(p)
        fig, ax = plt.subplots(1, 3, figsize=(8, 1.9))
        for i, (eps, g) in enumerate(df.groupby("eps")):
            ax[0].plot(g["step"], g["clip_fraction"], "o-", color=C[i], label=f"ε={eps:g}")
            ax[1].plot(g["step"], g["affected_fraction"], "o-", color=C[i])
            ax[2].plot(g["step"], g["clipped_surrogate"], "o-", color=C[i])
        for a, t in zip(ax, ["clip fraction", "affected-token fraction", "clipped surrogate"]):
            a.set_title(t)
            a.set_xlabel("optimiser steps on cached batch")
        ax[0].legend(fontsize=6, frameon=False)
        save(fig, "t2_clipping_cached")
    p = R / "task2_ppo" / "kl_trajectories.csv"
    if p.exists():
        df = pd.read_csv(p)
        keys = [("rm_score", "learned reward"), ("kl_token_mean", "KL to ref"), ("entropy", "entropy"), ("response_length", "tokens")]
        fig, ax = plt.subplots(1, 4, figsize=(8, 1.9))
        for i, (b, g) in enumerate(df.groupby("kl_beta")):
            for a, (k, lab) in zip(ax, keys):
                a.plot(g["update"], g[k], "o-", ms=2, color=C[i], label=f"β_KL={b:g}")
                a.set_title(lab)
                a.set_xlabel("update")
        ax[0].legend(fontsize=6, frameon=False)
        save(fig, "t2_kl_forks")


def fig_grpo_studies():
    p = R / "task3_grpo" / "group_size_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        fig, ax = plt.subplots(1, 3, figsize=(8, 1.9))
        subsets = [s for s in ["all", "hard", "medium", "easy"] if s in set(df["subset"])]
        for i, s in enumerate(subsets):
            g = df[df.subset == s].sort_values("K")
            ax[0].plot(g["K"], g["informative_rate"], "o-", color=C[i], label=s)
            ax[1].plot(g["K"], g["mean_within_group_std"], "o-", color=C[i])
            ax[2].plot(g["K"], g["baseline_mse"], "o-", color=C[i])
        for a, t in zip(ax, ["informative-group rate", "within-group reward std", "baseline MSE vs 8-sample mean"]):
            a.set_title(t)
            a.set_xlabel("K (equal total generations)")
            a.set_xticks([2, 4, 8])
        ax[0].legend(fontsize=6, frameon=False)
        save(fig, "t3_group_size")
    p = R / "task3_grpo" / "normalization_per_completion.csv"
    if p.exists():
        df = pd.read_csv(p)
        df = df[(~df["masked"]) & (df["advantage"].abs() > 0)]
        if "policy_grad_norm" in df:
            fig, ax = plt.subplots(1, 2, figsize=(6, 2.0))
            for i, (lt, g) in enumerate(df.groupby("loss_type")):
                ax[0].scatter(g["length"], g["policy_grad_norm"], s=6, color=C[i], alpha=0.7, label=lt)
                ax[1].scatter(g["length"], g["policy_grad_norm"] / g["advantage"].abs(), s=6, color=C[i], alpha=0.7)
            ax[0].set_title("per-completion grad norm")
            ax[1].set_title("grad norm / |A|")
            for a in ax:
                a.set_xlabel("completion tokens")
                a.set_yscale("log")
            ax[0].legend(fontsize=6, frameon=False)
            save(fig, "t3_normalization_gradnorm")


# ------------------------------------------------------------------------------ Task 4/5
def fig_safety():
    p = R / "task4_safety" / "safety_summary.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.0))
    x = np.arange(len(df))
    ax[0].bar(x - 0.2, df["safe_answer_rate"], 0.4, color=C[2], label="safe answer")
    ax[0].bar(x + 0.2, df["safe_over_refusal_rate"], 0.4, color=C[3], label="over-refusal")
    ax[0].set_title("SAFE prompts")
    ax[1].bar(x - 0.2, df["justified_refusal_rate"], 0.4, color=C[0], label="justified refusal")
    ax[1].bar(x + 0.2, df["unsafe_compliance_rate"], 0.4, color=C[1], label="unsafe compliance")
    ax[1].set_title("UNSAFE prompts")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(df["policy"].str.upper())
        a.set_ylim(0, 1)
        a.legend(fontsize=6, frameon=False)
    save(fig, "t4_safety_rates")
    p = R / "task4_safety" / "safety_by_category.csv"
    if p.exists():
        cat = pd.read_csv(p)
        labs = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
        pols = [p for p in ["sft", "dpo", "ppo", "grpo"] if p in set(cat["policy"])]
        types = list(dict.fromkeys(cat["type"]))
        fig, ax = plt.subplots(1, len(pols), figsize=(8, 3.2), sharey=True)
        ax = np.atleast_1d(ax)
        for a, pol in zip(ax, pols):
            g = cat[cat.policy == pol].set_index("type").reindex(types)
            left = np.zeros(len(types))
            for i, l in enumerate(labs):
                v = g[l].fillna(0).values
                a.barh(types, v, left=left, color=C[[2, 0, 1, 3, 7][i]], label=l if pol == pols[0] else None)
                left += v
            a.set_title(pol.upper())
        fig.legend(*ax[0].get_legend_handles_labels(), fontsize=6, frameon=False, loc="outside lower center", ncol=5)
        save(fig, "t4_safety_by_category")


def fig_feedback():
    d = _json(R / "task5_feedback" / "diagnostics" / "summary.json")
    if d is None:
        return
    df = pd.DataFrame(d["pair_rates"])
    types = ["reasoning", "outcome", "distractor", "filler", "conflict"]
    mechs = list(dict.fromkeys(df["mechanism"]))
    fig, ax = plt.subplots(1, len(mechs), figsize=(4 * len(mechs), 2.0), sharey=True)
    ax = np.atleast_1d(ax)
    for a, m in zip(ax, mechs):
        g = df[df.mechanism == m].set_index("pair_type").reindex(types)
        a.bar(types, g["better"], color=C[2], label="prefers better / A")
        a.bar(types, g["tie"], bottom=g["better"], color="#BBBBBB", label="tie")
        a.bar(types, g["wrong"], bottom=g["better"] + g["tie"], color=C[3], label="prefers worse / B")
        a.set_title(m)
        a.tick_params(axis="x", rotation=30)
    ax[0].legend(fontsize=6, frameon=False)
    save(fig, "t5_diagnostic_pairs")


# ------------------------------------------------------------------------------ cross-task tables
def cross_task_tables():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, sub in [("task1_dpo", "DPO"), ("task2_ppo", "PPO"), ("task3_grpo", "GRPO")]:
        for f in sorted((R / task / "eval").glob("*/summary.json")):
            s = json.loads(f.read_text())
            g = s.get("generation", s)
            if not g or "kl_token_mean" not in g:
                continue
            rows.append({"task": sub, "condition": f.parent.name,
                         "prompt_set": "DPO held-out prompts" if task == "task1_dpo" else "RL eval prompts",
                         "rm_score": (g.get("reward") or {}).get("mean"), "kl_token_mean": g.get("kl_token_mean"),
                         "kl_seq_mean": g.get("kl_seq_mean"), "entropy": g.get("entropy_token_mean"),
                         "len_mean": g["length"]["mean"], "len_std": g["length"]["std"], "trunc": g.get("truncation_rate"),
                         "max_new_tokens": g["decoding"]["max_new_tokens"]})
    ct = pd.DataFrame(rows)
    if len(ct):
        ct.to_csv(OUT / "cross_task_policies.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(7, 2.3))
        for i, (t, g) in enumerate(ct.groupby("task")):
            ax[0].scatter(g["kl_token_mean"], g["rm_score"], color=C[i], label=t, s=14)
            ax[1].scatter(g["kl_token_mean"], g["len_mean"], color=C[i], s=14)
            for _, r in g.iterrows():
                ax[0].annotate(r["condition"], (r["kl_token_mean"], r["rm_score"]), fontsize=4)
        ax[0].set_xlabel("KL to reference (token mean)")
        ax[0].set_ylabel("RM score")
        ax[1].set_xlabel("KL to reference (token mean)")
        ax[1].set_ylabel("response tokens")
        ax[0].legend(fontsize=6, frameon=False)
        save(fig, "t6_drift_vs_reward")

    md = ["# Cross-task synthesis tables\n"]
    if len(ct):
        md += ["## Reward / drift / length per evaluated condition\n",
               "RM scores are comparable only within a prompt set (DPO and RL prompt sets differ).\n",
               ct.to_markdown(index=False, floatfmt=".4g"), "\n"]
    s4 = R / "task4_safety" / "safety_summary.csv"
    if s4.exists():
        sf = pd.read_csv(s4)
        link = {"dpo": R / "task1_dpo" / "eval" / "standard" / "summary.json",
                "ppo": R / "task2_ppo" / "eval" / "standard" / "summary.json",
                "grpo": R / "task3_grpo" / "eval" / "standard" / "summary.json"}
        extra = []
        for pol in sf["policy"]:
            s = _json(link[pol]) if pol in link else None
            g = (s or {}).get("generation", s or {})
            extra.append({"policy": pol, "train_task_rm": (g.get("reward") or {}).get("mean") if g else None,
                          "train_task_kl": g.get("kl_token_mean") if g else None,
                          "heldout_pref_acc": ((s or {}).get("pairs_standard_eval") or {}).get("overall", {}).get("preference_accuracy")})
        sv = sf.merge(pd.DataFrame(extra), on="policy")
        sv.to_csv(OUT / "safety_vs_reward.csv", index=False)
        md += ["## Safety calibration vs training-signal metrics\n", sv.to_markdown(index=False, floatfmt=".3f"), "\n"]
    for name, path in [("DPO β sweep", R / "task1_dpo" / "beta_sweep_summary.csv"),
                       ("DPO length study", R / "task1_dpo" / "length_study_summary.csv"),
                       ("PPO clipping forks", R / "task2_ppo" / "clipping_forks_summary.csv"),
                       ("PPO KL forks", R / "task2_ppo" / "kl_forks_summary.csv"),
                       ("GRPO group size", R / "task3_grpo" / "group_size_summary.csv"),
                       ("GRPO normalisation", R / "task3_grpo" / "normalization_summary.csv")]:
        if path.exists():
            md += [f"## {name}\n", pd.read_csv(path).to_markdown(index=False, floatfmt=".4g"), "\n"]
    for name, task in [("PPO standard", "task2_ppo"), ("GRPO standard", "task3_grpo")]:
        s = _json(R / task / "standard" / "train_summary.json")
        if s:
            md.append(f"**{name} compute:** wall-clock {s['wall_clock_s'] / 60:.1f} min, peak VRAM "
                      f"{s['peak_vram_gib'] if s['peak_vram_gib'] is None else round(s['peak_vram_gib'], 2)} GiB on {s.get('gpu')}, "
                      f"{s['generated_tokens']} generated tokens.\n")
    (OUT / "synthesis_tables.md").write_text("\n".join(md), encoding="utf-8")
    print("  tables:", OUT / "synthesis_tables.md")


def main():
    global R, FIG, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--fig-dir", default="report/figures")
    args = ap.parse_args()
    R, FIG = repo_path(args.results_dir), repo_path(args.fig_dir)
    OUT = R / "synthesis"
    print(f"[task6] building figures and tables from {R}")
    fig_dpo()
    fig_traj("task2_ppo", "standard",
             [("rm_score", "learned reward"), ("kl_token_mean", "KL to ref"), ("policy_loss", "policy loss"),
              ("value_loss", "value loss"), ("entropy", "entropy"), ("clip_fraction", "clip fraction"),
              ("policy_grad_norm", "grad norm"), ("response_length", "response tokens")], "t2_ppo_standard")
    fig_traj("task3_grpo", "standard",
             [("reward_mean", "reward"), ("kl_token_mean", "KL to ref"), ("group_reward_std_mean", "within-group std"),
              ("uninformative_group_frac", "uninformative groups"), ("policy_loss", "policy loss"),
              ("grad_norm", "grad norm"), ("entropy", "entropy"), ("response_length", "response tokens")], "t3_grpo_standard")
    fig_ppo_studies()
    fig_grpo_studies()
    fig_safety()
    fig_feedback()
    cross_task_tables()


if __name__ == "__main__":
    main()
