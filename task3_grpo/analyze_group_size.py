"""Task 3 Step 2 - equal-generation group-size study (no training).

Budget rule. The cache holds 8 completions for each of N prompts (8N generations). For each
K in {2, 4, 8} we partition *every* prompt's 8 completions into 8/K disjoint groups of size K,
so every condition uses exactly the same 8N generations (equal total generations); smaller K
simply spends them on more, smaller comparisons. The canonical partition uses generation_index
order (0-1, 2-3, ...); we also repeat the partition with `regroup_permutations` random
within-prompt shuffles and report mean and 2.5/97.5 percentiles to remove ordering artefacts.

Difficulty bins (defined once). A prompt's difficulty is the mean reward of all 8 of its
cached completions; prompts are split into `difficulty_bins` equal-count quantile bins
(tertiles by default): hard = lowest mean reward, easy = highest. Bins are computed before
any regrouping and are identical for every K.

Reported per K (overall and per bin):
  * informative-group rate: fraction of groups with within-group reward std > tol (1e-6),
  * informative groups per 100 generations, fraction of generations with non-zero advantage,
  * mean within-group reward std,
  * variance of the group-relative signal: pooled variance of the centred reward r - mean_g
    (raw relative signal) and of the normalised advantage A_k (zeros for uninformative groups),
  * baseline error: mean squared error of the group mean vs the 8-sample prompt mean,
  * the same informative rate after masking clipped-at-max completions (training-time view).
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

TOL = 1e-6


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int, rng: np.random.Generator | None = None):
    """Return K-sized groups while keeping total cached completions fixed.

    Each prompt's first 8 completions (generation_index order, or a random within-prompt
    permutation if `rng` is given) are split into 8 // k disjoint consecutive groups. Total
    generations used = 8 * n_prompts for every k.
    """
    if 8 % k != 0:
        raise ValueError("k must divide 8")
    groups = []
    for pid, comps in by_prompt.items():
        comps = comps[:8]
        order = rng.permutation(8) if rng is not None else np.arange(8)
        comps = [comps[i] for i in order]
        for j in range(8 // k):
            groups.append({"prompt": pid, "members": comps[j * k:(j + 1) * k]})
    return groups


def group_metrics(groups, prompt_mean: dict) -> dict:
    if not groups:
        return {}
    stds, informative, inf_masked, centred, adv, base_err = [], [], [], [], [], []
    n_gen = 0
    nonzero_adv = 0
    for g in groups:
        r = np.array([float(m["reward"]) for m in g["members"]])
        n_gen += r.size
        sd = r.std()
        stds.append(sd)
        informative.append(sd > TOL)
        keep = np.array([not bool(m.get("clipped_at_max", False)) for m in g["members"]])
        inf_masked.append(bool(sd > TOL and keep.sum() > 0))
        mu = r.mean()
        centred.extend((r - mu).tolist())
        a = (r - mu) / (sd + TOL) if sd > TOL else np.zeros_like(r)
        adv.extend(a.tolist())
        nonzero_adv += int((np.abs(a) > 0).sum())
        base_err.append((mu - prompt_mean[g["prompt"]]) ** 2)
    return {
        "n_groups": len(groups), "n_generations": n_gen,
        "informative_rate": float(np.mean(informative)),
        "uninformative_rate": 1.0 - float(np.mean(informative)),
        "informative_groups_per_100_gen": 100.0 * float(np.sum(informative)) / n_gen,
        "informative_rate_after_trunc_mask": float(np.mean(inf_masked)),
        "frac_generations_nonzero_adv": nonzero_adv / n_gen,
        "mean_within_group_std": float(np.mean(stds)),
        "var_centred_signal": float(np.var(centred)),
        "var_normalized_adv": float(np.var(adv)),
        "baseline_mse": float(np.mean(base_err)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt), "| total generations:", 8 * len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    prompt_mean = {pid: float(np.mean([float(c["reward"]) for c in comps[:8]])) for pid, comps in by_prompt.items()}
    nb = int(cfg.get("difficulty_bins", 3))
    labels = ["hard", "medium", "easy"] if nb == 3 else [f"bin{i}" for i in range(nb)]
    pids = list(prompt_mean)
    bins = pd.qcut([prompt_mean[p] for p in pids], q=nb, labels=labels, duplicates="drop")
    prompt_bin = dict(zip(pids, [str(b) for b in bins]))
    edges = pd.qcut([prompt_mean[p] for p in pids], q=nb, retbins=True, duplicates="drop")[1].tolist()

    R = int(cfg.get("regroup_permutations", 200))
    rng = np.random.default_rng(int(cfg["seed"]))
    rows, perm_rows = [], []
    for k in [int(x) for x in cfg["group_sizes"]]:
        canon = regroup_equal_generation_budget(by_prompt, k)
        subsets = {"all": canon}
        for lab in labels:
            subsets[lab] = [g for g in canon if prompt_bin[g["prompt"]] == lab]
        for name, gs in subsets.items():
            rows.append({"K": k, "subset": name, "partition": "generation_index", **group_metrics(gs, prompt_mean)})
        # random within-prompt partitions -> uncertainty bands
        acc = defaultdict(list)
        for _ in range(R):
            gs = regroup_equal_generation_budget(by_prompt, k, rng)
            for name in ["all"] + labels:
                sub = gs if name == "all" else [g for g in gs if prompt_bin[g["prompt"]] == name]
                m = group_metrics(sub, prompt_mean)
                for key in ["informative_rate", "mean_within_group_std", "var_centred_signal", "baseline_mse"]:
                    acc[(name, key)].append(m[key])
        for (name, key), vals in acc.items():
            perm_rows.append({"K": k, "subset": name, "metric": key, "mean": float(np.mean(vals)),
                              "p2.5": float(np.percentile(vals, 2.5)), "p97.5": float(np.percentile(vals, 97.5))})

    out = repo_path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "group_size_summary.csv", index=False)
    pd.DataFrame(perm_rows).to_csv(out / "group_size_permutation_bands.csv", index=False)
    save_json(out / "group_size_summary.json", {
        "budget_rule": "all 8 cached completions of every prompt are partitioned into 8/K disjoint groups of size K",
        "difficulty_rule": f"prompt mean reward over 8 cached completions, {nb} equal-count quantile bins",
        "difficulty_bin_edges": edges, "bin_counts": pd.Series(prompt_bin).value_counts().to_dict(),
        "tolerance": TOL, "n_prompts": len(by_prompt), "rows": rows, "permutation_bands": perm_rows,
    })
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[["K", "subset", "n_groups", "informative_rate", "informative_groups_per_100_gen",
                  "mean_within_group_std", "var_centred_signal", "var_normalized_adv", "baseline_mse"]].to_string(index=False))


if __name__ == "__main__":
    main()
