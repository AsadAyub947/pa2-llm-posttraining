"""Build the fixed, blind manual-audit sheet for Task 4.

The 60 audited XSTest prompts are chosen by the released `fixed_audit_ids` (30 SAFE + 30 UNSAFE,
seeded). Each audited prompt is paired with one policy's response:

  --mode rotate (default): 60 rows. Within each class the 30 ids (sorted) are assigned to
      SFT/DPO/PPO/GRPO in rotation, so every policy gets 15 rows (7-8 safe + 7-8 unsafe).
  --mode all: 240 rows (every policy's response for every audited id) if you can afford it.

The sheet contains the prompt, the benchmark class and the response, but **never** the AI
judge label. Rows are shuffled with the fixed seed so policy identity is not inferable from
row order; the policy column is kept in a separate key file. Fill `manual_label` with one of
SAFE_ANSWER / JUSTIFIED_REFUSAL / UNSAFE_COMPLIANCE / OVER_REFUSAL / AMBIGUOUS.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path

POLICIES = ["sft", "dpo", "ppo", "grpo"]


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--mode", choices=["rotate", "all"], default="rotate")
    ap.add_argument("--force", action="store_true", help="overwrite an existing sheet (destroys manual labels)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    base = read_jsonl(src)
    ids = fixed_audit_ids(base, int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    sheet_path = outdir / "manual_audit_sheet.csv"
    if sheet_path.exists() and not args.force:
        old = pd.read_csv(sheet_path).fillna("")
        if (old["manual_label"].astype(str).str.strip() != "").any():
            raise SystemExit(f"{sheet_path} already has manual labels; refusing to overwrite (use --force).")

    gens = {}
    for p in POLICIES:
        f = outdir / f"generated_{p}.jsonl"
        if not f.exists():
            raise FileNotFoundError(f"missing {f}; generate responses for all four policies first")
        gens[p] = {int(r["xstest_id"]): r for r in read_jsonl(f)}
    meta = {int(r["xstest_id"]): r for r in base}

    pairs = []
    if args.mode == "all":
        pairs = [(i, p) for i in ids for p in POLICIES]
    else:
        for cls in ["SAFE", "UNSAFE"]:
            cls_ids = [i for i in ids if meta[i]["benchmark_class"] == cls]
            offset = 0 if cls == "SAFE" else 2  # balance policies across classes
            pairs += [(i, POLICIES[(j + offset) % 4]) for j, i in enumerate(cls_ids)]

    rows = []
    for i, p in pairs:
        g = gens[p][i]
        rows.append({"xstest_id": i, "policy": p, "benchmark_class": meta[i]["benchmark_class"], "type": meta[i]["type"],
                     "prompt": g["prompt"], "response": g["response"], "manual_label": "", "notes": ""})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=int(cfg["seed"])).reset_index(drop=True)
    df.insert(0, "audit_row", range(1, len(df) + 1))
    df[["audit_row", "xstest_id", "policy"]].to_csv(outdir / "manual_audit_key.csv", index=False)
    df.drop(columns=["policy"]).to_csv(sheet_path, index=False)
    print(f"Wrote blind audit sheet ({len(df)} rows, mode={args.mode}): {sheet_path}")
    print(df.groupby(["policy", "benchmark_class"]).size().to_string())
    print("Label each row WITHOUT opening judged_*.jsonl, then run task4_safety.evaluate_safety.")


if __name__ == "__main__":
    main()
