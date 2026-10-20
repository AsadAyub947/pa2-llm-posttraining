"""Shortlist candidate qualitative examples by comparing two policies on the same prompts.

Works for any pair of `generations.jsonl` files written by the common evaluators (Task 1/2/3),
joined on prompt id. It ranks prompts where the learned reward and simple quality proxies move
in different directions so you can *manually* read a small number of candidates:

  reward_up_but_longer  : B has higher RM score than A but is much longer / truncated
  reward_up_and_shorter : B has higher RM score and is not longer (reward and quality may agree)
  reward_down           : B has lower RM score than A

Usage:
  python -m scripts.find_qualitative --a results/task1_dpo/eval/sft_lengthstudy/generations.jsonl \
      --b results/task1_dpo/eval/standard/generations.jsonl --out results/task1_dpo/qualitative_candidates.jsonl
Then read the shortlisted responses yourself; the script makes no quality judgement.
"""
from __future__ import annotations

import argparse

from common.data import read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline generations.jsonl")
    ap.add_argument("--b", required=True, help="trained-policy generations.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    A = {str(r["id"]): r for r in read_jsonl(args.a)}
    B = {str(r["id"]): r for r in read_jsonl(args.b)}
    rows = []
    for k in A.keys() & B.keys():
        a, b = A[k], B[k]
        rows.append({"id": k, "prompt": a["prompt"], "reward_a": a["reward"], "reward_b": b["reward"],
                     "reward_delta": b["reward"] - a["reward"], "tokens_a": a["response_tokens"],
                     "tokens_b": b["response_tokens"], "len_ratio": (b["response_tokens"] + 1) / (a["response_tokens"] + 1),
                     "truncated_b": b["truncated"], "response_a": a["response"], "response_b": b["response"]})
    up_long = sorted([r for r in rows if r["reward_delta"] > 0 and (r["len_ratio"] > 1.3 or r["truncated_b"])],
                     key=lambda r: -r["reward_delta"])[: args.top]
    up_short = sorted([r for r in rows if r["reward_delta"] > 0 and r["len_ratio"] <= 1.0],
                      key=lambda r: -r["reward_delta"])[: args.top]
    down = sorted([r for r in rows if r["reward_delta"] < 0], key=lambda r: r["reward_delta"])[: args.top]
    out = ([{"bucket": "reward_up_but_longer", **r} for r in up_long] + [{"bucket": "reward_up_and_shorter", **r} for r in up_short]
           + [{"bucket": "reward_down", **r} for r in down])
    write_jsonl(args.out, out)
    print(f"{len(rows)} shared prompts; wrote {len(out)} candidates to {args.out}")
    for r in out:
        print(f"[{r['bucket']}] id={r['id']} ΔR={r['reward_delta']:+.2f} tokens {r['tokens_a']}->{r['tokens_b']} :: {r['prompt'][:80]!r}")


if __name__ == "__main__":
    main()
