"""Task 1 common evaluation protocol.

For one policy (an adapter path, or `none` for the untouched SFT model) this computes:

1. Held-out pairs (teacher forcing): DPO loss at the run's beta, preference accuracy
   (fraction with m_theta > 0), mean margin, implicit chosen/rejected rewards. If the pair file
   has `length_stratum` (the length-stratified eval), everything is also broken down by stratum.
2. Generation on a fixed prefix of held-out prompts: reward-model score, KL from reference
   (released sampled-response estimator), entropy, response-length statistics.
3. Word-limit prompts: greedy + sampled responses, word counts, explicit compliance.

Outputs (results/task1_dpo/eval/<name>/): summary.json, pairs.jsonl, generations.jsonl,
word_limit.jsonl.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from peft import PeftModel

from common.data import (
    encode_prompt_response,
    load_yaml,
    preference_responses,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
    pad_batch,
)
from common.eval_utils import evaluate_generations, length_stats, select_eval_rows
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.metrics import word_count, word_limit_compliance, parse_word_limit
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import pair_logps, prompt_fits


def load_evaluation_bundle(config_path: str, adapter: str | None, load_reward: bool = True):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg) if load_reward else None,
    }


def normalize_adapter(adapter: str | None):
    if adapter is None or str(adapter).lower() in {"none", "sft", "base", ""}:
        return None
    return adapter


# --------------------------------------------------------------------------------------
# 1. held-out pairs
# --------------------------------------------------------------------------------------

@torch.no_grad()
def score_pairs(policy, tokenizer, rows, beta: float, max_length: int, batch_size: int = 8):
    device = next(policy.parameters()).device
    records = []
    usable = [(i, r) for i, r in enumerate(rows) if prompt_fits(tokenizer, r, max_length)]
    for start in range(0, len(usable), batch_size):
        chunk = usable[start:start + batch_size]
        chosen, rejected = [], []
        for _, row in chunk:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        cb, rb = pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
        pc, pr = pair_logps(policy, cb, rb, tokenizer.pad_token_id, device)
        if isinstance(policy, PeftModel):
            with reference_mode(policy):
                rc, rr = pair_logps(policy, cb, rb, tokenizer.pad_token_id, device)
        else:
            rc, rr = pc.clone(), pr.clone()
        for j, (idx, row) in enumerate(chunk):
            margin = float((pc[j] - rc[j]) - (pr[j] - rr[j]))
            loss_j, _ = dpo_loss(pc[j:j + 1], pr[j:j + 1], rc[j:j + 1], rr[j:j + 1], beta)
            records.append({
                "row_index": idx,
                "prompt_id": row.get("prompt_id"),
                "source_index": row.get("source_index"),
                "length_stratum": row.get("length_stratum"),
                "chosen_tokens": int(cb["response_mask"][j].sum().item()),
                "rejected_tokens": int(rb["response_mask"][j].sum().item()),
                "policy_chosen_logp": float(pc[j]), "policy_rejected_logp": float(pr[j]),
                "ref_chosen_logp": float(rc[j]), "ref_rejected_logp": float(rr[j]),
                "margin": margin,
                "chosen_reward": beta * float(pc[j] - rc[j]),
                "rejected_reward": beta * float(pr[j] - rr[j]),
                "dpo_loss": float(loss_j),
                "correct": margin > 0,
            })
    return records, len(rows) - len(usable)


def summarize_pairs(records: list[dict]) -> dict:
    def agg(rs):
        if not rs:
            return {"n": 0}
        m = np.array([r["margin"] for r in rs])
        return {
            "n": len(rs),
            "dpo_loss": float(np.mean([r["dpo_loss"] for r in rs])),
            "preference_accuracy": float(np.mean(m > 0)),
            "tie_rate": float(np.mean(m == 0)),
            "margin_mean": float(m.mean()), "margin_std": float(m.std()),
            "chosen_reward_mean": float(np.mean([r["chosen_reward"] for r in rs])),
            "rejected_reward_mean": float(np.mean([r["rejected_reward"] for r in rs])),
            "chosen_tokens_mean": float(np.mean([r["chosen_tokens"] for r in rs])),
            "rejected_tokens_mean": float(np.mean([r["rejected_tokens"] for r in rs])),
            # Does the learned margin track the length difference of the pair?
            "corr_margin_vs_length_diff": float(np.corrcoef(
                m, [r["chosen_tokens"] - r["rejected_tokens"] for r in rs])[0, 1]) if len(rs) > 2 and m.std() > 0 else float("nan"),
        }

    out = {"overall": agg(records)}
    strata = defaultdict(list)
    for r in records:
        if r.get("length_stratum"):
            strata[r["length_stratum"]].append(r)
    if strata:
        out["by_stratum"] = {k: agg(v) for k, v in sorted(strata.items())}
    return out


# --------------------------------------------------------------------------------------
# 2./3. generation-based metrics
# --------------------------------------------------------------------------------------

def generation_rows_from_pairs(rows, n):
    """Unique held-out prompts (first occurrence order), as {'prompt_id','messages'} rows."""
    seen, out = set(), []
    for r in rows:
        pid = r.get("prompt_id", r.get("source_index"))
        if pid in seen:
            continue
        seen.add(pid)
        out.append({"prompt_id": pid, "messages": prompt_messages_from_preference(r)})
    return select_eval_rows(out, n)


@torch.no_grad()
def word_limit_eval(policy, tokenizer, cfg, n_samples: int):
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    prompts = [prompt_messages(r) for r in rows]
    records = []
    max_new = int(cfg["max_generation_tokens"])
    seed = int(cfg["seed"])
    # greedy (deterministic)
    gen = batch_generate(policy, tokenizer, prompts, max_prompt_length=256, max_new_tokens=max_new, do_sample=False)
    for r, p, txt, n_tok, trunc in zip(rows, prompts, gen["responses"], gen["response_lengths"], gen["truncated"]):
        records.append({"prompt_id": r["prompt_id"], "decoding": "greedy", "sample": 0,
                        "word_limit": parse_word_limit(p[-1]["content"]), "words": word_count(txt),
                        "response_tokens": int(n_tok), "truncated": bool(trunc),
                        "compliant": word_limit_compliance(p[-1]["content"], txt), "response": txt})
    # sampled (fixed seed, base-config sampling)
    g = cfg["generation"]
    for s in range(n_samples):
        set_seed(seed + 1000 + s)
        gen = batch_generate(policy, tokenizer, prompts, max_prompt_length=256, max_new_tokens=max_new,
                             temperature=float(g["temperature"]), top_p=float(g["top_p"]), do_sample=True)
        for r, p, txt, n_tok, trunc in zip(rows, prompts, gen["responses"], gen["response_lengths"], gen["truncated"]):
            records.append({"prompt_id": r["prompt_id"], "decoding": "sampled", "sample": s + 1,
                            "word_limit": parse_word_limit(p[-1]["content"]), "words": word_count(txt),
                            "response_tokens": int(n_tok), "truncated": bool(trunc),
                            "compliant": word_limit_compliance(p[-1]["content"], txt), "response": txt})

    def agg(rs):
        comp = [r["compliant"] for r in rs if r["compliant"] is not None]
        return {
            "n": len(rs),
            "compliance_rate": float(np.mean(comp)) if comp else float("nan"),
            "words": length_stats([r["words"] for r in rs]),
            "words_over_limit_mean": float(np.mean([max(0, r["words"] - r["word_limit"]) for r in rs if r["word_limit"]])),
            "ratio_words_to_limit_mean": float(np.mean([r["words"] / r["word_limit"] for r in rs if r["word_limit"]])),
            "response_tokens": length_stats([r["response_tokens"] for r in rs]),
        }

    summary = {
        "greedy": agg([r for r in records if r["decoding"] == "greedy"]),
        "sampled": agg([r for r in records if r["decoding"] == "sampled"]),
        "n_prompts": len(rows), "n_samples_per_prompt": n_samples,
    }
    return summary, records


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------

def evaluate_policy(config_path: str, adapter: str | None, name: str, beta: float | None = None,
                    pairs_file: str | None = None, extra_pairs_files: dict | None = None,
                    skip_generation: bool = False, skip_word_limit: bool = False):
    adapter = normalize_adapter(adapter)
    cfg = load_yaml(config_path)
    beta = float(cfg["beta"] if beta is None else beta)
    outdir = repo_path(cfg["results_dir"]) / "eval" / name
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = load_evaluation_bundle(config_path, adapter, load_reward=not skip_generation)
    policy, tok = bundle["policy"], bundle["tokenizer"]
    max_len = int(cfg["max_sequence_length"])
    summary = {"name": name, "adapter": adapter, "beta_for_loss": beta, "seed": int(cfg["seed"])}

    pair_sets = {"standard_eval": pairs_file or cfg["paths"]["dpo_standard_eval"]}
    pair_sets.update(extra_pairs_files or {})
    for set_name, path in pair_sets.items():
        rows = select_eval_rows(read_jsonl(path), cfg.get("eval_pairs"))
        recs, skipped = score_pairs(policy, tok, rows, beta, max_len, int(cfg.get("eval_batch_size", 8)))
        write_jsonl(outdir / f"pairs_{set_name}.jsonl", recs)
        summary[f"pairs_{set_name}"] = {**summarize_pairs(recs), "file": str(path), "skipped_long_prompt": skipped}
        o = summary[f"pairs_{set_name}"]["overall"]
        print(f"[eval:{name}] {set_name}: acc {o.get('preference_accuracy', float('nan')):.3f} "
              f"loss {o.get('dpo_loss', float('nan')):.4f} n={o['n']}")

    if not skip_generation:
        g = cfg["generation"]
        gen_rows = generation_rows_from_pairs(bundle["rows"], cfg.get("eval_generation_prompts"))
        gsum, grecs = evaluate_generations(
            policy, tok, gen_rows, reward=bundle["reward"], max_prompt_length=256,
            max_new_tokens=int(cfg["max_generation_tokens"]), batch_size=int(cfg.get("generation_batch_size", 8)),
            temperature=float(g["temperature"]), top_p=float(g["top_p"]), do_sample=bool(g["do_sample"]),
            seed=int(cfg["seed"]), reward_max_length=1280,
        )
        write_jsonl(outdir / "generations.jsonl", grecs)
        summary["generation"] = gsum
        print(f"[eval:{name}] reward {gsum['reward']['mean']:.3f} KL(tok) {gsum['kl_token_mean']:.4f} "
              f"len {gsum['length']['mean']:.1f}±{gsum['length']['std']:.1f}")

    if not skip_word_limit:
        wsum, wrecs = word_limit_eval(policy, tok, cfg, int(cfg.get("word_limit_samples", 4)))
        write_jsonl(outdir / "word_limit.jsonl", wrecs)
        summary["word_limit"] = wsum
        print(f"[eval:{name}] word-limit compliance greedy {wsum['greedy']['compliance_rate']:.2f} "
              f"sampled {wsum['sampled']['compliance_rate']:.2f}")

    save_json(outdir / "summary.json", summary)
    clear_gpu(bundle["policy"], bundle["reward"])
    del bundle
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the SFT starting policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used to report held-out DPO loss (default: config beta)")
    ap.add_argument("--with-length-strata", action="store_true", help="also score the length-stratified eval pairs")
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--skip-word-limit", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    extra = {"length_stratified": cfg["paths"]["dpo_length_eval"]} if args.with_length_strata else None
    evaluate_policy(args.config, args.adapter, args.name, args.beta, extra_pairs_files=extra,
                    skip_generation=args.skip_generation, skip_word_limit=args.skip_word_limit)


if __name__ == "__main__":
    main()
