"""Common held-out protocol for the online-RL policies (Task 2 PPO and Task 3 GRPO).

Same prompts (fixed prefix of rl_prompt_pool_eval), same decoding (base-config sampling with a
fixed seed), same generation cap, same reward model and the same KL/entropy/length conventions
for every condition, so standard runs and short forks are directly comparable.
"""
from __future__ import annotations

import numpy as np

from common.data import read_jsonl, repo_path, write_jsonl
from common.eval_utils import evaluate_generations, select_eval_rows
from common.logging_utils import save_json
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer


def resolve_adapter(cfg, adapter: str | None, midpoint_key: str):
    if adapter is None or str(adapter).lower() in {"none", "sft", "base"}:
        return None
    if str(adapter).lower() == "midpoint":
        return cfg["paths"][midpoint_key]
    return adapter


def evaluate_rl_policy(cfg: dict, adapter: str | None, name: str, *, max_new_tokens: int, midpoint_key: str,
                       reward=None, tokenizer=None, missing_eos_penalty: float = 0.0):
    adapter = resolve_adapter(cfg, adapter, midpoint_key)
    rows = select_eval_rows(read_jsonl(cfg["paths"]["rl_prompt_eval"]), cfg.get("eval_prompts"))
    tok = tokenizer or load_tokenizer(cfg["base_model"])
    own_reward = reward is None
    reward = reward or load_reward_model(cfg)
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    g = cfg["generation"]
    summary, recs = evaluate_generations(
        policy, tok, rows, reward=reward, max_prompt_length=int(cfg.get("max_prompt_length", 256)),
        max_new_tokens=int(max_new_tokens), batch_size=int(cfg.get("eval_batch_size", 4)),
        temperature=float(g["temperature"]), top_p=float(g["top_p"]), do_sample=bool(g["do_sample"]),
        seed=int(cfg["seed"]), reward_max_length=int(cfg.get("reward_max_length", 1280)),
    )
    pen = [r["reward"] - (missing_eos_penalty if not r["terminated_with_eos"] else 0.0) for r in recs]
    summary["reward_with_eos_penalty_mean"] = float(np.mean(pen)) if pen else float("nan")
    summary.update({"name": name, "adapter": adapter, "n_eval_prompts": len(rows)})
    outdir = repo_path(cfg["results_dir"]) / "eval" / name
    outdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(outdir / "generations.jsonl", recs)
    save_json(outdir / "summary.json", summary)
    print(f"[eval:{name}] RM {summary['reward']['mean']:+.3f}±{summary['reward']['sem']:.3f} "
          f"KL(tok) {summary['kl_token_mean']:.4f} H {summary['entropy_token_mean']:.3f} "
          f"len {summary['length']['mean']:.0f}±{summary['length']['std']:.0f} trunc {summary['truncation_rate']:.2f}")
    policy = None
    if own_reward:
        reward = None
    clear_gpu()
    return summary
