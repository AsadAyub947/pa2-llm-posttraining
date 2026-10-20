from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer
from common.rl_eval import evaluate_rl_policy


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate(config_path: str, adapter: str, name: str):
    cfg = load_yaml(config_path)
    return evaluate_rl_policy(cfg, adapter, name, max_new_tokens=int(cfg.get("eval_max_response_length", 768)),
                              midpoint_key="grpo_midpoint_policy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, 'midpoint', or 'none' (SFT)")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()
