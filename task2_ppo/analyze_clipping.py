"""Task 2 Step 2 - clipping study.

Part A (cached batch, immediate geometry). The supplied cached rollout batch is rebuilt into
token tensors; advantages/returns are computed exactly as in training (KL shaping with the
config beta_KL, GAE with the midpoint critic, whitening). Then for every epsilon:
  * step 0: ratio = pi_midpoint / pi_old(cached) -> clip fraction, affected-token fraction,
    clipped vs unclipped surrogate (no update yet; only epsilon differs);
  * steps 1..`cached_inner_steps`: from the identical midpoint weights, take optimiser steps on
    the *same* fixed batch with that epsilon and record the same quantities after each step.

Part B (matched short forks). From the same midpoint (policy + critic), run `fork_updates`
updates for each epsilon with identical prompts, seeds, reward and beta_KL, then evaluate each
fork with the common held-out protocol. Stability statistics are defined in fork_utils.

Outputs: results/task2_ppo/clipping_cached.json, clipping_forks_summary.{csv,json}.
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.eval_utils import (
    critic_values,
    disable_dropout,
    reference_logprobs,
    response_logprobs_entropy,
    score_rewards,
    upcast_trainable_to_fp32,
)
from common.logging_utils import save_json, set_seed
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, load_value_model, trainable_parameters
from task2_ppo.fork_utils import eval_row, load_train_log, stability_stats, train_or_reuse_and_eval
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_clip_diagnostics, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


# --------------------------------------------------------------------------------------
# rebuilding the cached batch
# --------------------------------------------------------------------------------------

def _first(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _as_list(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().flatten().tolist()
    return list(x)


def rebuild_cached_batch(rows, tokenizer, cfg, prompt_lookup):
    """Return padded tensors (left-padded prompt + right-padded response) for the cached batch."""
    examples = []
    report = {"n_rows": len(rows), "keys": sorted(rows[0].keys()), "response_from": None, "prompt_from": None,
              "length_fixups": 0}
    for row in rows:
        old = torch.as_tensor(_as_list(row["old_logprobs"]), dtype=torch.float32)
        ref = torch.as_tensor(_as_list(row["ref_logprobs"]), dtype=torch.float32)
        # ---- prompt ids
        p_ids = _as_list(_first(row, ["query_ids", "prompt_ids", "query_input_ids", "prompt_input_ids"]))
        msgs = None
        if p_ids is None:
            msgs = _first(row, ["messages", "prompt_messages"])
            if msgs is None and isinstance(row.get("prompt"), list):
                msgs = row["prompt"]
            if msgs is None:
                src = prompt_lookup.get(str(row["source_index"]))
                if src is None:
                    raise KeyError(f"cannot find prompt for cached source_index={row['source_index']}")
                msgs = prompt_messages(src)
            p_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
            p_ids = p_ids[-int(cfg["max_prompt_length"]):]
            report["prompt_from"] = "messages/source_index"
        else:
            report["prompt_from"] = "cached ids"
        # ---- response ids
        r_ids = _as_list(_first(row, ["response_ids", "response_token_ids", "response_input_ids"]))
        if r_ids is None:
            r_ids = tokenizer(str(row["response"]), add_special_tokens=False)["input_ids"]
            report["response_from"] = "retokenized text"
        else:
            report["response_from"] = "cached ids"
        n = int(old.numel())
        mask_cached = _as_list(_first(row, ["response_mask", "mask"]))
        if mask_cached is not None:
            n = int(sum(mask_cached))
        if len(r_ids) == n - 1 and tokenizer.eos_token_id is not None:
            r_ids = list(r_ids) + [tokenizer.eos_token_id]  # cached logprobs include the EOS token
        if len(r_ids) != n:
            report["length_fixups"] += 1
            r_ids = list(r_ids)[:n]
            n = len(r_ids)
        reward = _first(row, ["reward", "score", "rm_score", "task_reward", "scores", "rewards"])
        if reward is not None and torch.is_tensor(reward):
            reward = float(reward.flatten()[-1]) if reward.numel() else None
        elif isinstance(reward, (list, tuple)):
            reward = float(reward[-1])
        values = _as_list(_first(row, ["values", "old_values", "value"]))
        examples.append({
            "p_ids": list(p_ids), "r_ids": list(r_ids), "old": old[:n], "ref": ref[:n], "msgs": msgs,
            "response": str(row["response"]), "reward": reward,
            "values": torch.as_tensor(values[:n], dtype=torch.float32) if values is not None and len(values) >= n else None,
            "terminated": bool(row.get("terminated_with_eos", row.get("terminated", True))),
            "source_index": row["source_index"],
        })

    P = max(len(e["p_ids"]) for e in examples)
    T = max(len(e["r_ids"]) for e in examples)
    pad = tokenizer.pad_token_id
    B = len(examples)
    seq = torch.full((B, P + T), pad, dtype=torch.long)
    attn = torch.zeros((B, P + T), dtype=torch.long)
    rid = torch.full((B, T), pad, dtype=torch.long)
    mask = torch.zeros((B, T))
    old = torch.zeros((B, T))
    ref = torch.zeros((B, T))
    for b, e in enumerate(examples):
        lp, lr = len(e["p_ids"]), len(e["r_ids"])
        seq[b, P - lp:P] = torch.tensor(e["p_ids"])
        attn[b, P - lp:P + lr] = 1
        seq[b, P:P + lr] = torch.tensor(e["r_ids"])
        rid[b, :lr] = torch.tensor(e["r_ids"])
        mask[b, :lr] = 1
        old[b, :lr] = e["old"]
        ref[b, :lr] = e["ref"]
    return {"examples": examples, "seq": seq, "attn": attn, "pw": P, "rid": rid, "mask": mask,
            "old": old, "ref": ref, "report": report}


def cached_study(cfg, args):
    tok = load_tokenizer(cfg["base_model"])
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Cache keys:", sorted(rows[0].keys()))
    lookup = {}
    for key in ["rl_prompt_train", "rl_prompt_eval"]:
        for r in read_jsonl(cfg["paths"][key]):
            lookup.setdefault(str(r.get("source_index")), r)
    batch = rebuild_cached_batch(rows, tok, cfg, lookup)
    print("Rebuild report:", batch["report"])

    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    disable_dropout(policy)
    upcast_trainable_to_fp32(policy)
    dev = next(policy.parameters()).device
    seq, attn, rid = batch["seq"].to(dev), batch["attn"].to(dev), batch["rid"].to(dev)
    mask, old, ref_cached = batch["mask"].to(dev), batch["old"].to(dev), batch["ref"].to(dev)
    pw = batch["pw"]
    beta = float(cfg["kl_beta"])

    # ---- rewards (cached if present, otherwise the frozen RM) and critic values
    ex = batch["examples"]
    if all(e["reward"] is not None for e in ex):
        rm = [float(e["reward"]) for e in ex]
        reward_src = "cache"
    else:
        rmodel, rtok = load_reward_model(cfg)
        msgs = [e["msgs"] or prompt_messages(lookup[str(e["source_index"])]) for e in ex]
        rm = score_rewards(rmodel, rtok, msgs, [e["response"] for e in ex], max_length=int(cfg.get("reward_max_length", 1280)))
        rmodel = None
        clear_gpu()
        reward_src = "reward_model"
    pen = float(cfg.get("missing_eos_penalty", 0.0))
    task_reward = torch.tensor([r - (0.0 if e["terminated"] else pen) for r, e in zip(rm, ex)], device=dev)
    if all(e["values"] is not None for e in ex):
        values = torch.zeros_like(mask)
        for b, e in enumerate(ex):
            values[b, :len(e["values"])] = e["values"].to(dev)
        value_src = "cache"
    else:
        vm = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
        with torch.no_grad():
            values = torch.cat([critic_values(vm, seq[b:b + 1], attn[b:b + 1])[:, pw - 1:-1][:, :rid.shape[1]]
                                for b in range(seq.shape[0])]) * mask
        vm = None
        clear_gpu()
        value_src = "midpoint_critic"

    rewards = shaped_rewards(task_reward, old, ref_cached, mask, beta)
    adv, returns = compute_gae(rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    adv = normalize_advantages(adv, mask)

    # sanity: does the cached reference agree with our adapter-disabled reference?
    with torch.no_grad():
        ref_now = torch.cat([reference_logprobs(policy, seq[b:b + 1], attn[b:b + 1], pw, rid[b:b + 1]) for b in range(seq.shape[0])])
    ref_check = float(((ref_now - ref_cached).abs() * mask).sum() / mask.sum())

    init_state = copy.deepcopy({k: v.detach().clone() for k, v in policy.state_dict().items() if "lora_" in k})
    params = trainable_parameters(policy)
    lr = float(cfg["policy_learning_rate"])
    results = {"reward_source": reward_src, "value_source": value_src, "kl_beta": beta,
               "rebuild": batch["report"], "mean_abs_ref_logprob_diff_vs_cache": ref_check,
               "valid_tokens": int(mask.sum()), "n_sequences": int(mask.shape[0]),
               "inner_steps": int(cfg.get("cached_inner_steps", 4)), "learning_rate": lr, "by_eps": {}}

    def batch_logp():
        return torch.cat([response_logprobs_entropy(policy, seq[b:b + 1], attn[b:b + 1], pw, rid[b:b + 1])[0]
                          for b in range(seq.shape[0])])

    for eps in [float(e) for e in cfg["clip_values"]]:
        policy.load_state_dict(init_state, strict=False)
        opt = AdamW(params, lr=lr)
        traj = []
        with torch.no_grad():
            d0 = ppo_clip_diagnostics(batch_logp(), old, adv, mask, eps=eps)
        traj.append({"step": 0, **d0})
        for step in range(1, results["inner_steps"] + 1):
            opt.zero_grad(set_to_none=True)
            tot = mask.sum()
            for b in range(seq.shape[0]):
                lp, _ = response_logprobs_entropy(policy, seq[b:b + 1], attn[b:b + 1], pw, rid[b:b + 1])
                loss, _, _ = ppo_policy_loss(lp, old[b:b + 1], adv[b:b + 1], mask[b:b + 1], eps=eps)
                (loss * mask[b].sum() / tot).backward()
            gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0))))
            opt.step()
            with torch.no_grad():
                d = ppo_clip_diagnostics(batch_logp(), old, adv, mask, eps=eps)
            traj.append({"step": step, "grad_norm_pre_clip": gn, **d})
        results["by_eps"][str(eps)] = traj
        print(f"[clip-cache] eps={eps}: step0 clip {traj[0]['clip_fraction']:.4f} affected {traj[0]['affected_fraction']:.4f} | "
              f"after {results['inner_steps']} steps clip {traj[-1]['clip_fraction']:.4f} affected {traj[-1]['affected_fraction']:.4f}")
    policy.load_state_dict(init_state, strict=False)
    policy = None
    clear_gpu()
    save_json(repo_path(cfg["results_dir"]) / "clipping_cached.json", results)
    rows_out = []
    for eps, traj in results["by_eps"].items():
        for t in traj:
            rows_out.append({"eps": float(eps), **t})
    pd.DataFrame(rows_out).to_csv(repo_path(cfg["results_dir"]) / "clipping_cached.csv", index=False)
    return results


def fork_study(cfg, args):
    from task2_ppo.continue_train import run_ppo
    from task2_ppo.evaluate import evaluate

    rows = []
    for eps in [float(e) for e in cfg["clip_values"]]:
        name = f"clip_{eps:g}"
        out = f"outputs/task2_ppo/{name}"
        ev = train_or_reuse_and_eval(
            cfg, args.config, name, out,
            lambda: run_ppo(args.config, output=out, updates=int(cfg["fork_updates"]), clip_epsilon=eps, run_name=name),
            lambda: evaluate(args.config, out, name), skip_train=args.skip_train, force=args.force)
        st = stability_stats(load_train_log(cfg, name))
        rows.append({"eps": eps, **eval_row(ev), **st})
    df = pd.DataFrame(rows)
    rdir = repo_path(cfg["results_dir"])
    df.to_csv(rdir / "clipping_forks_summary.csv", index=False)
    save_json(rdir / "clipping_forks_summary.json", rows)
    print(df.T.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-cached", action="store_true")
    ap.add_argument("--skip-forks", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="forks already trained; only re-evaluate")
    ap.add_argument("--force", action="store_true", help="retrain and re-evaluate forks even if results exist")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    print("Required epsilon values:", cfg["clip_values"])
    if not args.skip_cached:
        cached_study(cfg, args)
    if not args.skip_forks:
        fork_study(cfg, args)


if __name__ == "__main__":
    main()
