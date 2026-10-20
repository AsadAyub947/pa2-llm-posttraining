"""Task 3 - GRPO continuation from the supplied midpoint adapter.

One update =
  1. take `prompts_per_update` prompts (fixed seeded order, identical across forks) and sample
     K = num_generations completions each (cap = max_completion_length),
  2. score with the frozen reward model, compute *within-group* advantages
     A_k = (r_k - mean_group) / (std_group + eps)  (fixed group_relative_advantages),
  3. mask max-length (truncated) completions out of the loss if configured,
  4. clipped surrogate with sequence normalisation `loss_type` (grpo: 1/T_k; dr_grpo: 1/L_max)
     plus beta * k3-KL to the reference (adapter disabled), `policy_epochs` passes.

With micro_batch_size == 1 the exact gradient norm of each completion's policy term is logged,
which is the length-conditioned gradient statistic used in the normalisation study.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.eval_utils import (
    disable_dropout,
    peak_memory_gib,
    reference_logprobs,
    reset_peak_memory,
    response_logprobs_entropy,
    score_rewards,
    upcast_trainable_to_fp32,
)
from common.generation import batch_generate
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import masked_mean, sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, trainable_parameters
from task3_grpo.grpo import (
    group_relative_advantages,
    group_reward_stats,
    grpo_kl_k3,
    grpo_policy_loss,
    mask_truncated_sequences,
)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    disable_dropout(policy)
    upcast_trainable_to_fp32(policy)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def prompt_schedule(n_prompts: int, n_updates: int, per_update: int, seed: int) -> list[list[int]]:
    order = np.random.default_rng(seed).permutation(n_prompts).tolist()
    out, k = [], 0
    for _ in range(n_updates):
        out.append([order[(k + i) % n_prompts] for i in range(per_update)])
        k += per_update
    return out


def grpo_step(bundle, rows, cfg, loss_type: str):
    policy, tok, opt = bundle["policy"], bundle["tokenizer"], bundle["optimizer"]
    params = trainable_parameters(policy)
    K = int(cfg["num_generations"])
    g = cfg["generation"]
    msgs, gids = [], []
    for i, r in enumerate(rows):
        for _ in range(K):
            msgs.append(prompt_messages(r))
            gids.append(i)
    L = int(cfg["max_completion_length"])
    gen = batch_generate(policy, tok, msgs, max_prompt_length=int(cfg["max_prompt_length"]), max_new_tokens=L,
                         temperature=float(g["temperature"]), top_p=float(g["top_p"]), do_sample=True)
    seq, attn, pw, rid = gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"]
    mask = gen["response_mask"].float()
    dev = seq.device
    rewards = torch.tensor(score_rewards(bundle["reward_model"], bundle["reward_tokenizer"], msgs, gen["responses"],
                                         max_length=int(cfg.get("reward_max_length", 1280))), device=dev)
    group_ids = torch.tensor(gids, device=dev)
    adv = group_relative_advantages(rewards, group_ids)
    stds, uninf = group_reward_stats(rewards, group_ids)
    train_mask = mask_truncated_sequences(mask, gen["truncated"]) if bool(cfg.get("mask_truncated_completions", True)) else mask

    with torch.no_grad():
        ref_logp = reference_logprobs(policy, seq, attn, pw, rid)

    N = seq.shape[0]
    mb = max(1, int(cfg.get("micro_batch_size", 1)))
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    total_train_tok = train_mask.sum().clamp_min(1.0)
    per_seq = [dict() for _ in range(N)]
    epoch_logs = []
    old_logp_all = None
    for epoch in range(int(cfg.get("policy_epochs", 1))):
        opt.zero_grad(set_to_none=True)
        accum = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        stats = {"policy_term": 0.0, "kl_k3": 0.0, "clip_fraction": 0.0, "ent_num": 0.0, "ent_den": 0.0}
        new_all = []
        for s in range(0, N, mb):
            sl = slice(s, s + mb)
            n_mb = min(mb, N - s)
            new_logp, ent = response_logprobs_entropy(policy, seq[sl], attn[sl], pw, rid[sl], with_entropy=(epoch == 0))
            # policy_epochs == 1: old == new (ratio 1); otherwise old = first-epoch log-probs.
            old = new_logp.detach() if old_logp_all is None else old_logp_all[sl]
            new_all.append(new_logp.detach())
            pol_loss, info = grpo_policy_loss(new_logp, old, adv[sl], train_mask[sl], ref_logp[sl], eps, 0.0,
                                              loss_type=loss_type, max_completion_length=L)
            pol_part = pol_loss * (n_mb / N)        # policy term is a mean over sequences
            kl_tok = grpo_kl_k3(new_logp, ref_logp[sl])
            kl_part = (kl_tok * train_mask[sl]).sum() / total_train_tok  # KL is a mean over all tokens
            if pol_part.requires_grad and train_mask[sl].sum() > 0:
                pol_part.backward(retain_graph=True)
                gn_i = 0.0
                for a, p in zip(accum, params):
                    if p.grad is not None:
                        gn_i += float(p.grad.float().pow(2).sum())
                        a.add_(p.grad.float())
                        p.grad = None
                if mb == 1:
                    per_seq[s]["policy_grad_norm"] = gn_i ** 0.5
                (beta * kl_part).backward()
                for a, p in zip(accum, params):
                    if p.grad is not None:
                        a.add_(p.grad.float())
                        p.grad = None
            stats["policy_term"] += float(pol_part.detach())
            stats["kl_k3"] += float(kl_part.detach())
            stats["clip_fraction"] += float(info["clip_fraction"]) * float(train_mask[sl].sum() / total_train_tok)
            if ent is not None:
                stats["ent_num"] += float((ent * mask[sl]).sum())
                stats["ent_den"] += float(mask[sl].sum())
            del new_logp, ent
        for a, p in zip(accum, params):
            p.grad = a.to(p.dtype)
        gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0))))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if old_logp_all is None:
            old_logp_all = torch.cat(new_all)
        epoch_logs.append({**stats, "grad_norm": gn})
        del accum

    lens = mask.sum(-1)
    for i in range(N):
        T = float(lens[i])
        per_seq[i].update({
            "group": int(gids[i]), "length": int(T), "reward": float(rewards[i]), "advantage": float(adv[i]),
            "truncated": bool(gen["truncated"][i]), "masked": bool(train_mask[i].sum() == 0),
            # analytic per-token gradient weight of this completion's policy term
            "token_weight": (abs(float(adv[i])) / max(T, 1.0)) if loss_type == "grpo" else abs(float(adv[i])) / L,
        })
    e0 = epoch_logs[0]
    return {
        "reward_mean": float(rewards.mean()), "reward_std_all": float(rewards.std(unbiased=False)),
        "group_reward_std_mean": float(np.mean(stds)), "uninformative_group_frac": float(np.mean(uninf)),
        "kl_token_mean": float(sampled_kl(old_logp_all, ref_logp, mask)),
        "kl_seq_mean": float(((old_logp_all - ref_logp) * mask).sum(-1).mean()),
        "kl_k3": e0["kl_k3"],
        "policy_loss": e0["policy_term"] + beta * e0["kl_k3"], "policy_term": e0["policy_term"],
        "grad_norm": e0["grad_norm"], "clip_fraction": e0["clip_fraction"],
        "entropy": e0["ent_num"] / max(e0["ent_den"], 1.0),
        "sample_entropy": float(-masked_mean(old_logp_all, mask)),
        "response_length": float(lens.mean()), "response_length_std": float(lens.std(unbiased=False)),
        "truncated_frac": float(np.mean(gen["truncated"])),
        "masked_completions": int(sum(1 for i in range(N) if train_mask[i].sum() == 0)),
        "generated_tokens": int(lens.sum()), "per_completion": per_seq, "epochs": epoch_logs,
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo",
             run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    n_updates, seed = int(cfg["updates"]), int(cfg["seed"])
    rows_all = bundle["prompt_rows"]
    schedule = prompt_schedule(len(rows_all), n_updates, int(cfg["prompts_per_update"]), seed)
    rdir = repo_path(cfg["results_dir"]) / run_name
    rdir.mkdir(parents=True, exist_ok=True)
    log_path = rdir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    run_config = {k: cfg.get(k) for k in ["updates", "prompts_per_update", "policy_epochs", "learning_rate",
                                          "clip_epsilon", "kl_beta", "num_generations", "max_prompt_length",
                                          "max_completion_length", "mask_truncated_completions", "max_grad_norm",
                                          "seed", "generation", "lora", "micro_batch_size"]}
    run_config.update({"run_name": run_name, "loss_type": loss_type, "output": str(out),
                       "start_policy": cfg["paths"]["grpo_midpoint_policy"], "reward_model": cfg["reward_model"],
                       "advantage": "within-group (r-mean)/(std+1e-6)", "dropout": "disabled"})
    save_json(rdir / "run_config.json", run_config)
    print(f"[grpo:{run_name}] {n_updates} updates, loss_type={loss_type}, K={cfg['num_generations']}")

    reset_peak_memory()
    t0 = time.perf_counter()
    gen_tokens = 0
    for u in range(n_updates):
        set_seed(seed + u)
        rows = [rows_all[i] for i in schedule[u]]
        tu = time.perf_counter()
        m = grpo_step(bundle, rows, cfg, loss_type)
        gen_tokens += m["generated_tokens"]
        rec = {"update": u + 1, "prompt_source_index": [r.get("source_index") for r in rows],
               "prompt_id": [r.get("prompt_id") for r in rows], **m,
               "cumulative_generated_tokens": gen_tokens, "update_seconds": time.perf_counter() - tu,
               "elapsed_s": time.perf_counter() - t0, "peak_vram_gib_so_far": peak_memory_gib()}
        append_jsonl(log_path, rec)
        print(f"[grpo:{run_name}] u{u + 1:02d} R {m['reward_mean']:+.3f} σg {m['group_reward_std_mean']:.3f} "
              f"uninf {m['uninformative_group_frac']:.2f} KL {m['kl_token_mean']:.4f} H {m['entropy']:.3f} "
              f"len {m['response_length']:.0f} loss {m['policy_loss']:+.4f} gn {m['grad_norm']:.3f} masked {m['masked_completions']}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wall = time.perf_counter() - t0
    peak = peak_memory_gib()
    bundle["policy"].save_pretrained(str(out))
    bundle["tokenizer"].save_pretrained(str(out))
    summary = {**run_config, "wall_clock_s": wall, "peak_vram_gib": peak, "generated_tokens": gen_tokens,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    save_json(rdir / "train_summary.json", summary)
    print(f"[grpo:{run_name}] done in {wall / 60:.1f} min, peak VRAM {peak} GiB -> {out}")
    for k in list(bundle):
        bundle[k] = None
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
