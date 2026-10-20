"""Task 2 - PPO continuation from the supplied midpoint (policy LoRA + critic).

One update =
  1. sample `prompts_per_update` prompts (fixed seeded order, identical across forks),
  2. generate on-policy responses (base-config sampling, cap = max_response_length),
  3. old log-probs (current policy), reference log-probs (adapter disabled), critic values,
     learned reward (frozen RM; missing-EOS penalty for truncated responses),
  4. KL-shaped token rewards r_t = R*1[t=T] - beta_KL (log pi - log ref), GAE(gamma, lambda),
     returns = A + V, advantage whitening,
  5. `ppo_epochs` passes of clipped-surrogate policy updates + critic MSE updates.

Dropout is disabled in policy and critic so the first-epoch ratio is exactly 1 (TRL convention).
Every quantity the manual asks for is logged per update to results/task2_ppo/<run>/train_log.jsonl.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.eval_utils import (
    critic_values,
    disable_dropout,
    explained_variance,
    grad_norm,
    peak_memory_gib,
    reference_logprobs,
    reset_peak_memory,
    response_logprobs_entropy,
    score_rewards,
    upcast_trainable_to_fp32,
)
from common.generation import batch_generate
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import masked_mean
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_clip_diagnostics,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    disable_dropout(policy)
    disable_dropout(value_model)
    upcast_trainable_to_fp32(value_model)
    upcast_trainable_to_fp32(policy)

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def prompt_schedule(n_prompts: int, n_updates: int, per_update: int, seed: int) -> list[list[int]]:
    """Fixed seeded permutation of the train prompt pool; identical for every fork."""
    order = np.random.default_rng(seed).permutation(n_prompts).tolist()
    out, k = [], 0
    for _ in range(n_updates):
        idx = []
        for _ in range(per_update):
            idx.append(order[k % n_prompts])
            k += 1
        out.append(idx)
    return out


@torch.no_grad()
def collect_rollout(bundle, rows, cfg):
    policy, tok = bundle["policy"], bundle["tokenizer"]
    g = cfg["generation"]
    msgs = [prompt_messages(r) for r in rows]
    gen = batch_generate(policy, tok, msgs, max_prompt_length=int(cfg["max_prompt_length"]),
                         max_new_tokens=int(cfg["max_response_length"]), temperature=float(g["temperature"]),
                         top_p=float(g["top_p"]), do_sample=True)
    seq, attn, pw, rid = gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"]
    mask = gen["response_mask"].float()
    old_logp, ent = response_logprobs_entropy(policy, seq, attn, pw, rid, with_entropy=True)
    ref_logp = reference_logprobs(policy, seq, attn, pw, rid)
    T = rid.shape[1]
    values = critic_values(bundle["value_model"], seq, attn)[:, pw - 1: -1][:, :T] * mask
    rm = score_rewards(bundle["reward_model"], bundle["reward_tokenizer"], msgs, gen["responses"],
                       max_length=int(cfg.get("reward_max_length", 1280)))
    penalty = float(cfg.get("missing_eos_penalty", 0.0))
    task_reward = [r - (penalty if not eos else 0.0) for r, eos in zip(rm, gen["terminated_with_eos"])]
    return {
        "msgs": msgs, "gen": gen, "seq": seq, "attn": attn, "pw": pw, "rid": rid, "mask": mask,
        "old_logp": old_logp, "ref_logp": ref_logp, "entropy": ent, "values": values.float(),
        "rm_score": rm, "task_reward": torch.tensor(task_reward, device=seq.device, dtype=torch.float32),
    }


def ppo_update(bundle, ro, cfg, eps: float, kl_beta: float):
    """KL shaping + GAE + `ppo_epochs` clipped updates. Returns a metrics dict."""
    policy, value_model = bundle["policy"], bundle["value_model"]
    p_opt, v_opt = bundle["policy_optimizer"], bundle["value_optimizer"]
    p_params, v_params = trainable_parameters(policy), trainable_parameters(value_model)
    mask, old_logp, ref_logp = ro["mask"], ro["old_logp"], ro["ref_logp"]

    rewards = shaped_rewards(ro["task_reward"], old_logp, ref_logp, mask, kl_beta)
    adv, returns = compute_gae(rewards, ro["values"], mask, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]))
    raw_adv = adv.clone()
    if bool(cfg.get("whiten_advantages", True)):
        adv = normalize_advantages(adv, mask)

    B = mask.shape[0]
    mb = max(1, int(cfg.get("micro_batch_size", 1)))
    total_tok = mask.sum().clamp_min(1.0)
    max_grad = float(cfg.get("max_grad_norm", 1.0))
    vcoef = float(cfg.get("value_coef", 0.5))
    epochs_log = []
    for epoch in range(int(cfg["ppo_epochs"])):
        p_opt.zero_grad(set_to_none=True)
        v_opt.zero_grad(set_to_none=True)
        acc = {"policy_loss": 0.0, "value_loss": 0.0}
        diag_acc = {}
        for s in range(0, B, mb):
            sl = slice(s, s + mb)
            w = mask[sl].sum() / total_tok  # exact token-level normalisation across micro-batches
            new_logp, _ = response_logprobs_entropy(policy, ro["seq"][sl], ro["attn"][sl], ro["pw"], ro["rid"][sl])
            loss, _, _ = ppo_policy_loss(new_logp, old_logp[sl], adv[sl], mask[sl], eps=eps)
            (loss * w).backward()
            d = ppo_clip_diagnostics(new_logp.detach(), old_logp[sl], adv[sl], mask[sl], eps=eps)
            for k, v in d.items():
                if k != "valid_tokens":
                    diag_acc[k] = diag_acc.get(k, 0.0) + float(v) * float(w)
            acc["policy_loss"] += float(loss.item()) * float(w)

            v_new = critic_values(value_model, ro["seq"][sl], ro["attn"][sl])[:, ro["pw"] - 1: -1][:, :mask.shape[1]]
            vloss = value_mse_loss(v_new, returns[sl], mask[sl])
            (vcoef * vloss * w).backward()
            acc["value_loss"] += float(vloss.item()) * float(w)
            del new_logp, v_new

        pgn = grad_norm(p_params)
        vgn = grad_norm(v_params)
        torch.nn.utils.clip_grad_norm_(p_params, max_grad)
        torch.nn.utils.clip_grad_norm_(v_params, max_grad)
        p_opt.step()
        v_opt.step()
        epochs_log.append({"epoch": epoch, **acc, **diag_acc, "policy_grad_norm": pgn, "value_grad_norm": vgn})

    # how far did this update move the policy on its own batch? (stability statistic)
    with torch.no_grad():
        post = []
        for s in range(0, B, mb):
            sl = slice(s, s + mb)
            lp, _ = response_logprobs_entropy(policy, ro["seq"][sl], ro["attn"][sl], ro["pw"], ro["rid"][sl])
            post.append(lp)
        post = torch.cat(post)
        log_ratio = post - old_logp
        approx_kl_old_new = float(masked_mean(torch.exp(log_ratio) - 1.0 - log_ratio, mask))
        post_diag = ppo_clip_diagnostics(post, old_logp, adv, mask, eps=eps)

    return {
        "epochs": epochs_log,
        "policy_loss": float(np.mean([e["policy_loss"] for e in epochs_log])),
        "value_loss": float(np.mean([e["value_loss"] for e in epochs_log])),
        "policy_grad_norm": float(np.mean([e["policy_grad_norm"] for e in epochs_log])),
        "policy_grad_norm_max": float(np.max([e["policy_grad_norm"] for e in epochs_log])),
        "value_grad_norm": float(np.mean([e["value_grad_norm"] for e in epochs_log])),
        # clip fraction averaged over epochs (epoch 0 has ratio==1 by construction)
        "clip_fraction": float(np.mean([e["clip_fraction"] for e in epochs_log])),
        "affected_fraction": float(np.mean([e["affected_fraction"] for e in epochs_log])),
        "clip_fraction_last_epoch": float(epochs_log[-1]["clip_fraction"]),
        "post_update_clip_fraction": post_diag["clip_fraction"],
        "approx_kl_old_new": approx_kl_old_new,
        "ratio_abs_dev_post": post_diag["ratio_abs_dev_mean"],
        "explained_variance": explained_variance(ro["values"], returns, mask),
        "value_mean": float(masked_mean(ro["values"], mask)),
        "return_mean": float(masked_mean(returns, mask)),
        "raw_advantage_mean": float(masked_mean(raw_adv, mask)),
        "kl_penalty_sum_mean": float(((-(old_logp - ref_logp) * mask).sum(-1) * kl_beta).mean()),
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None,
            kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    n_updates, seed = int(cfg["updates"]), int(cfg["seed"])
    rows_all = bundle["prompt_rows"]
    schedule = prompt_schedule(len(rows_all), n_updates, int(cfg["prompts_per_update"]), seed)

    rdir = repo_path(cfg["results_dir"]) / run_name
    rdir.mkdir(parents=True, exist_ok=True)
    log_path = rdir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    run_config = {k: cfg[k] for k in ["updates", "prompts_per_update", "ppo_epochs", "policy_learning_rate",
                                      "value_lora_learning_rate", "value_head_learning_rate", "value_train_mode",
                                      "clip_epsilon", "kl_beta", "gamma", "gae_lambda", "value_coef",
                                      "missing_eos_penalty", "max_prompt_length", "max_response_length",
                                      "reward_max_length", "max_grad_norm", "seed", "generation", "lora"]}
    run_config.update({"run_name": run_name, "output": str(out), "start_policy": cfg["paths"]["ppo_midpoint_policy"],
                       "start_value": cfg["paths"]["ppo_midpoint_value"], "reward_model": cfg["reward_model"],
                       "micro_batch_size": int(cfg.get("micro_batch_size", 1)),
                       "whiten_advantages": bool(cfg.get("whiten_advantages", True)), "dropout": "disabled"})
    save_json(rdir / "run_config.json", run_config)
    print(f"[ppo:{run_name}] {n_updates} updates, eps={eps}, beta_kl={beta}")

    reset_peak_memory()
    t0 = time.perf_counter()
    gen_tokens = 0
    for u in range(n_updates):
        set_seed(seed + u)  # identical sampling stream across forks at a given update index
        rows = [rows_all[i] for i in schedule[u]]
        tu = time.perf_counter()
        ro = collect_rollout(bundle, rows, cfg)
        m = ppo_update(bundle, ro, cfg, eps, beta)
        mask = ro["mask"]
        lens = mask.sum(-1)
        gen_tokens += int(lens.sum().item())
        rec = {
            "update": u + 1,
            "prompt_source_index": [r.get("source_index") for r in rows],
            "prompt_id": [r.get("prompt_id") for r in rows],
            "rm_score": float(np.mean(ro["rm_score"])),
            "task_reward": float(ro["task_reward"].mean()),
            "kl_token_mean": float(masked_mean(ro["old_logp"] - ro["ref_logp"], mask)),
            "kl_seq_mean": float(((ro["old_logp"] - ro["ref_logp"]) * mask).sum(-1).mean()),
            "entropy": float(masked_mean(ro["entropy"], mask)),
            "sample_entropy": float(-masked_mean(ro["old_logp"], mask)),
            "response_length": float(lens.float().mean()),
            "truncated_frac": float(np.mean(ro["gen"]["truncated"])),
            "cumulative_generated_tokens": gen_tokens,
            **{k: v for k, v in m.items() if k != "epochs"},
            "per_epoch": m["epochs"],
            "update_seconds": time.perf_counter() - tu,
            "elapsed_s": time.perf_counter() - t0,
            "peak_vram_gib_so_far": peak_memory_gib(),
        }
        append_jsonl(log_path, rec)
        print(f"[ppo:{run_name}] u{u + 1:02d} RM {rec['rm_score']:+.3f} KL {rec['kl_token_mean']:.4f} "
              f"H {rec['entropy']:.3f} len {rec['response_length']:.0f} pl {rec['policy_loss']:+.4f} "
              f"vl {rec['value_loss']:.3f} clip {rec['clip_fraction']:.3f} gn {rec['policy_grad_norm']:.3f}")
        del ro
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wall = time.perf_counter() - t0
    peak = peak_memory_gib()
    bundle["policy"].save_pretrained(str(out))
    bundle["tokenizer"].save_pretrained(str(out))
    try:
        bundle["value_model"].save_pretrained(str(out) + "_value_adapter")
    except Exception as exc:  # head-only / full critic modes are not PEFT models
        print("value model not saved as adapter:", exc)
    summary = {**run_config, "wall_clock_s": wall, "peak_vram_gib": peak, "generated_tokens": gen_tokens,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    save_json(rdir / "train_summary.json", summary)
    print(f"[ppo:{run_name}] done in {wall / 60:.1f} min, peak VRAM {peak} GiB -> {out}")
    for k in list(bundle):
        bundle[k] = None
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
