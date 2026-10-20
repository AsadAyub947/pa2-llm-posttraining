"""Shared evaluation / rollout helpers used by Tasks 1-3 (and the Task 6 summary).

Everything here is built on the released helpers in common.generation / common.metrics so
that KL, entropy and length are computed with one convention across all conditions:

* KL from reference: the released sampled-response estimator `common.metrics.sampled_kl`
  (token-level mean of log pi - log pi_ref over valid response tokens of *policy samples*).
  We additionally report the per-sequence sum (mean over sequences) under a separate key;
  both are reported for every condition so the averaging convention never changes.
* Entropy: exact token-level entropy of the full next-token distribution, averaged over
  valid response tokens (manual definition). The sampled-token estimate (-mean log p) is
  also logged as `sample_entropy` for reference.
* Response length: valid response tokens (up to and including EOS), excluding padding.
"""
from __future__ import annotations

import math
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel

from common.data import prompt_messages, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.metrics import masked_mean, sampled_kl
from common.models import reference_mode


# --------------------------------------------------------------------------------------
# resume helpers (Colab sessions disconnect; finished forks/evals are reused, --force redoes them)
# --------------------------------------------------------------------------------------

def training_done(adapter_dir, train_summary) -> bool:
    """A run is complete when its adapter and its end-of-run train_summary.json both exist."""
    return repo_path(train_summary).exists() and (repo_path(adapter_dir) / "adapter_config.json").exists()


def reusable_eval(eval_summary, train_summary=None, required=()):
    """Return a saved eval summary if it is newer than the training run and has `required` keys."""
    import json

    p = repo_path(eval_summary)
    if not p.exists():
        return None
    if train_summary is not None:
        t = repo_path(train_summary)
        if t.exists() and t.stat().st_mtime > p.stat().st_mtime:
            return None
    s = json.loads(p.read_text())
    if any(k not in s or s[k] is None for k in required):
        return None
    print(f"[resume] reusing {p}")
    return s


# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------

def disable_dropout(model) -> None:
    """Set every dropout probability to 0 (TRL does the same for PPO/GRPO).

    With LoRA dropout active, old (no-grad) and new (training) log-probs of the *same* weights
    differ, so the importance ratio would not start at 1 and the clip statistics would mostly
    measure dropout noise.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0


def length_stats(lengths) -> dict:
    x = np.asarray(list(lengths), dtype=float)
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"),
                "q25": float("nan"), "q75": float("nan"), "iqr": float("nan"), "max": float("nan"), "n": 0}
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return {
        "mean": float(x.mean()), "std": float(x.std(ddof=0)), "median": float(q50),
        "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25), "max": float(x.max()), "n": int(x.size),
    }


def mean_std(xs) -> dict:
    x = np.asarray(list(xs), dtype=float)
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "n": 0}
    return {"mean": float(x.mean()), "std": float(x.std(ddof=0)),
            "sem": float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan"), "n": int(x.size)}


def device_of(model):
    return next(model.parameters()).device


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated() / 2**30)


def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


# --------------------------------------------------------------------------------------
# log-probs / entropy for sampled responses
# --------------------------------------------------------------------------------------

def response_logprobs_entropy(model, sequences, attention_mask, prompt_width, response_ids, with_entropy=False):
    """Per-token log pi(a_t|s_t) for the generated response tokens (+ optional exact entropy).

    Same slicing as common.generation.response_token_logprobs, but avoids materialising a full
    log_softmax copy (logsumexp + gather instead) and returns the exact entropy when asked.
    Gradients flow through the log-probs if the caller is not under no_grad.
    """
    out = model(input_ids=sequences, attention_mask=attention_mask, use_cache=False, return_dict=True)
    T = response_ids.shape[1]
    logits = out.logits[:, prompt_width - 1: -1, :][:, :T, :].float()
    lse = torch.logsumexp(logits, dim=-1)
    tok = torch.gather(logits, -1, response_ids.unsqueeze(-1)).squeeze(-1) - lse
    ent = None
    if with_entropy:
        with torch.no_grad():
            lg = logits.detach()
            p = torch.softmax(lg, dim=-1)
            ent = lse.detach() - (p * lg).sum(-1)
            del p, lg
    del out, logits
    return tok, ent


@torch.no_grad()
def reference_logprobs(model, sequences, attention_mask, prompt_width, response_ids):
    """Frozen reference = base model with the LoRA adapter disabled."""
    if isinstance(model, PeftModel):
        with reference_mode(model):
            tok, _ = response_logprobs_entropy(model, sequences, attention_mask, prompt_width, response_ids)
        return tok
    # No adapter means the policy *is* the reference.
    tok, _ = response_logprobs_entropy(model, sequences, attention_mask, prompt_width, response_ids)
    return tok


# --------------------------------------------------------------------------------------
# PPO critic helpers
# --------------------------------------------------------------------------------------

def upcast_trainable_to_fp32(model) -> int:
    """Trainable fp16 parameters (e.g. the critic's `score` head copy) are unsafe with AdamW
    (eps=1e-8 underflows in fp16). Keep the frozen backbone in fp16, train in fp32."""
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n += 1
    return n


def critic_values(value_model, input_ids, attention_mask):
    """Per-position scalar values [B, L] from the critic's last hidden state.

    Equivalent to common.models.token_values, but (i) resolves the decoder backbone of a
    PEFT-wrapped classifier explicitly and (ii) casts hidden states to the head dtype so an
    fp32 trainable head works with an fp16 backbone.
    """
    base = value_model.get_base_model() if isinstance(value_model, PeftModel) else value_model
    backbone = getattr(base, base.base_model_prefix)
    out = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    hidden = out.last_hidden_state
    head = base.score if hasattr(base, "score") else base.classifier
    trainable = [p for p in head.parameters() if p.requires_grad]
    dtype = trainable[0].dtype if trainable else next(head.parameters()).dtype
    return head(hidden.to(dtype)).squeeze(-1).float()


def explained_variance(pred, target, mask) -> float:
    m = mask.bool()
    p, t = pred[m].float(), target[m].float()
    if t.numel() < 2:
        return float("nan")
    var_t = t.var(unbiased=False)
    if float(var_t) == 0.0:
        return float("nan")
    return float(1.0 - (t - p).var(unbiased=False) / var_t)


# --------------------------------------------------------------------------------------
# reward model
# --------------------------------------------------------------------------------------

@torch.no_grad()
def score_rewards(rm_model, rm_tokenizer, prompts, responses, max_length=1280, batch_size=4) -> list[float]:
    out = []
    for i in range(0, len(prompts), batch_size):
        r = score_reward_pairs(rm_model, rm_tokenizer, prompts[i:i + batch_size], responses[i:i + batch_size], max_length=max_length)
        out.extend(float(x) for x in r.cpu().tolist())
    return out


# --------------------------------------------------------------------------------------
# held-out generation evaluation (common protocol for Tasks 1-3)
# --------------------------------------------------------------------------------------

def evaluate_generations(
    policy,
    tokenizer,
    rows: list[dict],
    *,
    reward=None,
    max_prompt_length: int = 256,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    seed: int = 6304,
    reward_max_length: int = 1280,
    id_key_candidates=("prompt_id", "source_index", "id"),
    compute_kl: bool = True,
):
    """Generate one response per prompt and compute reward / KL / entropy / length.

    Returns (summary_dict, per_example_records). Decoding settings, seed and prompt order are
    fixed by the caller so every condition in a comparison sees the same protocol.
    """
    from common.logging_utils import set_seed

    set_seed(seed)
    records = []
    tok_kl_num, tok_kl_den = 0.0, 0.0
    ent_num, ent_den = 0.0, 0.0
    t0 = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prompts = [prompt_messages(r) for r in chunk]
        gen = batch_generate(policy, tokenizer, prompts, max_prompt_length=max_prompt_length,
                             max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=do_sample)
        mask = gen["response_mask"]
        seq_kl = [float("nan")] * len(chunk)
        seq_ent = [float("nan")] * len(chunk)
        if compute_kl:
            with torch.no_grad():
                pol, ent = response_logprobs_entropy(policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"],
                                                     gen["response_ids"], with_entropy=True)
                ref = reference_logprobs(policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
            diff = (pol - ref) * mask
            tok_kl_num += float(diff.sum().item())
            tok_kl_den += float(mask.sum().item())
            ent_num += float((ent * mask).sum().item())
            ent_den += float(mask.sum().item())
            seq_kl = diff.sum(-1).cpu().tolist()
            seq_ent = ((ent * mask).sum(-1) / mask.sum(-1).clamp_min(1)).cpu().tolist()
            del pol, ent, ref, diff
        rewards = [float("nan")] * len(chunk)
        if reward is not None:
            rm_model, rm_tok = reward
            rewards = score_rewards(rm_model, rm_tok, prompts, gen["responses"], max_length=reward_max_length)
        for j, row in enumerate(chunk):
            rid = next((row[k] for k in id_key_candidates if k in row), start + j)
            records.append({
                "id": rid,
                "row_index": start + j,
                "prompt": prompts[j][-1]["content"] if prompts[j] else "",
                "response": gen["responses"][j],
                "response_tokens": int(gen["response_lengths"][j]),
                "terminated_with_eos": bool(gen["terminated_with_eos"][j]),
                "truncated": bool(gen["truncated"][j]),
                "reward": rewards[j],
                "seq_kl": float(seq_kl[j]),
                "token_entropy": float(seq_ent[j]),
            })
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lens = [r["response_tokens"] for r in records]
    summary = {
        "n_prompts": len(records),
        "reward": mean_std([r["reward"] for r in records]) if reward is not None else None,
        # Released convention: token-level mean of (log pi - log ref) over all valid tokens.
        "kl_token_mean": (tok_kl_num / max(tok_kl_den, 1.0)) if compute_kl else None,
        # Same estimator, summed per sequence then averaged over sequences.
        "kl_seq_mean": float(np.mean([r["seq_kl"] for r in records])) if compute_kl else None,
        "entropy_token_mean": (ent_num / max(ent_den, 1.0)) if compute_kl else None,
        "length": length_stats(lens),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])) if records else float("nan"),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in records])) if records else float("nan"),
        "decoding": {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p,
                     "do_sample": do_sample, "seed": seed, "max_prompt_length": max_prompt_length},
        "eval_seconds": time.perf_counter() - t0,
    }
    return summary, records


def select_eval_rows(rows: list[dict], n: int | None) -> list[dict]:
    """Fixed, order-preserving prefix of the released eval file (no shuffling)."""
    if n is None or int(n) <= 0 or int(n) >= len(rows):
        return list(rows)
    return list(rows[: int(n)])
