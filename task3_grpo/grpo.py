from __future__ import annotations

import torch

from common.metrics import masked_mean, sampled_kl, sample_entropy


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6):
    """Return one scalar advantage per sampled completion.

    `group_ids[i]` identifies which prompt produced reward `rewards[i]`.
    Validate this implementation against the group-relative definition in the assignment manual.
    """
    # FIX (defect in the released starter): the starter normalised with the *batch* mean/std,
    # pooling completions from different prompts. That makes the advantage of a completion
    # depend on how hard the *other* prompts in the batch were (a mediocre answer to an easy
    # prompt can look good next to a hard prompt), and gives non-zero advantages to groups whose
    # own rewards are all equal. GRPO's A_k = (r_k - mu_r) / (sigma_r + eps) uses the statistics
    # of the K completions for the same prompt only.
    rewards = rewards.float()
    group_ids = torch.as_tensor(group_ids, device=rewards.device)
    adv = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        idx = group_ids == g
        r = rewards[idx]
        mu = r.mean()
        sd = r.std(unbiased=False)
        adv[idx] = (r - mu) / (sd + eps)
    return adv


def group_reward_stats(rewards: torch.Tensor, group_ids: torch.Tensor, tol: float = 1e-6):
    """Per-group reward std and the uninformative (zero-std) flag, with the same tolerance."""
    rewards = rewards.float()
    group_ids = torch.as_tensor(group_ids, device=rewards.device)
    stds, uninformative = [], []
    for g in torch.unique(group_ids):
        r = rewards[group_ids == g]
        sd = float(r.std(unbiased=False).item()) if r.numel() > 1 else 0.0
        stds.append(sd)
        uninformative.append(sd <= tol)
    return stds, uninformative


def grpo_kl_k3(new_logp, ref_logp):
    """Per-token k3 estimator of KL(pi || ref): exp(ref - pi) - (ref - pi) - 1 (>= 0)."""
    log_ratio_ref_over_policy = ref_logp - new_logp
    return torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    `token_mask` may be all-zero for a completion that was deliberately masked because it hit the
    maximum generation length.
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask).sum(-1)
    if loss_type == "grpo":
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        # Constant normalization rather than dividing by each response's realized length.
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    per_token_kl = grpo_kl_k3(new_logp, ref_logp)
    kl = masked_mean(per_token_kl, token_mask)
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
