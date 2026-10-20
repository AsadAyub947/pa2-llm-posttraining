from __future__ import annotations

import torch

from common.metrics import masked_mean


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level GAE over response positions.

    rewards, values, mask: [batch, response_steps]. Padding positions must have mask=0.
    The final valid response position bootstraps with zero.
    """
    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(steps)):
        current_valid = mask[:, t]
        if t + 1 < steps:
            next_valid = mask[:, t + 1]
            next_value = values[:, t + 1] * next_valid
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(last_adv)

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * next_valid * last_adv
        last_adv = last_adv * current_valid
        advantages[:, t] = last_adv

    returns = advantages + values
    return advantages, returns


def shaped_rewards(task_reward, policy_logp, ref_logp, response_mask, beta_kl):
    """Sampled-action KL shaping plus terminal learned reward."""
    rewards = -float(beta_kl) * (policy_logp - ref_logp) * response_mask
    for b in range(rewards.shape[0]):
        valid = int(response_mask[b].sum().item())
        if valid > 0:
            rewards[b, valid - 1] += task_reward[b]
    return rewards


def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    """Return PPO clipped policy loss and diagnostics.

    L_clip = E_t[ min( rho_t A_t, clip(rho_t, 1-eps, 1+eps) A_t ) ]   (maximised)

    FIX (defect in the released starter): the starter took `torch.maximum` of the two
    surrogates. The max is an *optimistic* bound: for A>0 it keeps the unclipped term when
    rho > 1+eps (so the incentive to keep increasing rho never switches off) and for A<0 it
    keeps the unclipped term when rho < 1-eps. Clipping therefore never removed gradient
    and the trust region disappeared. PPO uses the pessimistic `min`.

    Returns (loss, ratio, clip_fraction). `clip_fraction` is the manual's definition: the
    fraction of valid tokens whose ratio lies outside [1-eps, 1+eps] *before* clipping.
    Use `ppo_clip_diagnostics` for the affected-token fraction (tokens whose gradient is
    actually zeroed by the clipped branch).
    """
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantage
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage

    objective = torch.minimum(surr1, surr2)

    loss = -masked_mean(objective, mask)
    outside = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    clip_fraction = masked_mean(outside, mask)
    return loss, ratio.detach(), clip_fraction.detach()


@torch.no_grad()
def ppo_clip_diagnostics(new_logp, old_logp, advantage, mask, eps=0.2):
    """Extra clipping geometry diagnostics computed with the same response mask.

    - clip_fraction: ratio outside [1-eps, 1+eps] (manual definition).
    - affected_fraction: tokens where the clipped branch is the active branch of the min, i.e.
      (A>0 and rho>1+eps) or (A<0 and rho<1-eps). Those tokens contribute zero gradient.
    - clipped_surrogate / unclipped_surrogate: masked means of the two objectives.
    """
    ratio = torch.exp(new_logp - old_logp)
    outside = (ratio < (1.0 - eps)) | (ratio > (1.0 + eps))
    affected = ((advantage > 0) & (ratio > 1.0 + eps)) | ((advantage < 0) & (ratio < 1.0 - eps))
    surr1 = ratio * advantage
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage
    m = mask.float()
    return {
        "clip_fraction": float(masked_mean(outside.float(), m)),
        "affected_fraction": float(masked_mean(affected.float(), m)),
        "clip_high_fraction": float(masked_mean((ratio > 1.0 + eps).float(), m)),
        "clip_low_fraction": float(masked_mean((ratio < 1.0 - eps).float(), m)),
        "unclipped_surrogate": float(masked_mean(surr1, m)),
        "clipped_surrogate": float(masked_mean(torch.minimum(surr1, surr2), m)),
        "ratio_mean": float(masked_mean(ratio, m)),
        "ratio_abs_dev_mean": float(masked_mean((ratio - 1.0).abs(), m)),
        "valid_tokens": int(m.sum().item()),
    }


def value_mse_loss(predicted_values, returns, mask):
    return masked_mean((predicted_values - returns) ** 2, mask)


def normalize_advantages(advantages, mask, eps=1e-6):
    valid = advantages[mask.bool()]
    if valid.numel() <= 1:
        return advantages
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    return ((advantages - mean) / std) * mask
