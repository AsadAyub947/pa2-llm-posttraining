from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    L = -log sigma( beta * ( [log pi(y+) - log ref(y+)] - [log pi(y-) - log ref(y-)] ) )
      = -log sigma( beta * ( policy_margin - ref_margin ) )

    FIX (defect in the released starter): the starter used `policy_margin + ref_margin`,
    which *adds* the reference log-ratio instead of subtracting it. That rewards the policy
    for whatever the reference already prefers, makes the loss non-zero-centred at
    initialisation (pi == ref should give logits == 0, loss == log 2), and is not the
    objective derived in Rafailov et al. (2023). The implicit-reward margin m_theta in the
    manual is exactly `policy_margin - ref_margin`.
    """

    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp

    margin = policy_margin - ref_margin  # m_theta in the manual
    logits = beta * margin

    loss = -F.logsigmoid(logits).mean()

    chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)

    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "dpo_margin_mean": margin.detach().mean(),
        "chosen_reward_mean": chosen_reward.detach().mean(),
        "rejected_reward_mean": rejected_reward.detach().mean(),
        "preference_accuracy": (margin > 0).float().mean().detach(),
    }
