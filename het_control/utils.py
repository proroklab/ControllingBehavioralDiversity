#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch


def clamp_preserve_gradients(x: torch.Tensor, min, max) -> torch.Tensor:
    """
    This helper function clamps gradients but still passes through the
    gradient in clamped regions
    """
    return x + (x.clamp(min, max) - x).detach()


def overflowing_logits_norm(logits, action_spec):
    """Compute the l2 norm of actions overflowing the space bounds"""
    action_max = action_spec.space.high
    action_min = action_spec.space.low

    logits_clamped = torch.clamp(logits, min=action_min, max=action_max).detach()
    overflowing_logits = logits - logits_clamped
    overflowing_logits_norm = torch.linalg.vector_norm(overflowing_logits, dim=-1)

    return overflowing_logits_norm
