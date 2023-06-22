#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

from typing import List

import torch


def compute_behavioral_distance(
    agent_actions: List[torch.Tensor],
    just_mean: bool,
):
    """Takes as input actions and computes the distance between agent pairs.

    Args:
        agent_actions (list of Tensor): each has shape [*batch, action_features]
        just_mean (bool): if the actions are mean and std_dev or just mean
    Returns:
         [*batch, n_agent_pairs]

    """
    n_agents = len(agent_actions)

    pair_results = []
    for agent_i in range(n_agents):
        for agent_j in range(n_agents):
            if agent_j <= agent_i:
                continue
            out_i = agent_actions[agent_i]
            out_j = agent_actions[agent_j]

            pair_results.append(
                compute_statistical_distance(out_i, out_j, just_mean=just_mean)
            )
    result = torch.stack(pair_results, dim=-1)
    n_pairs = (n_agents * (n_agents - 1)) // 2
    return result.view((*agent_actions[0].shape[:-1], n_pairs))


def compute_statistical_distance(logits_i, logits_j, just_mean: bool):
    if just_mean:
        loc_i = logits_i
        loc_j = logits_j
        var_i = var_j = None
    else:
        loc_i, scale_i = logits_i.chunk(2, -1)
        var_i = scale_i.pow(2)  # Variance

        loc_j, scale_j = logits_j.chunk(2, -1)
        var_j = scale_j.pow(2)  # Variance

    out = wasserstein_distance(
        mean=loc_i,
        mean2=loc_j,
        sigma=torch.diag_embed(var_i) if not just_mean else None,
        sigma2=torch.diag_embed(var_j) if not just_mean else None,
        just_mean=just_mean,
    )

    return out.view(logits_i.shape[:-1])


def wasserstein_distance(mean, sigma, mean2, sigma2, just_mean: bool = False):
    # mean is a tensor of shape [*B, S]
    # sigma is a tensor of shape [*B, S, S]
    # output has shape [*B]

    mean_component = torch.linalg.norm(mean - mean2, ord=2, dim=-1)
    if just_mean:
        return mean_component

    # check that covariances are positive definite
    # by computing matrix square roots
    SC = torch.linalg.cholesky(sigma)
    SC2 = torch.linalg.cholesky(sigma2)
    return torch.sqrt(
        mean_component**2 + torch.linalg.norm(SC - SC2, ord="fro", dim=(-1, -2)) ** 2
    )
