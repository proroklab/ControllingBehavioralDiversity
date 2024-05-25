#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Type, Sequence, Optional

import torch
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor
from torch import nn
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
from het_control.snd import compute_behavioral_distance
from het_control.utils import overflowing_logits_norm
from .utils import squash


class HetControlMlpEmpirical(Model):
    def __init__(
        self,
        activation_class: Type[nn.Module],
        num_cells: Sequence[int],
        desired_snd: float,
        probabilistic: bool,
        scale_mapping: Optional[str],
        tau: float,
        bootstrap_from_desired_snd: bool,
        process_shared: bool,
        **kwargs,
    ):
        """DiCo policy model

        Args:
            activation_class (Type[nn.Module]): activation class to be used.
            num_cells (int or Sequence[int], optional): number of cells of every layer in between the input and output. If
                an integer is provided, every layer will have the same number of cells. If an iterable is provided,
                the linear layers out_features will match the content of num_cells.
            desired_snd (float): The desired SND diversity
            probabilistic (bool):  Whether the model has stochastic actions or not.
            scale_mapping (str, optional): Type of mapping to use to make the std_dev output of the policy positive
                (choices: "softplus", "exp", "relu", "biased_softplus_1")
            tau (float): The soft-update parameter of the estimated diversity.  Must be between 0 and 1
            bootstrap_from_desired_snd (bool):  Whether on the first iteration the estimated SND should be bootstrapped
                from the desired snd (True) or from the measured SND (False)
            process_shared (bool): Whether to process the homogeneous part of the policy with a tanh squashing operation to the action space domain
        """

        super().__init__(**kwargs)

        self.num_cells = num_cells
        self.activation_class = activation_class
        self.probabilistic = probabilistic
        self.scale_mapping = scale_mapping
        self.tau = tau
        self.bootstrap_from_desired_snd = bootstrap_from_desired_snd
        self.process_shared = process_shared

        self.register_buffer(
            name="desired_snd",
            tensor=torch.tensor([desired_snd], device=self.device, dtype=torch.float),
        )  # Buffer for SND_{des}
        self.register_buffer(
            name="estimated_snd",
            tensor=torch.tensor([float("nan")], device=self.device, dtype=torch.float),
        )  # Buffer for \widehat{SND}

        self.scale_extractor = (
            NormalParamExtractor(scale_mapping=scale_mapping)
            if scale_mapping is not None
            else None
        )  # Components that maps std_dev according to scale_mapping

        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]

        self.shared_mlp = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=False,
            share_params=True,  # Parameter-shared
            device=self.device,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
        )  # Shared network that outputs mean and std_dev in stochastic policies and just mean in deterministic policies

        agent_outputs = (
            self.output_features // 2 if self.probabilistic else self.output_features
        )
        self.agent_mlps = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=agent_outputs,
            n_agents=self.n_agents,
            centralised=False,
            share_params=False,  # Not parameter-shared
            device=self.device,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
        )  # Per-agent networks that output mean deviations from the shared policy

    def _perform_checks(self):
        super()._perform_checks()

        if self.centralised or not self.input_has_agent_dim:
            raise ValueError(f"{self.__class__.__name__} can only be used for policies")

        # Run some checks
        if self.input_has_agent_dim and self.input_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError(
                "If the MLP input has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the MLP output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )

    def _forward(
        self,
        tensordict: TensorDictBase,
        agent_index: int = None,
        update_estimate: bool = True,
        compute_estimate: bool = True,
    ) -> TensorDictBase:
        # Gather in_key

        input = tensordict.get(
            self.in_key
        )  # Observation tensor of shape [*batch, n_agents, n_features]
        shared_out = self.shared_mlp.forward(input)
        if agent_index is None:  # Gather outputs for all agents on the obs
            # tensor of shape [*batch, n_agents, n_actions], where the outputs
            # along the n_agent dimension are taken with the different agent networks
            agent_out = self.agent_mlps.forward(input)
        else:  # Gather outputs for one agent on the obs
            # tensor of shape [*batch, n_agents, n_actions], where the outputs
            # along the n_agent dimension are taken with the same (agent_index) agent network
            agent_out = self.agent_mlps.agent_networks[agent_index].forward(input)

        shared_out = self.process_shared_out(shared_out)

        if (
            self.desired_snd > 0
            and torch.is_grad_enabled()  # we are training
            and compute_estimate
            and self.n_agents > 1
        ):
            # Update \widehat{SND}
            distance = self.estimate_snd(input)
            if update_estimate:
                self.estimated_snd[:] = distance.detach()
        else:
            distance = self.estimated_snd
        if self.desired_snd == 0:
            scaling_ratio = 0.0
        elif (
            self.desired_snd == -1  # Unconstrained networks
            or distance.isnan().any()  # It is the first iteration
            or self.n_agents == 1
        ):
            scaling_ratio = 1.0
        else:  # DiCo scaling
            scaling_ratio = torch.where(
                distance != self.desired_snd,
                self.desired_snd / distance,
                1,
            )

        if self.probabilistic:
            shared_loc, shared_scale = shared_out.chunk(2, -1)

            # DiCo scaling
            agent_loc = shared_loc + agent_out * scaling_ratio
            out_loc_norm = overflowing_logits_norm(
                agent_loc, self.action_spec[self.agent_group, "action"]
            )  # For logging
            agent_scale = shared_scale

            out = torch.cat([agent_loc, agent_scale], dim=-1)
        else:
            # DiCo scaling
            out = shared_out + scaling_ratio * agent_out
            out_loc_norm = overflowing_logits_norm(
                out, self.action_spec[self.agent_group, "action"]
            )  # For logging

        tensordict.set(
            (self.agent_group, "estimated_snd"),
            self.estimated_snd.expand(tensordict.get_item_shape(self.agent_group)),
        )
        tensordict.set(
            (self.agent_group, "scaling_ratio"),
            (
                torch.tensor(scaling_ratio, device=self.device).expand_as(out)
                if not isinstance(scaling_ratio, torch.Tensor)
                else scaling_ratio.expand_as(out)
            ),
        )
        tensordict.set((self.agent_group, "logits"), out)
        tensordict.set((self.agent_group, "out_loc_norm"), out_loc_norm)

        tensordict.set(self.out_key, out)

        return tensordict

    def process_shared_out(self, logits: torch.Tensor):
        if not self.probabilistic and self.process_shared:
            return squash(
                logits,
                action_spec=self.action_spec[self.agent_group, "action"],
                clamp=False,
            )
        elif self.probabilistic:
            loc, scale = self.scale_extractor(logits)
            if self.process_shared:
                loc = squash(
                    loc,
                    action_spec=self.action_spec[self.agent_group, "action"],
                    clamp=False,
                )
            return torch.cat([loc, scale], dim=-1)
        else:
            return logits

    # @torch.no_grad()
    def estimate_snd(self, obs: torch.Tensor):
        """
        Update \widehat{SND}
        """
        agent_actions = []
        # Gather what actions each agent would take if given the obs tensor
        for agent_net in self.agent_mlps.agent_networks:
            agent_outputs = agent_net(obs)
            agent_actions.append(agent_outputs)

        distance = (
            compute_behavioral_distance(agent_actions=agent_actions, just_mean=True)
            .mean()
            .unsqueeze(-1)
        )  # Compute the SND if these unscaled policies
        if self.estimated_snd.isnan().any():  # First iteration
            distance = self.desired_snd if self.bootstrap_from_desired_snd else distance
        else:
            # Soft update of \widehat{SND}
            distance = (1 - self.tau) * self.estimated_snd + self.tau * distance

        return distance


@dataclass
class HetControlMlpEmpiricalConfig(ModelConfig):
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING

    desired_snd: float = MISSING
    tau: float = MISSING
    bootstrap_from_desired_snd: bool = MISSING
    process_shared: bool = MISSING

    probabilistic: Optional[bool] = MISSING
    scale_mapping: Optional[str] = MISSING

    @staticmethod
    def associated_class():
        return HetControlMlpEmpirical
