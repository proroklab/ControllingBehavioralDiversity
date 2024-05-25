#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import List

import torch
from tensordict import TensorDictBase, TensorDict

from benchmarl.experiment.callback import Callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpirical
from het_control.snd import compute_behavioral_distance
from het_control.utils import overflowing_logits_norm


def get_het_model(policy):
    model = policy.module[0]
    while not isinstance(model, HetControlMlpEmpirical):
        model = model[0]
    return model


class SndCallback(Callback):
    """
    Callback used to compute SND during evaluations
    """

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        for group in self.experiment.group_map.keys():
            if not len(self.experiment.group_map[group]) > 1:
                # If agent group has 1 agent
                continue
            policy = self.experiment.group_policies[group]
            # Cat observations over time
            obs = torch.cat(
                [rollout.select((group, "observation")) for rollout in rollouts], dim=0
            )  # tensor of shape [*batch_size, n_agents, n_features]
            model = get_het_model(policy)
            agent_actions = []
            # Compute actions that each agent would take in this obs
            for i in range(model.n_agents):
                agent_actions.append(
                    model._forward(obs, agent_index=i, compute_estimate=False).get(
                        model.out_key
                    )
                )
            # Compute SND
            distance = compute_behavioral_distance(agent_actions, just_mean=True)
            self.experiment.logger.log(
                {f"eval/{group}/snd": distance.mean().item()},
                step=self.experiment.n_iters_performed,
            )


class NormLoggerCallback(Callback):
    """
    Callback to log some training metrics
    """

    def on_batch_collected(self, batch: TensorDictBase):
        for group in self.experiment.group_map.keys():
            keys_to_norm = [
                (group, "f"),
                (group, "g"),
                (group, "fdivg"),
                (group, "logits"),
                (group, "observation"),
                (group, "out_loc_norm"),
                (group, "estimated_snd"),
                (group, "scaling_ratio"),
            ]
            to_log = {}

            for key in keys_to_norm:
                value = batch.get(key, None)
                if value is not None:
                    to_log.update(
                        {"/".join(("collection",) + key): torch.mean(value).item()}
                    )
            self.experiment.logger.log(
                to_log,
                step=self.experiment.n_iters_performed,
            )


class TagCurriculum(Callback):
    """
    Tag curriculum used to freeze the green agents' policies during training
    """

    def __init__(self, simple_tag_freeze_policy_after_frames, simple_tag_freeze_policy):
        super().__init__()
        self.n_frames_train = simple_tag_freeze_policy_after_frames
        self.simple_tag_freeze_policy = simple_tag_freeze_policy
        self.activated = not simple_tag_freeze_policy

    def on_setup(self):
        # Log params
        self.experiment.logger.log_hparams(
            simple_tag_freeze_policy_after_frames=self.n_frames_train,
            simple_tag_freeze_policy=self.simple_tag_freeze_policy,
        )
        # Make agent group homogeneous
        policy = self.experiment.group_policies["agents"]
        model = get_het_model(policy)
        # Set the desired SND of the green agent team to 0
        # This is not important as the green agent team is composed of 1 agent
        model.desired_snd[:] = 0

    def on_batch_collected(self, batch: TensorDictBase):
        if (
            self.experiment.total_frames >= self.n_frames_train
            and not self.activated
            and self.simple_tag_freeze_policy
        ):
            del self.experiment.train_group_map["agents"]
            self.activated = True


class ActionSpaceLoss(Callback):
    """
    Loss to disincentivize actions outside of the space
    """

    def __init__(self, use_action_loss, action_loss_lr):
        super().__init__()
        self.opt_dict = {}
        self.use_action_loss = use_action_loss
        self.action_loss_lr = action_loss_lr

    def on_setup(self):
        # Log params
        self.experiment.logger.log_hparams(
            use_action_loss=self.use_action_loss, action_loss_lr=self.action_loss_lr
        )

    def on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        if not self.use_action_loss:
            return
        policy = self.experiment.group_policies[group]
        model = get_het_model(policy)
        if group not in self.opt_dict:
            self.opt_dict[group] = torch.optim.Adam(
                model.parameters(), lr=self.action_loss_lr
            )
        opt = self.opt_dict[group]
        loss = self.action_space_loss(group, model, batch)
        loss_td = TensorDict({"loss_action_space": loss}, [])

        loss.backward()

        grad_norm = self.experiment._grad_clip(opt)
        loss_td.set(
            f"grad_norm_action_space",
            torch.tensor(grad_norm, device=self.experiment.config.train_device),
        )

        opt.step()
        opt.zero_grad()

        return loss_td

    def action_space_loss(self, group, model, batch):
        logits = model._forward(
            batch.select(*model.in_keys), compute_estimate=True, update_estimate=False
        ).get(
            model.out_key
        )  # Compute logits from batch
        if model.probabilistic:
            logits, _ = torch.chunk(logits, 2, dim=-1)
        out_loc_norm = overflowing_logits_norm(
            logits, self.experiment.action_spec[group, "action"]
        )  # Compute how much they overflow outside the action space bounds

        # Penalise the maximum overflow over the agents
        max_overflowing_logits_norm = out_loc_norm.max(dim=-1)[0]

        loss = max_overflowing_logits_norm.pow(2).mean()
        return loss
