#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import benchmarl.models
from benchmarl.algorithms import *
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.hydra_config import (
    load_algorithm_config_from_hydra,
    load_experiment_config_from_hydra,
    load_task_config_from_hydra,
    load_model_config_from_hydra,
)
from het_control.callback import *
from het_control.environments.vmas import render_callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpiricalConfig


def setup(task_name):
    benchmarl.models.model_config_registry.update(
        {
            "hetcontrolmlpempirical": HetControlMlpEmpiricalConfig,
        }
    )
    if task_name == "vmas/navigation":
        # Set the render callback for the navigatio case study
        VmasTask.render_callback = render_callback


def get_experiment(cfg: DictConfig) -> Experiment:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    setup(task_name)

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    task_config = load_task_config_from_hydra(cfg.task, task_name)
    critic_model_config = load_model_config_from_hydra(cfg.critic_model)
    model_config = load_model_config_from_hydra(cfg.model)

    if isinstance(algorithm_config, (MappoConfig, IppoConfig, MasacConfig, IsacConfig)):
        model_config.probabilistic = True
        model_config.scale_mapping = algorithm_config.scale_mapping
        algorithm_config.scale_mapping = (
            "relu"  # The scaling of std_dev will be done in the model
        )
    else:
        model_config.probabilistic = False

    experiment = Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=cfg.seed,
        config=experiment_config,
        callbacks=[
            SndCallback(),
            NormLoggerCallback(),
            ActionSpaceLoss(
                use_action_loss=cfg.use_action_loss, action_loss_lr=cfg.action_loss_lr
            ),
        ]
        + (
            [
                TagCurriculum(
                    cfg.simple_tag_freeze_policy_after_frames,
                    cfg.simple_tag_freeze_policy,
                )
            ]
            if task_name == "vmas/simple_tag"
            else []
        ),
    )
    return experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    experiment = get_experiment(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
