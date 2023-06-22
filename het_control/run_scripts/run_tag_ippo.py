#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import hydra
from omegaconf import DictConfig

from het_control.run import get_experiment


@hydra.main(version_base=None, config_path="../conf", config_name="tag_ippo_config")
def hydra_experiment(cfg: DictConfig) -> None:
    experiment = get_experiment(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
