# Controlling Behavioral Diversity

<img src="https://github.com/matteobettini/vmas-media/blob/main/dico/dico.png?raw=true" alt="drawing"/> 

This is the code accompanying the paper: "Controlling Behavioral Diversity in Multi-Agent Reinforcement Learning".

## Video

Watch the presentation video of DiCo.

<p align="center">

[![DiCo Video](https://img.youtube.com/vi/ImcuXnmX43g/0.jpg)](https://www.youtube.com/watch?v=ImcuXnmX43g)
</p>

## Installing

1. Create a virtual environment (e.g., conda) with python 3.9
2. Install our versions of `VMAS`, `TensorDict`, `TorchRL`, and `BenchMARL`.

```bash
git clone -b het_control https://github.com/proroklab/VectorizedMultiAgentSimulator.git
pip install -e VectorizedMultiAgentSimulator

git clone -b het_control https://github.com/matteobettini/tensordict.git
cd tensordict
python setup.py develop
cd ..

git clone -b het_control https://github.com/matteobettini/rl.git
cd rl
python setup.py develop
cd ..

git clone -b het_control https://github.com/matteobettini/BenchMARL.git
pip install -e BenchMARL
```
3. Install optional dependencies for logging
```bash
pip installl wandb moviepy
```
4. Install this project via
```bash
git clone https://github.com/proroklab/ControllingBehavioralDiversity.git
pip install -e ControllingBehavioralDiversity
```
5. Try running a script (it will ask for cuda and wandb, you can change these values in `ControllingBehavioralDiversity/het_control/conf/experiment/het_control_experiment.yaml`)
```
python ControllingBehavioralDiversity/het_control/run_scripts/run_navigation_ippo.py model.desired_snd=0.1
```

## Running

The [`het_control/run_scripts`](het_control/run_scripts) folder contains the files to run the various experiments
included in the paper.

To run an experiment just do
```bash
python run_navigation_ippo.py model.desired_snd=0.3
```

You can run the same experiment over multiple config values, such as desired diversity values and seeds
```bash
python run_navigation_ippo.py -m model.desired_snd=-1,0,0.3 seed=0,1,2
```

`model.desired_snd=-1` instructs to run unconstrained heterogeneous policies

For more information on the running syntax and options, 
check out the [BenchMARL guide](https://benchmarl.readthedocs.io/en/latest/usage/running.html).

## Configuring

The configuration for the codebase is available in the [`het_control/conf`](het_control/conf) folder.

This configuration overrides the default BenchMARL configuration, contained in  [`benchmarl/conf`](https://github.com/facebookresearch/BenchMARL/tree/main/benchmarl/conf).
So, if a value is not present in the configuration of this project, it will take the default value from that folder.

At the top level of the [`het_control/conf`](het_control/conf) folder, you can find a config file for each experiment.
For example: [`navigation_ippo_config.yaml`](het_control/conf/navigation_ippo_config.yaml).

These files define:
- an algorithm
- a policy model (Which, in this repository, is always our proposed method)
- a critic model
- a task
- an experiment hyperparameter configuration

The values of these fields determine which configuration to load for each of these components. The configurations are loaded from the sub-folders:
[`het_control/conf/algorithm`](het_control/conf/algorithm), [`het_control/conf/model`](het_control/conf/model), 
 [`het_control/conf/task`](het_control/conf/task), [`het_control/conf/experiment`](het_control/conf/experiment).
The top level experiment file sometimes overrides certain values of these components.

All the configurations attributes in these sub-folders are documented to explain their meaning.
Docs on these values are also available in [BenchMARL](https://benchmarl.readthedocs.io/en/latest/concepts/configuring.html).

You can override any of these values from the command-line using the basic [Hydra](https://github.com/facebookresearch/hydra) syntax.
For example:
```bash
python run_navigation_ippo.py model.desired_snd=0.3 seed=1 experiment.max_n_frames=1_000_000 algorithm.lmbda=0.8
```

## Citation

```BibTeX
@inproceedings{bettini2024controlling,
    title={Controlling Behavioral Diversity in Multi-Agent Reinforcement Learning},
    author={Bettini, Matteo and Kortvelesy, Ryan and Prorok, Amanda},
    booktitle={Forty-first International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/forum?id=qQjUgItPq4}
}
```
