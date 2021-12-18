# Meta-Residual Policy Learning
This repo contains code accompaning the paper, **Meta-Residual Policy Learning: Zero-trial Robot Skill Adaptation via Knowledge Fusion** (IEEE RA-L submission). It includes code for running the robotic peg-in-hole assembly tasks.
This repository is based on [PEARL](https://github.com/katerakelly/oyster).

#### Dependencies
We recommend using conda create environment with 

`conda env create -f mrplenv.yaml`

This installation has been tested only on 64-bit Ubuntu 16.04.

#### Usage
To reproduce an experiment, run:

`python launch_experiment.py ./configs/pih-meta.json`

Output files will be written to `./output/pih-meta/[EXP NAME]` where the experiment name is uniquely generated based on the date.
The file `progress.csv` contains statistics logged over the course of training.
To visualize learning curves, run: 

`python viskit/viskit/frontend.py output/pih-meta/`

For evaluating the learned model, run 

`python sim_policy.py ./configs/pih-meta.json ./output/pih-meta/[EXP NAME] --num_trajs=20`

To visualize the evaluation results, modify variable `expdir=output/pih-meta/[EXP NAME]/eval_trajectories/` in `plot_fig.py`, and run

`python plot_fig.py`

--------------------------------------
#### Contact

To ask questions or report bugs, please open an issue.
