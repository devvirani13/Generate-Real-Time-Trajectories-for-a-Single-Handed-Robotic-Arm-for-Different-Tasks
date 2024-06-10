# Generate Real-Time Trajectories for a Single-Handed Robotic Arm for Different Tasks


- [Overview](#overview)
- [Project Website](#project-website)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Example Usages](#example-usages)
  - [Simulated Experiments](#simulated-experiments)
  - [Training ACT](#to-train-act)
  - [Evaluating the Policy](#to-evaluate-the-policy)
  - [Real-World Experiments](#real-world-experiments)
- [Tuning Tips](#tuning-tips)
- [Acknowledgements](#acknowledgements)

## Overview

This project implements real-time trajectory generation for a single-handed Kinova robotic arm using Action Chunking with Transformers (ACT). Originally developed for dual-arm setups with four cameras, this adaptation is tailored for a single Kinova arm equipped with three cameras. The project includes both simulated and real-world environments for training and evaluation.


 

## Repository Structure
- `imitate_episodes.py` - Train and evaluate ACT
- `policy.py` - Adaptor for ACT policy
- `detr` - Model definitions of ACT, modified from DETR
- `sim_env.py` - Mujoco + DM_Control environments with joint space control
- `ee_sim_env.py` - Mujoco + DM_Control environments with EE space control
- `scripted_policy.py` - Scripted policies for simulated environments
- `constants.py` - Constants shared across files
- `utils.py` - Data loading and helper functions
- `visualize_episodes.py` - Save videos from a .hdf5 dataset

## Installation
- Follow these steps to set up the environment and dependencies:

  ```sh
    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .


## Example Usages

- To set up a new terminal, run:
  ```sh
  conda activate aloha
  cd <path to act repo>
  
## Simulated Experiments
- We use the 'sim_transfer_cube_scripted' task in the examples below. Another option is 'sim_insertion_scripted'. To generate 50 episodes of scripted data, run:
    ```sh
    python3 record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

- You can add the flag --onscreen_render to see real-time rendering. To visualize the episode after it is collected, run:
    ```sh
    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

## To Train ACT

- Transfer Cube Task
    ```sh
    python3 imitate_episodes.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir <ckpt dir> \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000 --lr 1e-5 \
--seed 0

## To Evaluate the Policy:

- Run the same command but add '--eval'. This loads the best validation checkpoint. The success rate should be around 90% for transfer cube, and around 50% for insertion. To enable temporal ensembling, add flag '--temporal_agg'. Videos will be saved to '<ckpt_dir>' for each rollout. You can also add '--onscreen_render' to see real-time rendering during evaluation.

## Real-World Experiments
- For real-world data where modeling can be more challenging, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued. Please refer to tuning tips for more information.

## Tuning Tips

- If your ACT policy is jerky or pauses in the middle of an episode, train for longer. Success rate and smoothness can improve significantly even after the loss plateaus.

## Acknowledgements

- This project builds upon the work originally developed for dual-arm setups and adapts it for single-handed Kinova robotic arm implementations. Special thanks to the authors of the original ACT paper and the ALOHA project.
    ### Project Website
    - Visit the project website [here](https://tonyzhaozh.github.io/aloha/).

