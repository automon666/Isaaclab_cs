#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train EVA02 locomotion with CTS (Concurrent Teacher-Student)."""

import argparse

from isaacsim import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser(description="Train EVA02 locomotion with CTS.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-EVA02-CTS-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = SimulationApp({"headless": not args_cli.video})

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from isaaclab_rl.rsl_rl.runners import CTSRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper


def main():
    """Train with CTS (Concurrent Teacher-Student) agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.video
    )
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print(f"[INFO] Video directory: {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create CTS runner from rsl-rl
    runner = CTSRunner(env, agent_cfg, log_dir="logs/rsl_rl", device=env.unwrapped.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # set seed of the environment
    env.seed(args_cli.seed)

    print("=" * 80)
    print("CTS Training Configuration:")
    print(f"  Task: {args_cli.task}")
    print(f"  Number of environments: {env.num_envs}")
    print(f"  Privileged encoder latent dim: {agent_cfg.privileged_encoder.latent_dim}")
    print(f"  Proprioceptive encoder latent dim: {agent_cfg.proprioceptive_encoder.latent_dim}")
    print(f"  Teacher batch size: {agent_cfg.algorithm.teacher_batch_size}")
    print(f"  Student batch size: {agent_cfg.algorithm.student_batch_size}")
    print(f"  Reconstruction loss coef: {agent_cfg.algorithm.reconstruction_loss_coef}")
    print("=" * 80)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    app_launcher.close()
