#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with CTS (Concurrent Teacher-Student) using RSL-RL.

This script demonstrates how to use the CTS framework for training legged locomotion policies.

Example usage:
    python train_cts.py --task Isaac-Velocity-Rough-Anymal-C-v0 --num_envs 8192 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CTS (Concurrent Teacher-Student).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=8192, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=5000, help="RL Policy training iterations.")
parser.add_argument(
    "--teacher_student_ratio", type=float, default=3.0, help="Ratio of teacher to student environments (default: 3:1)"
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import CTSRunner, RslRlCtsRunnerCfg, RslRlEncoderCfg, RslRlCtsAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train with CTS (Concurrent Teacher-Student) agent."""

    # Parse task name
    if args_cli.task is None:
        raise ValueError("Please specify a task using --task argument")

    # Create environment configuration
    # Note: You need to ensure your environment provides both privileged and proprioceptive observations
    print(f"[INFO] Creating environment: {args_cli.task}")

    # Create environment
    env = gym.make(args_cli.task, render_mode="rgb_array" if args_cli.video else None)

    # Wrap environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Create CTS training configuration
    cts_cfg = RslRlCtsRunnerCfg(
        seed=args_cli.seed,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        num_steps_per_env=24,
        max_iterations=args_cli.max_iterations,

        # Privileged encoder (teacher)
        privileged_encoder=RslRlEncoderCfg(
            latent_dim=24,
            hidden_dims=[512, 256],
            activation="elu",
            obs_normalization=True,
            use_history=False,
        ),

        # Proprioceptive encoder (student)
        proprioceptive_encoder=RslRlEncoderCfg(
            latent_dim=24,
            hidden_dims=[512, 256],
            activation="elu",
            obs_normalization=True,
            use_history=True,
            history_length=5,
        ),

        # Actor configuration (shared policy)
        actor=RslRlMLPModelCfg(
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,  # Already normalized in encoder
        ),

        # Critic configuration
        critic=RslRlMLPModelCfg(
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
        ),

        # CTS algorithm configuration
        algorithm=RslRlCtsAlgorithmCfg(
            # PPO parameters
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            clip_param=0.2,
            entropy_coef=0.01,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,

            # CTS-specific parameters
            teacher_batch_size=24576,  # 8192 × 3
            student_batch_size=12288,  # 2048 × 6
            reconstruction_loss_coef=1.0,
            student_learning_rate=1e-3,
            student_num_learning_epochs=5,
        ),

        # Environment split
        teacher_student_ratio=args_cli.teacher_student_ratio,
        teacher_steps_per_env=24,
        student_steps_per_env=24,

        # Observation groups
        obs_groups={
            "privileged": ["policy", "privileged"],  # Teacher sees both
            "proprioceptive": ["policy"],  # Student sees only policy observations
        },

        # Logging
        experiment_name=f"cts_{args_cli.task}",
        run_name="",
        save_interval=100,
        logger="tensorboard",
    )

    # Create log directory
    log_root_path = os.path.join("logs", "cts", cts_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cts_cfg.run_name:
        log_dir += f"_{cts_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Create CTS runner
    runner = CTSRunner(
        env=env,
        train_cfg=cts_cfg.to_dict() if hasattr(cts_cfg, 'to_dict') else vars(cts_cfg),
        log_dir=log_dir,
        device=cts_cfg.device,
    )

    # Dump configuration
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "cts_config.yaml"), cts_cfg)

    # Start training
    print(f"[INFO] Starting CTS training for {cts_cfg.max_iterations} iterations")
    print(f"[INFO] Teacher/Student ratio: {args_cli.teacher_student_ratio}:1")
    print(f"[INFO] Total environments: {args_cli.num_envs}")

    runner.learn(num_learning_iterations=cts_cfg.max_iterations, init_at_random_ep_len=True)

    # Close environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
