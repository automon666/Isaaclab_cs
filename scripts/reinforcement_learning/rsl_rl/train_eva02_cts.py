#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train EVA02 locomotion with CTS (Concurrent Teacher-Student)."""

import argparse
from datetime import datetime

from isaacsim import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser(description="Train EVA02 locomotion with CTS.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-EVA02-CTS-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=5000, help="RL Policy training iterations.")
parser.add_argument(
    "--teacher_student_ratio", type=float, default=3.0, help="Ratio of teacher to student environments (default: 3:1)"
)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = SimulationApp({"headless": not args_cli.video})

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from isaaclab_rl.rsl_rl import (
    CTSRunner,
    RslRlCtsAlgorithmCfg,
    RslRlCtsRunnerCfg,
    RslRlEncoderCfg,
    RslRlMLPModelCfg,
    RslRlVecEnvWrapper,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Train with CTS (Concurrent Teacher-Student) agent."""
    # parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.video
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "cts", f"cts_{args_cli.task}", "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Build CTS training configuration (following train_cts.py pattern)
    agent_cfg = RslRlCtsRunnerCfg(
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
            obs_normalization=False,
        ),

        # Critic configuration
        critic=RslRlMLPModelCfg(
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
        ),

        # CTS algorithm configuration
        algorithm=RslRlCtsAlgorithmCfg(
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
            teacher_batch_size=12288,
            student_batch_size=6144,
            reconstruction_loss_coef=1.0,
            student_learning_rate=1e-3,
            student_num_learning_epochs=5,
        ),

        # Environment split
        teacher_student_ratio=args_cli.teacher_student_ratio,
        teacher_steps_per_env=24,
        student_steps_per_env=24,

        # Observation groups (EVA02 uses policy obs for both)
        obs_groups={
            "privileged": ["policy"],
            "proprioceptive": ["policy"],
        },

        # Logging
        experiment_name=f"cts_{args_cli.task}",
        run_name="",
        save_interval=100,
        logger="tensorboard",
    )

    # Create log directory
    log_root_path = os.path.join("logs", "cts", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

    print("=" * 80)
    print("CTS Training Configuration:")
    print(f"  Task: {args_cli.task}")
    print(f"  Number of environments: {env.num_envs}")
    print(f"  Device: {agent_cfg.device}")
    print(f"  Max iterations: {agent_cfg.max_iterations}")
    print(f"  Teacher/Student ratio: {agent_cfg.teacher_student_ratio}")
    print(f"  Privileged encoder latent dim: {agent_cfg.privileged_encoder.latent_dim}")
    print(f"  Proprioceptive encoder latent dim: {agent_cfg.proprioceptive_encoder.latent_dim}")
    print(f"  Teacher batch size: {agent_cfg.algorithm.teacher_batch_size}")
    print(f"  Student batch size: {agent_cfg.algorithm.student_batch_size}")
    print("=" * 80)

    # create CTS runner (convert config to dict as CTSRunner expects dict)
    runner = CTSRunner(
        env=env,
        train_cfg=agent_cfg.to_dict() if hasattr(agent_cfg, 'to_dict') else vars(agent_cfg),
        log_dir=log_dir,
        device=agent_cfg.device,
    )
    # set seed of the environment
    env.seed(args_cli.seed)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    app_launcher.close()
