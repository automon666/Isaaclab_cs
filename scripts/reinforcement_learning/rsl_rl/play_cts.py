#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a CTS (Concurrent Teacher-Student) trained policy.

Example usage:
    python scripts/reinforcement_learning/rsl_rl/play_cts.py --task Isaac-Velocity-Flat-EVA02-v0 \\
        --checkpoint logs/cts/cts_Isaac-Velocity-Flat-EVA02-v0/2026-07-03_21-06-27/model_100.pt

    # Or auto-detect latest checkpoint:
    python scripts/reinforcement_learning/rsl_rl/play_cts.py --task Isaac-Velocity-Flat-EVA02-v0
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play with a CTS-trained RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_student", action="store_true", default=True,
                    help="Use student policy (proprioceptive only). Set --no-use_student to use teacher.")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras for rendering (required for visual play)
args_cli.enable_cameras = True

# Disable headless mode for play - must clear env var too (training may have set HEADLESS=1)
args_cli.headless = False
os.environ["HEADLESS"] = "0"

# Set ROS_PACKAGE_PATH so Isaac Sim URDF importer resolves package://EVA02_description paths
os.environ.setdefault("ROS_PACKAGE_PATH", "")
ros_paths = ["/home/tino66/Isaaclab_cs/assets"]
if os.environ["ROS_PACKAGE_PATH"]:
    ros_paths.append(os.environ["ROS_PACKAGE_PATH"])
os.environ["ROS_PACKAGE_PATH"] = ":".join(ros_paths)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
from collections import deque
from datetime import datetime
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.cts_networks import CTSActor, CTSCritic
from isaaclab_rl.rsl_rl.encoder_model import EncoderModel

import isaaclab_tasks  # noqa: F401


def find_latest_checkpoint(log_root: str) -> str | None:
    """Find the latest checkpoint in the log directory.

    Args:
        log_root: Root directory containing timestamped run directories.

    Returns:
        Path to the latest model checkpoint, or None if not found.
    """
    if not os.path.isdir(log_root):
        return None

    # Find all run directories (timestamped)
    run_dirs = sorted([
        d for d in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, d))
    ], reverse=True)

    for run_dir in run_dirs:
        run_path = os.path.join(log_root, run_dir)
        # Find model files, sorted by iteration number
        model_files = sorted([
            f for f in os.listdir(run_path)
            if f.startswith("model_") and f.endswith(".pt")
        ], key=lambda f: int(f.replace("model_", "").replace(".pt", "")), reverse=True)

        if model_files:
            return os.path.join(run_path, model_files[0])

    return None


def main():
    """Play with CTS-trained agent."""
    if args_cli.task is None:
        raise ValueError("Please specify a task using --task argument")

    # Determine checkpoint path
    if args_cli.checkpoint:
        checkpoint_path = args_cli.checkpoint
    else:
        # Auto-detect latest checkpoint
        log_root = os.path.join("logs", "cts", f"cts_{args_cli.task}")
        log_root = os.path.abspath(log_root)
        checkpoint_path = find_latest_checkpoint(log_root)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint found in {log_root}. "
                f"Please specify --checkpoint path explicitly."
            )

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint.get("iteration", "unknown")
    print(f"[INFO] Checkpoint iteration: {iteration}")

    # Create environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    env_spec = gym.spec(args_cli.task)
    env_cfg_entry_point = env_spec.kwargs.get("env_cfg_entry_point")
    if env_cfg_entry_point is None:
        raise ValueError(f"Task {args_cli.task} does not have env_cfg_entry_point in gym registry")

    env_cfg = env_cfg_entry_point()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # Create play environment (with rendering)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Wrap for video recording
    if args_cli.video:
        log_dir = os.path.dirname(checkpoint_path)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Get observation to determine dimensions
    obs = env.get_observations()

    # Determine observation dimension
    if "policy" in obs:
        obs_dim = obs["policy"].shape[-1]
        obs_groups = {
            "privileged": ["policy"],
            "proprioceptive": ["policy"],
        }
    elif "proprioceptive" in obs:
        obs_dim = obs["proprioceptive"].shape[-1]
        obs_groups = {
            "privileged": ["proprioceptive"],
            "proprioceptive": ["proprioceptive"],
        }
    else:
        obs_dim = obs.shape[-1] if isinstance(obs, torch.Tensor) else list(obs.values())[0].shape[-1]
        obs_groups = {
            "privileged": ["policy"],
            "proprioceptive": ["policy"],
        }

    num_actions = env.num_actions
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Build encoder and actor models
    latent_dim = 24
    hidden_dims = [512, 256]

    privileged_encoder = EncoderModel(
        obs=obs,
        obs_groups=obs_groups,
        obs_set="privileged",
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation="elu",
        obs_normalization=True,
        use_history=False,
    ).to(device)

    proprioceptive_encoder = EncoderModel(
        obs=obs,
        obs_groups=obs_groups,
        obs_set="proprioceptive",
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation="elu",
        obs_normalization=True,
        use_history=True,
        history_length=5,
    ).to(device)

    actor = CTSActor(
        input_dim=obs_dim + latent_dim,
        num_actions=num_actions,
        hidden_dims=[512, 256, 128],
        activation="elu",
    ).to(device)

    # Load weights
    privileged_encoder.load_state_dict(checkpoint["privileged_encoder"])
    proprioceptive_encoder.load_state_dict(checkpoint["proprioceptive_encoder"])
    if "actor" in checkpoint:
        actor.load_state_dict(checkpoint["actor"])
        print("[INFO] Actor weights loaded from checkpoint")
    else:
        print("[WARNING] No actor weights found in checkpoint (old format)")

    # Set to eval mode
    privileged_encoder.eval()
    proprioceptive_encoder.eval()
    actor.eval()

    policy_type = "student" if args_cli.use_student else "teacher"
    print(f"[INFO] Using {policy_type} policy for inference")
    print(f"[INFO] Running {args_cli.num_episodes} episodes, max {args_cli.max_steps} steps each")

    # Initialize observation history buffer (for student)
    history_length = 5
    obs_history = deque(maxlen=history_length)

    for episode in range(args_cli.num_episodes):
        obs = env.get_observations()
        done = False
        step = 0
        total_reward = 0.0

        # Initialize history
        obs_history.clear()
        proprio = _get_proprioceptive_obs(obs)
        for _ in range(history_length):
            obs_history.append(proprio.clone())

        while not done and step < args_cli.max_steps:
            with torch.no_grad():
                if args_cli.use_student:
                    # Student: use proprioceptive encoder with history
                    obs_history_tensor = torch.cat(list(obs_history), dim=-1)
                    z = proprioceptive_encoder(obs_history_tensor)
                else:
                    # Teacher: use privileged encoder
                    z = privileged_encoder(obs)

                # Get proprioceptive obs and concatenate with latent
                obs_proprio = _get_proprioceptive_obs(obs)
                policy_input = torch.cat([obs_proprio, z], dim=-1)

                # Get action (deterministic for evaluation)
                actions = actor({"policy": policy_input}, stochastic_output=False)

            # Step environment
            next_obs, rewards, dones, infos = env.step(actions)

            # Update history
            next_proprio = _get_proprioceptive_obs(next_obs)
            obs_history.append(next_proprio.clone())

            total_reward += rewards.mean().item()
            obs = next_obs
            step += 1

            # Check if any environment is done
            if dones.any():
                done = True

        print(f"  Episode {episode + 1}/{args_cli.num_episodes}: "
              f"steps={step}, reward={total_reward:.2f}")

    # Close environment
    env.close()
    print("[INFO] Play completed.")


def _get_proprioceptive_obs(obs: TensorDict) -> torch.Tensor:
    """Extract proprioceptive observations.

    Args:
        obs: Observation dictionary.

    Returns:
        Proprioceptive observations tensor.
    """
    if "proprioceptive" in obs:
        return obs["proprioceptive"]
    elif "policy" in obs:
        return obs["policy"]
    else:
        raise ValueError("Cannot find proprioceptive observations in TensorDict")


if __name__ == "__main__":
    main()
    simulation_app.close()
