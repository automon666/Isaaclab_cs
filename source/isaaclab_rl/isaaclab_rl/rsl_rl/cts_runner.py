# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CTS (Concurrent Teacher-Student) runner for training."""

from __future__ import annotations

import os
import time
from collections import deque
from datetime import datetime

import torch
from tensordict import TensorDict

from isaaclab.envs import VecEnv

from .cts_algorithm import CTS
from .encoder_model import EncoderModel


class CTSRunner:
    """Runner for CTS (Concurrent Teacher-Student) training.

    This runner manages the concurrent training of teacher and student policies,
    handling rollout collection, network updates, and logging.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        """Initialize CTS runner.

        Args:
            env: Vectorized environment.
            train_cfg: Training configuration dictionary.
            log_dir: Directory for logging.
            device: Device to run on.
        """
        self.env = env
        self.cfg = train_cfg
        self.log_dir = log_dir
        self.device = device

        # Extract configuration
        self.num_envs = env.num_envs
        self.teacher_steps_per_env = train_cfg.get("teacher_steps_per_env", 24)
        self.student_steps_per_env = train_cfg.get("student_steps_per_env", 24)

        # Calculate teacher/student split based on ratio
        teacher_student_ratio = train_cfg.get("teacher_student_ratio", 3.0)
        num_teacher = int(self.num_envs * teacher_student_ratio / (teacher_student_ratio + 1))
        num_student = self.num_envs - num_teacher

        self.num_envs_teacher = num_teacher
        self.num_envs_student = num_student

        print(f"[CTS] Total envs: {self.num_envs}")
        print(f"[CTS] Teacher envs: {self.num_envs_teacher}")
        print(f"[CTS] Student envs: {self.num_envs_student}")

        # Initialize networks
        self._init_networks()

        # Initialize observation history buffer for student
        history_length = train_cfg.get("proprioceptive_encoder", {}).get("history_length", 5)
        self.obs_history_buffer = deque(maxlen=history_length)

        # Training statistics
        self.tot_timesteps = 0
        self.tot_iterations = 0

    def _init_networks(self):
        """Initialize encoder, actor, and critic networks."""
        # Get sample observations
        obs = self.env.get_observations()

        # Extract configuration
        encoder_cfg = self.cfg.get("privileged_encoder", {})
        student_encoder_cfg = self.cfg.get("proprioceptive_encoder", {})
        actor_cfg = self.cfg.get("actor", {})
        critic_cfg = self.cfg.get("critic", {})
        algorithm_cfg = self.cfg.get("algorithm", {})

        # Observation groups
        obs_groups = self.cfg.get("obs_groups", {"privileged": ["policy", "privileged"], "proprioceptive": ["policy"]})

        # Create privileged encoder (teacher)
        self.privileged_encoder = EncoderModel(
            obs=obs,
            obs_groups=obs_groups,
            obs_set="privileged",
            latent_dim=encoder_cfg.get("latent_dim", 24),
            hidden_dims=encoder_cfg.get("hidden_dims", [512, 256]),
            activation=encoder_cfg.get("activation", "elu"),
            obs_normalization=encoder_cfg.get("obs_normalization", True),
            use_history=False,
        )

        # Create proprioceptive encoder (student)
        self.proprioceptive_encoder = EncoderModel(
            obs=obs,
            obs_groups=obs_groups,
            obs_set="proprioceptive",
            latent_dim=student_encoder_cfg.get("latent_dim", 24),
            hidden_dims=student_encoder_cfg.get("hidden_dims", [512, 256]),
            activation=student_encoder_cfg.get("activation", "elu"),
            obs_normalization=student_encoder_cfg.get("obs_normalization", True),
            use_history=True,
            history_length=student_encoder_cfg.get("history_length", 5),
        )

        # Create actor (policy network) - shared between teacher and student
        # Note: Actor input = proprioceptive_obs + latent
        from .rl_cfg import RslRlMLPModelCfg

        # This is a placeholder - you need to properly instantiate the actor based on your setup
        print("[CTS] Actor and Critic network initialization needs to be completed based on rsl_rl structure")

        # Create CTS algorithm
        self.alg = CTS(
            privileged_encoder=self.privileged_encoder,
            proprioceptive_encoder=self.proprioceptive_encoder,
            actor=None,  # Placeholder
            critic=None,  # Placeholder
            num_learning_epochs=algorithm_cfg.get("num_learning_epochs", 5),
            num_mini_batches=algorithm_cfg.get("num_mini_batches", 4),
            clip_param=algorithm_cfg.get("clip_param", 0.2),
            gamma=algorithm_cfg.get("gamma", 0.99),
            lam=algorithm_cfg.get("lam", 0.95),
            value_loss_coef=algorithm_cfg.get("value_loss_coef", 1.0),
            entropy_coef=algorithm_cfg.get("entropy_coef", 0.01),
            learning_rate=algorithm_cfg.get("learning_rate", 1e-3),
            student_learning_rate=algorithm_cfg.get("student_learning_rate", 1e-3),
            max_grad_norm=algorithm_cfg.get("max_grad_norm", 1.0),
            reconstruction_loss_coef=algorithm_cfg.get("reconstruction_loss_coef", 1.0),
            student_num_learning_epochs=algorithm_cfg.get("student_num_learning_epochs", 5),
            device=self.device,
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Main training loop for CTS.

        Args:
            num_learning_iterations: Number of training iterations.
            init_at_random_ep_len: Whether to initialize at random episode length.
        """
        print(f"[CTS] Starting training for {num_learning_iterations} iterations")

        # Get initial observations
        obs = self.env.get_observations()

        # Initialize observation history
        for _ in range(self.obs_history_buffer.maxlen):
            self.obs_history_buffer.append(self._get_proprioceptive_obs(obs))

        start_time = time.time()

        for it in range(num_learning_iterations):
            iter_start = time.time()

            # ===== Rollout Phase =====
            teacher_data, student_data = self._collect_rollouts(obs)

            # ===== Update Phase =====
            if it % 10 == 0:
                print(f"[CTS] Iteration {it}/{num_learning_iterations}")

            metrics = self.alg.update(teacher_data, student_data)

            # ===== Logging =====
            if it % 10 == 0:
                iter_time = time.time() - iter_start
                print(f"  Time: {iter_time:.2f}s")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")

            # ===== Save checkpoints =====
            if self.log_dir and it % 100 == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            self.tot_iterations += 1

        total_time = time.time() - start_time
        print(f"[CTS] Training completed in {total_time:.2f} seconds")

    def _collect_rollouts(self, obs: TensorDict) -> tuple[dict, dict]:
        """Collect rollout data from teacher and student environments.

        Args:
            obs: Current observations.

        Returns:
            Tuple of (teacher_data, student_data) dictionaries.
        """
        teacher_data = {"obs": [], "actions": [], "rewards": [], "dones": [], "log_probs": [], "values": []}
        student_data = {"obs": [], "obs_history": [], "actions": [], "rewards": [], "dones": []}

        # Split observations for teacher and student
        obs_teacher = self._split_obs(obs, 0, self.num_envs_teacher)
        obs_student = self._split_obs(obs, self.num_envs_teacher, self.num_envs)

        # Teacher rollout
        with torch.no_grad():
            for step in range(self.teacher_steps_per_env):
                actions, latent = self.alg.act_teacher(obs_teacher)
                next_obs, rewards, dones, infos = self.env.step(actions)

                teacher_data["obs"].append(obs_teacher)
                teacher_data["actions"].append(actions)
                teacher_data["rewards"].append(rewards)
                teacher_data["dones"].append(dones)

                obs_teacher = next_obs

        # Student rollout
        with torch.no_grad():
            for step in range(self.student_steps_per_env):
                # Get observation history
                obs_history = torch.cat(list(self.obs_history_buffer), dim=-1)

                actions = self.alg.act_student(obs_student, obs_history)
                next_obs, rewards, dones, infos = self.env.step(actions)

                student_data["obs"].append(obs_student)
                student_data["obs_history"].append(obs_history)
                student_data["actions"].append(actions)
                student_data["rewards"].append(rewards)
                student_data["dones"].append(dones)

                # Update history
                self.obs_history_buffer.append(self._get_proprioceptive_obs(next_obs))

                obs_student = next_obs

        return teacher_data, student_data

    def _split_obs(self, obs: TensorDict, start_idx: int, end_idx: int) -> TensorDict:
        """Split observations for a subset of environments.

        Args:
            obs: Full observation dictionary.
            start_idx: Start index.
            end_idx: End index.

        Returns:
            Subset of observations.
        """
        obs_split = TensorDict({}, batch_size=end_idx - start_idx)
        for key, value in obs.items():
            obs_split[key] = value[start_idx:end_idx]
        return obs_split

    def _get_proprioceptive_obs(self, obs: TensorDict) -> torch.Tensor:
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

    def save(self, path: str):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "privileged_encoder": self.privileged_encoder.state_dict(),
                "proprioceptive_encoder": self.proprioceptive_encoder.state_dict(),
                "iteration": self.tot_iterations,
            },
            path,
        )
        print(f"[CTS] Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.privileged_encoder.load_state_dict(checkpoint["privileged_encoder"])
        self.proprioceptive_encoder.load_state_dict(checkpoint["proprioceptive_encoder"])
        self.tot_iterations = checkpoint.get("iteration", 0)
        print(f"[CTS] Model loaded from {path}")
