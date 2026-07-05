# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CTS (Concurrent Teacher-Student) algorithm implementation.

Reference:
    Wang et al. "CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion" (2024)
    arXiv:2405.10830
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from .encoder_model import EncoderModel


class CTS:
    """Concurrent Teacher-Student Reinforcement Learning algorithm.

    This algorithm implements the CTS framework that trains teacher and student policies concurrently.
    The teacher uses privileged information while the student uses only proprioceptive observations.
    Both share the same policy network but use different encoders.
    """

    def __init__(
        self,
        privileged_encoder: EncoderModel,
        proprioceptive_encoder: EncoderModel,
        actor: nn.Module,
        critic: nn.Module,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-3,
        student_learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        reconstruction_loss_coef: float = 1.0,
        student_num_learning_epochs: int = 5,
        device: str = "cpu",
    ):
        """Initialize CTS algorithm.

        Args:
            privileged_encoder: Teacher encoder for privileged observations.
            proprioceptive_encoder: Student encoder for proprioceptive observations.
            actor: Policy network (shared between teacher and student).
            critic: Value function network.
            num_learning_epochs: Number of PPO epochs.
            num_mini_batches: Number of mini-batches for PPO.
            clip_param: PPO clipping parameter.
            gamma: Discount factor.
            lam: GAE lambda parameter.
            value_loss_coef: Coefficient for value loss.
            entropy_coef: Coefficient for entropy bonus.
            learning_rate: Learning rate for teacher (PPO).
            student_learning_rate: Learning rate for student encoder (supervised).
            max_grad_norm: Maximum gradient norm for clipping.
            use_clipped_value_loss: Whether to use clipped value loss.
            schedule: Learning rate schedule type.
            desired_kl: Desired KL divergence for adaptive learning rate.
            reconstruction_loss_coef: Coefficient for reconstruction loss.
            student_num_learning_epochs: Number of epochs for student encoder training.
            device: Device to run on.
        """
        self.device = device

        # Models
        self.privileged_encoder = privileged_encoder.to(device)
        self.proprioceptive_encoder = proprioceptive_encoder.to(device)
        self.actor = actor.to(device)
        self.critic = critic.to(device)

        # Hyperparameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.reconstruction_loss_coef = reconstruction_loss_coef
        self.student_num_learning_epochs = student_num_learning_epochs

        # Optimizers
        # Teacher optimizer: privileged_encoder + actor + critic
        self.teacher_params = (
            list(self.privileged_encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )
        self.teacher_optimizer = torch.optim.Adam(self.teacher_params, lr=learning_rate)
        self.learning_rate = learning_rate

        # Student optimizer: only proprioceptive_encoder (supervised learning)
        self.student_optimizer = torch.optim.Adam(
            self.proprioceptive_encoder.parameters(), lr=student_learning_rate
        )
        self.student_learning_rate = student_learning_rate

        # Storage for transitions
        self.teacher_storage = None
        self.student_storage = None

    def act_teacher(self, obs: TensorDict, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher action selection using privileged information.

        Args:
            obs: Observation dictionary containing privileged information.
            deterministic: Whether to use deterministic actions.

        Returns:
            Tuple of (actions, latent representation).
        """
        with torch.no_grad():
            # Encode privileged state
            z_privileged = self.privileged_encoder(obs)

            # Get proprioceptive observations and concatenate with latent
            obs_proprio = self._get_proprioceptive_obs(obs)
            policy_input = torch.cat([obs_proprio, z_privileged], dim=-1)

            # Sample action from policy
            actions = self.actor({"policy": policy_input}, stochastic_output=not deterministic)

            return actions, z_privileged

    def act_student(self, obs: TensorDict, obs_history: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Student action selection using only proprioceptive observations.

        Args:
            obs: Current observation dictionary.
            obs_history: Historical proprioceptive observations [batch, history_length * obs_dim].
            deterministic: Whether to use deterministic actions.

        Returns:
            Actions tensor.
        """
        with torch.no_grad():
            # Encode proprioceptive observation sequence
            z_proprio = self.proprioceptive_encoder(obs_history)

            # Get current proprioceptive observations and concatenate with latent
            obs_proprio = self._get_proprioceptive_obs(obs)
            policy_input = torch.cat([obs_proprio, z_proprio], dim=-1)

            # Sample action from policy
            actions = self.actor({"policy": policy_input}, stochastic_output=not deterministic)

            return actions

    def _get_proprioceptive_obs(self, obs: TensorDict) -> torch.Tensor:
        """Extract proprioceptive observations from observation dictionary.

        Args:
            obs: Observation dictionary.

        Returns:
            Concatenated proprioceptive observations.
        """
        # Assuming proprioceptive observations are in a specific group
        # This should be adapted based on your environment's observation structure
        if "proprioceptive" in obs:
            return obs["proprioceptive"]
        elif "policy" in obs:
            return obs["policy"]
        else:
            raise ValueError("Cannot find proprioceptive observations in TensorDict")

    def update(
        self, teacher_storage, student_storage
    ) -> dict[str, float]:
        """Update teacher and student networks.

        Args:
            teacher_storage: Rollout storage for teacher trajectories.
            student_storage: Rollout storage for student trajectories.

        Returns:
            Dictionary of training metrics.
        """
        self.teacher_storage = teacher_storage
        self.student_storage = student_storage

        metrics = {}

        # Phase 1: Teacher PPO Update
        teacher_metrics = self._update_teacher()
        metrics.update(teacher_metrics)

        # Phase 2: Student Encoder Supervised Learning
        reconstruction_loss = self._update_student_encoder()
        metrics["student/reconstruction_loss"] = reconstruction_loss

        # Phase 3: Student PPO Update (optional, if you want student to also learn via RL)
        # student_metrics = self._update_student_policy()
        # metrics.update(student_metrics)

        return metrics

    def _update_teacher(self) -> dict[str, float]:
        """Update teacher networks using PPO.

        Returns:
            Dictionary of teacher training metrics.
        """
        mean_policy_loss = 0.0
        mean_value_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0

        # PPO epochs
        for epoch in range(self.num_learning_epochs):
            # Mini-batch training
            for batch in self.teacher_storage.mini_batch_generator(self.num_mini_batches):
                # Get batch data
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]

                # Forward pass through encoder
                z_privileged = self.privileged_encoder(obs)
                obs_proprio = self._get_proprioceptive_obs(obs)
                policy_input = torch.cat([obs_proprio, z_privileged], dim=-1)

                # Policy forward pass
                action_dist = self.actor.get_distribution({"policy": policy_input})
                log_probs = action_dist.log_prob(actions).sum(dim=-1)
                entropy = action_dist.entropy().sum(dim=-1).mean()

                # Value forward pass
                values = self.critic({"policy": z_privileged})

                # PPO policy loss
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.use_clipped_value_loss:
                    value_pred_clipped = old_values + torch.clamp(
                        values - old_values, -self.clip_param, self.clip_param
                    )
                    value_losses = (values - returns).pow(2)
                    value_losses_clipped = (value_pred_clipped - returns).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (returns - values).pow(2).mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Gradient update
                self.teacher_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.teacher_params, self.max_grad_norm)
                self.teacher_optimizer.step()

                # Metrics
                with torch.no_grad():
                    kl = (old_log_probs - log_probs).mean()
                    mean_policy_loss += policy_loss.item()
                    mean_value_loss += value_loss.item()
                    mean_entropy += entropy.item()
                    mean_kl += kl.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        return {
            "teacher/policy_loss": mean_policy_loss / num_updates,
            "teacher/value_loss": mean_value_loss / num_updates,
            "teacher/entropy": mean_entropy / num_updates,
            "teacher/kl": mean_kl / num_updates,
        }

    def _update_student_encoder(self) -> float:
        """Update student encoder using supervised learning (reconstruction loss).

        Iterates over student_storage (which has obs_history), NOT teacher_storage.
        The privileged encoder runs on the student's full observation (incl. height map
        from simulation) to produce the target latent z_t.
        The student encoder runs on the student's proprioceptive history to produce z_s.
        Reconstruction loss = MSE(z_s, z_t).

        Returns:
            Mean reconstruction loss.
        """
        mean_reconstruction_loss = 0.0
        num_updates = 0

        # Training epochs for student encoder
        for epoch in range(self.student_num_learning_epochs):
            # Iterate over STUDENT storage (which contains obs_history)
            for batch in self.student_storage.mini_batch_generator(self.num_mini_batches):
                obs = batch["obs"]  # TensorDict with "policy" (235) and "proprioceptive" (48)
                obs_history = batch.get("obs_history")  # Historical proprioceptive: [B, 240]

                if obs_history is None:
                    continue

                # Teacher encoder on privileged observation (no grad)
                with torch.no_grad():
                    z_privileged = self.privileged_encoder(obs)

                # Student encoder on proprioceptive history
                z_proprio = self.proprioceptive_encoder(obs_history)

                # Reconstruction loss (MSE between student and teacher latent)
                reconstruction_loss = F.mse_loss(z_proprio, z_privileged)

                # Backward pass
                self.student_optimizer.zero_grad()
                reconstruction_loss.backward()
                nn.utils.clip_grad_norm_(self.proprioceptive_encoder.parameters(), self.max_grad_norm)
                self.student_optimizer.step()

                mean_reconstruction_loss += reconstruction_loss.item()
                num_updates += 1

        if num_updates == 0:
            return 0.0

        return mean_reconstruction_loss / num_updates

    def get_student_policy(self) -> tuple[nn.Module, nn.Module]:
        """Get student policy components for deployment.

        Returns:
            Tuple of (proprioceptive_encoder, actor).
        """
        return self.proprioceptive_encoder, self.actor
