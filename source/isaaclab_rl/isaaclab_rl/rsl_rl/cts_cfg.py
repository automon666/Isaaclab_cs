# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for CTS (Concurrent Teacher-Student) training."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from .rl_cfg import RslRlBaseRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlEncoderCfg:
    """Configuration for encoder networks in CTS."""

    latent_dim: int = 24
    """Dimension of the latent representation. Defaults to 24 (as in the paper)."""

    hidden_dims: list[int] = MISSING
    """Hidden dimensions of the encoder MLP."""

    activation: str = "elu"
    """Activation function for the encoder. Defaults to elu."""

    obs_normalization: bool = True
    """Whether to normalize observations. Defaults to True."""

    use_history: bool = False
    """Whether to use observation history (True for student, False for teacher). Defaults to False."""

    history_length: int = 5
    """Length of observation history window. Defaults to 5 (as in the paper)."""


@configclass
class RslRlCtsAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the CTS algorithm.

    Extends PPO algorithm configuration with CTS-specific parameters.
    """

    class_name: str = "CTS"
    """The algorithm class name. Defaults to CTS."""

    # CTS-specific parameters
    teacher_batch_size: int = MISSING
    """Batch size for teacher training. Paper uses 8192 × 3 = 24576."""

    student_batch_size: int = MISSING
    """Batch size for student training. Paper uses 2048 × 6 = 12288."""

    reconstruction_loss_coef: float = 1.0
    """Coefficient for the reconstruction loss (L^rec). Defaults to 1.0."""

    student_learning_rate: float = 1e-3
    """Learning rate for student encoder supervised learning. Defaults to 1e-3."""

    student_num_learning_epochs: int = 5
    """Number of learning epochs for student encoder. Defaults to 5."""


@configclass
class RslRlCtsRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration for the CTS runner."""

    class_name: str = "CTSRunner"
    """The runner class name. Defaults to CTSRunner."""

    # Encoder configurations
    privileged_encoder: RslRlEncoderCfg = MISSING
    """Configuration for the privileged encoder (teacher)."""

    proprioceptive_encoder: RslRlEncoderCfg = MISSING
    """Configuration for the proprioceptive encoder (student)."""

    # Actor and Critic configurations (shared between teacher and student)
    actor: RslRlMLPModelCfg = MISSING
    """The actor (policy) configuration."""

    critic: RslRlMLPModelCfg = MISSING
    """The critic (value function) configuration."""

    # Algorithm configuration
    algorithm: RslRlCtsAlgorithmCfg = MISSING
    """The CTS algorithm configuration."""

    # Training parameters
    teacher_steps_per_env: int = 24
    """Number of steps per teacher environment per update. Defaults to 24."""

    student_steps_per_env: int = 24
    """Number of steps per student environment per update. Defaults to 24."""

    num_envs_teacher: int = MISSING
    """Number of teacher environments. Paper uses 6144."""

    num_envs_student: int = MISSING
    """Number of student environments. Paper uses 2048."""

    teacher_student_ratio: float = 3.0
    """Ratio of teacher to student environments. Defaults to 3.0 (as in the paper)."""
