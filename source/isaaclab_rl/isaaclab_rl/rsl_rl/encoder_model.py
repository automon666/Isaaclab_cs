# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Encoder model for CTS (Concurrent Teacher-Student) architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


class EncoderModel(nn.Module):
    """VAE-style encoder for privileged/proprioceptive observations.

    This encoder is used in the CTS architecture to encode either:
    - Privileged information (teacher encoder): encodes full state including terrain, contact forces, etc.
    - Proprioceptive information (student encoder): encodes observation history sequence

    Reference:
        Wang et al. "CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion" (2024)
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        latent_dim: int = 24,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        activation: str = "elu",
        obs_normalization: bool = True,
        use_history: bool = False,
        history_length: int = 5,
    ) -> None:
        """Initialize the encoder model.

        Args:
            obs: Observation Dictionary containing all observation groups.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this encoder ("privileged" for teacher, "proprioceptive" for student).
            latent_dim: Dimension of the latent representation (output dimension).
            hidden_dims: Hidden dimensions of the MLP encoder.
            activation: Activation function (e.g., "elu", "relu", "tanh").
            obs_normalization: Whether to normalize observations before encoding.
            use_history: Whether to use observation history sequence (True for student, False for teacher).
            history_length: Length of observation history window (paper uses 5).
        """
        super().__init__()

        self.obs_set = obs_set
        self.latent_dim = latent_dim
        self.use_history = use_history
        self.history_length = history_length
        self.obs_normalization = obs_normalization

        # Resolve observation groups and compute input dimension
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        # If using history, multiply input dimension by history length
        if use_history:
            self.input_dim = self.obs_dim * history_length
        else:
            self.input_dim = self.obs_dim

        # Observation normalization
        if obs_normalization:
            self.obs_normalizer = nn.BatchNorm1d(self.input_dim)
        else:
            self.obs_normalizer = nn.Identity()

        # Build MLP encoder
        self.encoder = self._build_mlp(self.input_dim, latent_dim, hidden_dims, activation)

    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        """Get observation groups and total dimension for a given observation set.

        Args:
            obs: Observation dictionary.
            obs_groups: Mapping from observation sets to observation group names.
            obs_set: The observation set to use (e.g., "privileged", "proprioceptive").

        Returns:
            Tuple of (list of observation group names, total observation dimension).
        """
        if obs_set not in obs_groups:
            raise ValueError(
                f"Observation set '{obs_set}' not found in obs_groups. Available sets: {list(obs_groups.keys())}"
            )

        groups = obs_groups[obs_set]
        obs_dim = 0

        for group in groups:
            if group not in obs:
                raise ValueError(
                    f"Observation group '{group}' not found in observations. Available groups: {list(obs.keys())}"
                )
            obs_dim += obs[group].shape[-1]

        return groups, obs_dim

    def _build_mlp(
        self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] | list[int], activation: str
    ) -> nn.Module:
        """Build a multi-layer perceptron.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            hidden_dims: Tuple/list of hidden layer dimensions.
            activation: Activation function name.

        Returns:
            MLP module.
        """
        activation_fn = self._get_activation(activation)

        layers = []
        dims = [input_dim] + list(hidden_dims) + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # Add activation for all layers except the last one
            if i < len(dims) - 2:
                layers.append(activation_fn)

        return nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name.

        Args:
            activation: Name of activation function.

        Returns:
            Activation module.
        """
        activations = {
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid(),
        }

        if activation.lower() not in activations:
            raise ValueError(
                f"Unknown activation function: {activation}. Available: {list(activations.keys())}"
            )

        return activations[activation.lower()]

    def forward(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        """Encode observations to latent representation.

        Args:
            obs: Either a TensorDict containing observation groups, or a tensor of stacked observations.
                 If using history, this should be a tensor of shape [batch, history_length * obs_dim].

        Returns:
            Latent representation of shape [batch, latent_dim], normalized to unit hypersphere.
        """
        # Handle TensorDict input
        if isinstance(obs, TensorDict):
            obs_list = []
            for group in self.obs_groups:
                obs_list.append(obs[group])
            obs_tensor = torch.cat(obs_list, dim=-1)
        else:
            obs_tensor = obs

        # Normalize observations
        obs_normalized = self.obs_normalizer(obs_tensor)

        # Encode to latent space
        latent = self.encoder(obs_normalized)

        # L2 normalization to map to unit hypersphere (as described in the paper)
        latent_normalized = F.normalize(latent, p=2, dim=-1)

        return latent_normalized

    def update_normalization(self, obs: TensorDict | torch.Tensor) -> None:
        """Update observation normalization statistics (for BatchNorm).

        Args:
            obs: Observations to update normalization with.
        """
        if not self.obs_normalization:
            return

        # Handle TensorDict input
        if isinstance(obs, TensorDict):
            obs_list = []
            for group in self.obs_groups:
                obs_list.append(obs[group])
            obs_tensor = torch.cat(obs_list, dim=-1)
        else:
            obs_tensor = obs

        # Update BatchNorm statistics
        self.obs_normalizer.train()
        _ = self.obs_normalizer(obs_tensor)
        self.obs_normalizer.eval()
