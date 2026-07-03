# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple Actor and Critic wrappers for CTS."""

import torch
import torch.nn as nn


class CTSActor(nn.Module):
    """Actor network for CTS that accepts dict input and stochastic_output parameter."""

    def __init__(self, input_dim: int, num_actions: int, hidden_dims: list[int], activation: str = "elu"):
        """Initialize CTS Actor.

        Args:
            input_dim: Dimension of input (proprioceptive_obs + latent).
            num_actions: Number of action dimensions.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function name.
        """
        super().__init__()

        # Build MLP
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if activation == "elu":
                layers.append(nn.ELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            current_dim = hidden_dim

        # Output layer (mean of actions)
        layers.append(nn.Linear(current_dim, num_actions))
        self.mlp = nn.Sequential(*layers)

        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(num_actions))

    def forward(self, obs_dict: dict, stochastic_output: bool = True):
        """Forward pass.

        Args:
            obs_dict: Dictionary with 'policy' key containing latent representation.
            stochastic_output: Whether to sample from distribution (True) or return mean (False).

        Returns:
            Action tensor.
        """
        latent = obs_dict["policy"]
        mean = self.mlp(latent)

        if stochastic_output:
            # Sample from Gaussian distribution
            std = torch.exp(self.log_std)
            noise = torch.randn_like(mean)
            return mean + noise * std
        else:
            # Return mean (deterministic)
            return mean

    def get_distribution(self, obs_dict: dict) -> torch.distributions.Normal:
        """Get the action distribution for the given observation.

        Args:
            obs_dict: Dictionary with 'policy' key containing latent representation.

        Returns:
            Normal distribution with learned mean and std.
        """
        latent = obs_dict["policy"]
        mean = self.mlp(latent)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std)

    def get_distribution(self, obs_dict: dict):
        """Get action distribution.

        Args:
            obs_dict: Dictionary with 'policy' key containing latent representation.

        Returns:
            Normal distribution object.
        """
        latent = obs_dict["policy"]
        mean = self.mlp(latent)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std)

    def evaluate_actions(self, obs_dict: dict, actions: torch.Tensor):
        """Evaluate actions and get log probabilities and entropy.

        Args:
            obs_dict: Dictionary with 'policy' key containing latent representation.
            actions: Actions to evaluate.

        Returns:
            Tuple of (log_probs, entropy).
        """
        dist = self.get_distribution(obs_dict)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy


class CTSCritic(nn.Module):
    """Critic network for CTS that accepts dict input."""

    def __init__(self, latent_dim: int, hidden_dims: list[int], activation: str = "elu"):
        """Initialize CTS Critic.

        Args:
            latent_dim: Dimension of latent representation from encoder.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function name.
        """
        super().__init__()

        # Build MLP
        layers = []
        input_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "elu":
                layers.append(nn.ELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            input_dim = hidden_dim

        # Output layer (value)
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs_dict: dict):
        """Forward pass.

        Args:
            obs_dict: Dictionary with 'policy' key containing latent representation.

        Returns:
            Value tensor.
        """
        latent = obs_dict["policy"]
        return self.mlp(latent)
