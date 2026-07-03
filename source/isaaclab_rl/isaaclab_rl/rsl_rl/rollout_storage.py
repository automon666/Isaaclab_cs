# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple rollout storage for CTS training."""

import torch


class RolloutStorage:
    """Simple storage for rollout data with mini-batch generation."""

    def __init__(self, data: dict):
        """Initialize storage from data dictionary.

        Args:
            data: Dictionary containing rollout data (obs, actions, rewards, etc.)
        """
        self.data = {}

        # Stack lists into tensors
        for key, value_list in data.items():
            if len(value_list) > 0:
                self.data[key] = torch.stack(value_list, dim=0)  # [T, N, ...]

        self.num_steps = len(data["obs"]) if "obs" in data and len(data["obs"]) > 0 else 0
        self.num_envs = self.data["obs"].shape[1] if self.num_steps > 0 else 0

    def mini_batch_generator(self, num_mini_batches: int):
        """Generate mini-batches from stored data.

        Args:
            num_mini_batches: Number of mini-batches to generate.

        Yields:
            Dictionary of mini-batch data.
        """
        if self.num_steps == 0:
            return

        # Flatten time and env dimensions: [T, N, ...] -> [T*N, ...]
        flat_data = {}
        for key, value in self.data.items():
            if value.ndim >= 2:
                # Flatten first two dimensions
                flat_data[key] = value.reshape(-1, *value.shape[2:])
            else:
                flat_data[key] = value

        total_samples = self.num_steps * self.num_envs
        mini_batch_size = total_samples // num_mini_batches

        # Generate random permutation
        indices = torch.randperm(total_samples)

        for i in range(num_mini_batches):
            start_idx = i * mini_batch_size
            end_idx = start_idx + mini_batch_size if i < num_mini_batches - 1 else total_samples

            batch_indices = indices[start_idx:end_idx]

            # Extract mini-batch
            mini_batch = {}
            for key, value in flat_data.items():
                mini_batch[key] = value[batch_indices]

            yield mini_batch
