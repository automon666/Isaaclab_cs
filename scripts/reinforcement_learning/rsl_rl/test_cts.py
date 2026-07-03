#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script for CTS (Concurrent Teacher-Student) implementation.

This script performs basic tests to verify that the CTS components are correctly implemented.
"""

# Initialize Isaac Sim before importing any other modules
from isaacsim import SimulationApp

# Create a headless simulation app (required for Isaac Sim imports)
simulation_app = SimulationApp({"headless": True})

import torch
from tensordict import TensorDict

# Test imports
print("[TEST] Testing CTS imports...")
try:
    from isaaclab_rl.rsl_rl import (
        CTS,
        CTSRunner,
        EncoderModel,
        RslRlCtsRunnerCfg,
        RslRlEncoderCfg,
        RslRlCtsAlgorithmCfg,
    )
    print("✓ All CTS components imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    simulation_app.close()
    exit(1)


def test_encoder_model():
    """Test the encoder model."""
    print("\n[TEST] Testing EncoderModel...")

    # Create dummy observations
    batch_size = 128
    obs = TensorDict(
        {
            "policy": torch.randn(batch_size, 48),
            "privileged": torch.randn(batch_size, 72),
        },
        batch_size=batch_size,
    )

    obs_groups = {
        "privileged": ["policy", "privileged"],
        "proprioceptive": ["policy"],
    }

    # Test privileged encoder (teacher)
    try:
        privileged_encoder = EncoderModel(
            obs=obs,
            obs_groups=obs_groups,
            obs_set="privileged",
            latent_dim=24,
            hidden_dims=[512, 256],
            activation="elu",
            obs_normalization=True,
            use_history=False,
        )
        print("✓ Privileged encoder created successfully")

        # Forward pass
        latent = privileged_encoder(obs)
        assert latent.shape == (batch_size, 24), f"Expected shape ({batch_size}, 24), got {latent.shape}"
        assert torch.allclose(torch.norm(latent, dim=-1), torch.ones(batch_size), atol=1e-5), "Latent not normalized"
        print(f"✓ Privileged encoder forward pass successful: {latent.shape}")

    except Exception as e:
        print(f"✗ Privileged encoder test failed: {e}")
        return False

    # Test proprioceptive encoder (student)
    try:
        proprioceptive_encoder = EncoderModel(
            obs=obs,
            obs_groups=obs_groups,
            obs_set="proprioceptive",
            latent_dim=24,
            hidden_dims=[512, 256],
            activation="elu",
            obs_normalization=True,
            use_history=True,
            history_length=5,
        )
        print("✓ Proprioceptive encoder created successfully")

        # Forward pass with history
        obs_history = torch.randn(batch_size, 48 * 5)  # 5 steps of history
        latent = proprioceptive_encoder(obs_history)
        assert latent.shape == (batch_size, 24), f"Expected shape ({batch_size}, 24), got {latent.shape}"
        assert torch.allclose(torch.norm(latent, dim=-1), torch.ones(batch_size), atol=1e-5), "Latent not normalized"
        print(f"✓ Proprioceptive encoder forward pass successful: {latent.shape}")

    except Exception as e:
        print(f"✗ Proprioceptive encoder test failed: {e}")
        return False

    return True


def test_cts_algorithm():
    """Test the CTS algorithm."""
    print("\n[TEST] Testing CTS Algorithm...")

    batch_size = 128
    obs = TensorDict(
        {
            "policy": torch.randn(batch_size, 48),
            "privileged": torch.randn(batch_size, 72),
        },
        batch_size=batch_size,
    )

    obs_groups = {
        "privileged": ["policy", "privileged"],
        "proprioceptive": ["policy"],
    }

    try:
        # Create encoders
        privileged_encoder = EncoderModel(
            obs=obs,
            obs_groups=obs_groups,
            obs_set="privileged",
            latent_dim=24,
            hidden_dims=[256, 128],
            activation="elu",
            use_history=False,
        )

        proprioceptive_encoder = EncoderModel(
            obs=obs,
            obs_groups=obs_groups,
            obs_set="proprioceptive",
            latent_dim=24,
            hidden_dims=[256, 128],
            activation="elu",
            use_history=True,
            history_length=5,
        )

        # Create dummy actor and critic
        import torch.nn as nn

        class DummyActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(48 + 24, 256),  # obs + latent
                    nn.ELU(),
                    nn.Linear(256, 12),  # 12 actions
                )

            def forward(self, obs_dict, stochastic_output=False):
                x = obs_dict["policy"]
                return self.mlp(x)

        class DummyCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(72 + 24, 256),  # full obs + latent
                    nn.ELU(),
                    nn.Linear(256, 1),
                )

            def forward(self, obs, latent):
                obs_tensor = torch.cat([obs["policy"], obs["privileged"]], dim=-1)
                x = torch.cat([obs_tensor, latent], dim=-1)
                return self.mlp(x).squeeze(-1)

        actor = DummyActor()
        critic = DummyCritic()

        # Create CTS algorithm
        cts = CTS(
            privileged_encoder=privileged_encoder,
            proprioceptive_encoder=proprioceptive_encoder,
            actor=actor,
            critic=critic,
            num_learning_epochs=2,
            num_mini_batches=2,
            learning_rate=1e-3,
            device="cpu",
        )

        print("✓ CTS algorithm created successfully")

        # Test teacher action
        actions, latent = cts.act_teacher(obs, deterministic=True)
        print(f"✓ Teacher action shape: {actions.shape}, latent: {latent.shape}")

        # Test student action
        obs_history = torch.randn(batch_size, 48 * 5)
        actions = cts.act_student(obs, obs_history, deterministic=True)
        print(f"✓ Student action shape: {actions.shape}")

    except Exception as e:
        print(f"✗ CTS algorithm test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_configuration():
    """Test configuration classes."""
    print("\n[TEST] Testing Configuration Classes...")

    try:
        # Test encoder config
        encoder_cfg = RslRlEncoderCfg(
            latent_dim=24,
            hidden_dims=[512, 256],
            activation="elu",
            obs_normalization=True,
            use_history=False,
            history_length=5,
        )
        print(f"✓ Encoder config created: latent_dim={encoder_cfg.latent_dim}")

        # Test algorithm config
        algo_cfg = RslRlCtsAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            teacher_batch_size=24576,
            student_batch_size=12288,
            reconstruction_loss_coef=1.0,
        )
        print(f"✓ Algorithm config created: class_name={algo_cfg.class_name}")

        # Test runner config
        runner_cfg = RslRlCtsRunnerCfg(
            seed=42,
            device="cpu",
            num_steps_per_env=24,
            max_iterations=1000,
            privileged_encoder=encoder_cfg,
            proprioceptive_encoder=encoder_cfg,
        )
        print(f"✓ Runner config created: max_iterations={runner_cfg.max_iterations}")

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("CTS Implementation Test Suite")
    print("=" * 60)

    results = {
        "Encoder Model": test_encoder_model(),
        "CTS Algorithm": test_cts_algorithm(),
        "Configuration": test_configuration(),
    }

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    all_passed = True
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
        if not result:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! CTS implementation is ready to use.")
        exit_code = 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        exit_code = 1

    # Close the simulation app
    simulation_app.close()
    return exit_code


if __name__ == "__main__":
    exit(main())
