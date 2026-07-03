#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script for EVA02 CTS training configuration."""

from isaacsim import SimulationApp

# Launch simulation
simulation_app = SimulationApp({"headless": True})

import gymnasium as gym
import torch

# Register EVA02 CTS tasks
import isaaclab_tasks.direct.eva02  # noqa: F401

def main():
    """Test EVA02 CTS configuration."""

    print("=" * 80)
    print("Testing EVA02 CTS Configuration")
    print("=" * 80)

    # Test CTS task registration
    print("\n[1/3] Testing CTS task registration...")
    try:
        spec = gym.spec("Isaac-Velocity-Flat-EVA02-CTS-v0")
        print(f"✓ Task registered: {spec.id}")
        print(f"  Entry point: {spec.entry_point}")
        print(f"  CTS Config: {spec.kwargs['rsl_rl_cfg_entry_point']}")
    except Exception as e:
        print(f"✗ Task registration failed: {e}")
        return False

    # Test CTS configuration import
    print("\n[2/3] Testing CTS configuration import...")
    try:
        from isaaclab_tasks.direct.eva02.agents.rsl_rl_cts_cfg import (
            EVA02FlatCTSRunnerCfg,
            EVA02CTSEncoderCfg,
        )
        print(f"✓ CTS configuration imported successfully")

        # Print encoder configurations
        cfg = EVA02FlatCTSRunnerCfg()
        print(f"\n  Privileged Encoder:")
        print(f"    - Latent dim: {cfg.privileged_encoder.latent_dim}")
        print(f"    - Hidden dims: {cfg.privileged_encoder.hidden_dims}")
        print(f"    - Use history: {cfg.privileged_encoder.use_history}")

        print(f"\n  Proprioceptive Encoder:")
        print(f"    - Latent dim: {cfg.proprioceptive_encoder.latent_dim}")
        print(f"    - Hidden dims: {cfg.proprioceptive_encoder.hidden_dims}")
        print(f"    - Use history: {cfg.proprioceptive_encoder.use_history}")
        print(f"    - History length: {cfg.proprioceptive_encoder.history_length}")

        print(f"\n  CTS Algorithm:")
        print(f"    - Teacher batch size: {cfg.algorithm.teacher_batch_size}")
        print(f"    - Student batch size: {cfg.algorithm.student_batch_size}")
        print(f"    - Reconstruction loss coef: {cfg.algorithm.reconstruction_loss_coef}")

    except Exception as e:
        print(f"✗ Configuration import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test environment creation with CTS
    print("\n[3/3] Testing environment creation with CTS...")
    try:
        from isaaclab_tasks.direct.eva02.eva02_env_cfg import EVA02FlatEnvCfg

        # Create small environment for testing
        env_cfg = EVA02FlatEnvCfg()
        env_cfg.scene.num_envs = 2
        env_cfg.scene.env_spacing = 2.0

        env = gym.make("Isaac-Velocity-Flat-EVA02-CTS-v0", cfg=env_cfg)

        print(f"✓ Environment created successfully")
        print(f"  Number of environments: {env.unwrapped.num_envs}")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        # Test reset
        obs, _ = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation shape: {obs['policy'].shape}")

        # Test step
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, rewards, terminated, truncated, info = env.step(actions)
        print(f"✓ Environment step successful")
        print(f"  Mean reward: {rewards.mean().item():.4f}")

        env.close()

    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✅ All CTS configuration tests passed!")
    print("=" * 80)
    print("\nYou can now start CTS training with:")
    print("  python scripts/reinforcement_learning/rsl_rl/train_eva02_cts.py")
    print("=" * 80)

    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        simulation_app.close()
