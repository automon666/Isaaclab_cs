#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quick test script for EVA02 environment."""

from isaacsim import SimulationApp

# Launch simulation with better rendering settings
simulation_app = SimulationApp({
    "headless": False,
    "width": 1920,
    "height": 1080,
})

import gymnasium as gym
import torch

# Register EVA02 tasks
import isaaclab_tasks.direct.eva02  # noqa: F401

def main():
    """Test EVA02 environment."""

    # Import the environment configuration
    from isaaclab_tasks.direct.eva02.eva02_env_cfg import EVA02FlatEnvCfg

    # Create environment configuration with reduced number of envs for testing
    env_cfg = EVA02FlatEnvCfg()
    env_cfg.scene.num_envs = 2  # 减少到 2 个环境以节省 GPU 内存
    env_cfg.scene.env_spacing = 2.0
    env_cfg.viewer.eye = (4.0, 4.0, 2.5)  # Better camera position
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)  # Look at robot height

    # 禁用实时渲染以节省 GPU 内存
    env_cfg.sim.use_fabric = False

    # Create environment
    env = gym.make("Isaac-Velocity-Flat-EVA02-v0", cfg=env_cfg)

    print("=" * 80)
    print(f"Environment: {env.unwrapped.cfg.__class__.__name__}")
    print(f"Number of environments: {env.unwrapped.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("=" * 80)

    # Reset environment
    obs, _ = env.reset()
    print(f"\n✓ Environment reset successfully")
    print(f"  Observation shape: {obs['policy'].shape}")

    # Run for a few steps
    print(f"\nRunning simulation for 200 steps...")
    for i in range(200):
        # Random actions
        actions = 0.5 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 0.25
        obs, rewards, terminated, truncated, info = env.step(actions)

        if i % 40 == 0:
            print(f"  Step {i}: mean reward = {rewards.mean().item():.4f}")

    print(f"\n✓ EVA02 environment test completed successfully!")
    print("=" * 80)

    # Close environment
    env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
