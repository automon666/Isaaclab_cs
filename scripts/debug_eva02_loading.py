#!/usr/bin/env python3
"""Debug script to check EVA02 robot loading."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug EVA02 loading")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable rendering
args_cli.headless = False
args_cli.enable_cameras = True

# Set ROS_PACKAGE_PATH
os.environ.setdefault("ROS_PACKAGE_PATH", "")
ros_paths = ["/home/tino66/Downloads"]
if os.environ["ROS_PACKAGE_PATH"]:
    ros_paths.append(os.environ["ROS_PACKAGE_PATH"])
os.environ["ROS_PACKAGE_PATH"] = ":".join(ros_paths)

print(f"[DEBUG] ROS_PACKAGE_PATH = {os.environ['ROS_PACKAGE_PATH']}")

# Launch
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks

print("\n[DEBUG] Creating environment...")
try:
    env = gym.make("Isaac-Velocity-Flat-EVA02-v0", num_envs=1)
    print("[DEBUG] Environment created successfully!")

    # Check if robot exists in scene
    print(f"[DEBUG] Scene entities: {list(env.scene.articulations.keys())}")
    if "robot" in env.scene.articulations:
        robot = env.scene.articulations["robot"]
        print(f"[DEBUG] Robot prim path: {robot.cfg.prim_path}")
        print(f"[DEBUG] Robot num instances: {robot.num_instances}")
        print(f"[DEBUG] Robot root state shape: {robot.data.root_state_w.shape}")

    # Reset to initialize
    print("\n[DEBUG] Resetting environment...")
    obs, _ = env.reset()
    print(f"[DEBUG] Observation shape: {obs['policy'].shape if 'policy' in obs else obs.shape}")

    # Keep simulation running
    print("\n[DEBUG] Environment loaded. Check Isaac Sim window for robot.")
    print("[DEBUG] Press Ctrl+C to exit")

    import time
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[DEBUG] Exiting...")

    env.close()

except Exception as e:
    print(f"[ERROR] Failed to create environment: {e}")
    import traceback
    traceback.print_exc()

simulation_app.close()
