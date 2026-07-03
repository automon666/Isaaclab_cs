#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script to verify EVA02 robot can be loaded in Isaac Lab."""

from isaacsim import SimulationApp

# Create simulation app
simulation_app = SimulationApp({"headless": False})

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class EVA02SceneCfg(InteractiveSceneCfg):
    """Configuration for EVA02 test scene."""

    # Ground plane
    ground = sim_utils.AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # EVA02 robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="/home/tino66/Downloads/EVA02_description/urdf/EVA02_description.urdf",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.42),
            joint_pos={
                ".*_hip_joint": 0.0,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": sim_utils.ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=33.5,
                velocity_limit=21.0,
                stiffness=25.0,
                damping=0.5,
            ),
        },
    )

    # Lighting
    light = sim_utils.AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


def main():
    """Main function to test EVA02 robot loading."""

    # Setup simulation
    sim_cfg = SimulationCfg(dt=0.005, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set camera view
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    # Create scene
    scene_cfg = EVA02SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    print("=" * 60)
    print("EVA02 Robot Test")
    print("=" * 60)
    print(f"Number of joints: {scene['robot'].num_joints}")
    print(f"Joint names: {scene['robot'].joint_names}")
    print(f"Number of bodies: {scene['robot'].num_bodies}")
    print("=" * 60)

    # Simulate for a few seconds
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running() and sim_time < 10.0:
        # Apply zero actions (robot should just stand)
        scene['robot'].set_joint_position_target(scene['robot'].data.default_joint_pos)

        # Write data to sim
        scene.write_data_to_sim()

        # Perform step
        sim.step()

        # Update buffers
        scene.update(sim_dt)

        # Update sim time
        sim_time += sim_dt
        count += 1

        if count % 100 == 0:
            print(f"[{sim_time:.2f}s] Robot base position: {scene['robot'].data.root_pos_w[0]}")

    print("\n✓ EVA02 robot loaded and simulated successfully!")

    # Close simulation
    simulation_app.close()


if __name__ == "__main__":
    main()
