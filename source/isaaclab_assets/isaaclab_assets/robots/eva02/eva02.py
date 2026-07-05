# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the EVA02 quadruped robot.

Reference: https://github.com/unitreerobotics/unitree_rl_lab (adopted URDF loading pattern)
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

# Path to the EVA02 asset directory
EVA02_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "../../../../../assets/eva02")
EVA02_ASSETS_DIR = os.path.abspath(EVA02_ASSETS_DIR)
EVA02_URDF_PATH = f"{EVA02_ASSETS_DIR}/urdf/EVA02_full_relative.urdf"


@configclass
class EVA02UrdfFileCfg(sim_utils.UrdfFileCfg):
    """URDF file configuration for EVA02 robot.

    Follows unitree_rl_lab's UnitreeUrdfFileCfg pattern:
    - URDF-first loading (no pre-conversion to USD)
    - Proper solver iterations for stable simulation
    - Self-collisions enabled
    - Cylinders replaced with capsules for better collision detection
    - Zero joint drive gains (actuator handles control separately)
    """

    fix_base: bool = False
    merge_fixed_joints: bool = True
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = True
    joint_drive = sim_utils.UrdfFileCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0.0,
            damping=0.0,
        ),
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )


##
# Configuration
##

EVA02_CFG = ArticulationCfg(
    spawn=EVA02UrdfFileCfg(
        asset_path=EVA02_URDF_PATH,
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
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
        ),
    },
)
"""Configuration for the EVA02 quadruped robot."""
