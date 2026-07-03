# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
CTS (Concurrent Teacher-Student) Training Example Configuration

This example shows how to configure an environment for CTS training.
The key requirement is to provide both privileged and proprioceptive observation groups.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass


@configclass
class CTSObservationsCfg:
    """Observation specifications for CTS training.

    This configuration defines two observation groups:
    1. "policy" (proprioceptive): Available to both teacher and student
    2. "privileged": Only available to teacher during training
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Proprioceptive observations that student can access.

        These observations should include only on-board sensor information:
        - Base angular velocity (IMU)
        - Projected gravity (IMU)
        - Commands
        - Joint positions
        - Joint velocities
        - Previous actions
        """

        def __post_init__(self):
            self.enable_corruption = False  # No noise for policy obs in sim
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObservationGroupCfg):
        """Privileged observations for teacher.

        These include everything in policy observations plus:
        - Base linear velocity (ground truth, not measurable in real world)
        - Terrain heights around the robot
        - Contact forces
        - External forces
        - Friction coefficients
        - etc.
        """

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


# Example: Adding observation terms
def configure_cts_observations(env_cfg: ManagerBasedRLEnvCfg):
    """Configure observations for CTS training.

    Args:
        env_cfg: Environment configuration to modify.

    Example:
        >>> from isaaclab_tasks.manager_based.locomotion.velocity import velocity_env_cfg
        >>> cfg = velocity_env_cfg.AnymalCRoughEnvCfg()
        >>> configure_cts_observations(cfg)
    """

    # Policy observations (proprioceptive - student accessible)
    env_cfg.observations.policy = ObservationGroupCfg(
        concatenate_terms=True,
        enable_corruption=False,
        terms={
            # Base angular velocity (from IMU)
            "base_ang_vel": ObservationTermCfg(
                func="base_ang_vel",
                noise=None,  # Or add realistic IMU noise
            ),
            # Projected gravity (from IMU)
            "projected_gravity": ObservationTermCfg(
                func="projected_gravity",
                noise=None,
            ),
            # Commands
            "velocity_commands": ObservationTermCfg(
                func="generated_commands",
                params={"command_name": "base_velocity"},
            ),
            # Joint positions
            "joint_pos": ObservationTermCfg(
                func="joint_pos_rel",
                noise=None,
            ),
            # Joint velocities
            "joint_vel": ObservationTermCfg(
                func="joint_vel_rel",
                noise=None,
            ),
            # Previous actions
            "actions": ObservationTermCfg(
                func="last_action",
            ),
        },
    )

    # Privileged observations (teacher only)
    env_cfg.observations.privileged = ObservationGroupCfg(
        concatenate_terms=True,
        enable_corruption=False,
        terms={
            # Include all policy observations
            "base_ang_vel": ObservationTermCfg(func="base_ang_vel"),
            "projected_gravity": ObservationTermCfg(func="projected_gravity"),
            "velocity_commands": ObservationTermCfg(
                func="generated_commands",
                params={"command_name": "base_velocity"},
            ),
            "joint_pos": ObservationTermCfg(func="joint_pos_rel"),
            "joint_vel": ObservationTermCfg(func="joint_vel_rel"),
            "actions": ObservationTermCfg(func="last_action"),

            # Additional privileged information
            "base_lin_vel": ObservationTermCfg(
                func="base_lin_vel",
                # This is privileged - ground truth velocity
            ),
            "terrain_heights": ObservationTermCfg(
                func="height_scan",
                params={"sensor_cfg": "height_scanner"},
                # Terrain information around robot
            ),
            "contact_forces": ObservationTermCfg(
                func="contact_forces",
                params={"sensor_cfg": "contact_forces"},
                # Ground contact forces
            ),
        },
    )

    return env_cfg


"""
Usage Example:
--------------

1. For a new environment:

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.wrappers.rsl_rl import configure_cts_observations

@configclass
class MyRobotEnvCfg(ManagerBasedRLEnvCfg):
    # ... other configurations ...
    pass

# Configure for CTS
cfg = MyRobotEnvCfg()
cfg = configure_cts_observations(cfg)
```

2. For existing velocity environments:

```python
from isaaclab_tasks.manager_based.locomotion.velocity import velocity_env_cfg

# Use Anymal-C as base
cfg = velocity_env_cfg.AnymalCRoughEnvCfg()

# Add CTS observation structure
cfg = configure_cts_observations(cfg)
```

3. Training with CTS:

```bash
python scripts/reinforcement_learning/rsl_rl/train_cts.py \\
    --task Isaac-Velocity-Rough-Anymal-C-v0 \\
    --num_envs 8192 \\
    --teacher_student_ratio 3.0 \\
    --max_iterations 5000 \\
    --headless
```

Key Points:
-----------
1. **Policy observations**: Only use on-board sensors (IMU, joint encoders)
2. **Privileged observations**: Include ground truth information not available in real world
3. **Observation groups**: Must define "policy" and "privileged" groups
4. **History buffer**: Student encoder uses the last 5 proprioceptive observations (configured in encoder)
5. **Latent representation**: Both encoders output 24-dim normalized latent vectors
"""
