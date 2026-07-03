# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CTS (Concurrent Teacher-Student) configuration for EVA02 locomotion.

This configuration is used with the existing train_cts.py script:
    python scripts/reinforcement_learning/rsl_rl/train_cts.py --task Isaac-Velocity-Flat-EVA02-v0
"""

# Note: This file is intentionally minimal because train_cts.py creates the configuration
# dynamically from command-line arguments. The CTS runner expects a configuration dict,
# not a config class.

# EVA02 will use the default CTS configuration from train_cts.py with these task-specific notes:

# Observation spaces for EVA02:
# - policy: [base_lin_vel (3), base_ang_vel (3), projected_gravity (3),
#            commands (3), joint_pos (12), joint_vel (12), actions (12)] = 48 dims
# - privileged: Additional privileged observations (if any, e.g., terrain height map)

# The train_cts.py script will automatically:
# 1. Split environments between teacher and student based on --teacher_student_ratio
# 2. Create privileged_encoder for teacher (sees ["policy", "privileged"])
# 3. Create proprioceptive_encoder for student (sees ["policy"] with history)
# 4. Share actor/critic networks between teacher and student
# 5. Train student to reconstruct teacher's latent representation

# Example usage:
#   python scripts/reinforcement_learning/rsl_rl/train_cts.py \
#       --task Isaac-Velocity-Flat-EVA02-v0 \
#       --num_envs 1536 \
#       --teacher_student_ratio 2.0 \
#       --max_iterations 2000 \
#       --headless
