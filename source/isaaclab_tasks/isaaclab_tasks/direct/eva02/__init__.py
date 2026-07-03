# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EVA02 quadruped locomotion environment."""

import gymnasium as gym

from . import agents
from .eva02_env import EVA02Env
from .eva02_env_cfg import EVA02FlatEnvCfg, EVA02RoughEnvCfg

##
# Register Gym environments
##

gym.register(
    id="Isaac-Velocity-Flat-EVA02-v0",
    entry_point="isaaclab_tasks.direct.eva02:EVA02Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": EVA02FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:EVA02FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-EVA02-v0",
    entry_point="isaaclab_tasks.direct.eva02:EVA02Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": EVA02RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:EVA02RoughPPORunnerCfg",
    },
)

# CTS (Concurrent Teacher-Student) tasks
gym.register(
    id="Isaac-Velocity-Flat-EVA02-CTS-v0",
    entry_point="isaaclab_tasks.direct.eva02:EVA02Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": EVA02FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cts_cfg:EVA02FlatCTSRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-EVA02-CTS-v0",
    entry_point="isaaclab_tasks.direct.eva02:EVA02Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": EVA02RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cts_cfg:EVA02RoughCTSRunnerCfg",
    },
)
