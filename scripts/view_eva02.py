#!/usr/bin/env python3
"""Script to load and view the EVA02 robot scene (matching training/play visuals).

Usage:
    # Flat terrain
    ./isaaclab.sh -p scripts/view_eva02.py

    # Rough terrain (matches Rough play)
    ./isaaclab.sh -p scripts/view_eva02.py --task rough
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View EVA02 robot scene.")
parser.add_argument("--task", type=str, default="flat", choices=["flat", "rough"],
                    help="Terrain type: flat or rough (default: flat)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = False
os.environ["HEADLESS"] = "0"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor
from isaaclab.sim import SimulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.eva02.eva02 import EVA02_CFG


def main():
    """Load EVA02 robot with the exact same scene as training/play."""
    # Choose config matching the task
    if args_cli.task == "rough":
        from isaaclab_tasks.direct.eva02.eva02_env_cfg import EVA02RoughEnvCfg
        env_cfg = EVA02RoughEnvCfg()
        print("[INFO] Loading ROUGH terrain scene (matches rough play).")
    else:
        from isaaclab_tasks.direct.eva02.eva02_env_cfg import EVA02FlatEnvCfg
        env_cfg = EVA02FlatEnvCfg()
        print("[INFO] Loading FLAT terrain scene (matches flat play).")

    # Use same simulation settings as training
    env_cfg.scene.num_envs = 1

    # SimulationContext MUST be created before InteractiveScene
    sim_cfg = SimulationCfg(dt=1/200, render_interval=4)
    sim_ctx = sim_utils.SimulationContext(sim_cfg)

    # Build scene exactly like the environment does
    scene = InteractiveScene(env_cfg.scene)
    robot_cfg = EVA02_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot = Articulation(robot_cfg)
    scene.articulations["robot"] = robot

    # Add terrain (same as _setup_scene)
    env_cfg.terrain.num_envs = 1
    env_cfg.terrain.env_spacing = 4.0
    terrain = env_cfg.terrain.class_type(env_cfg.terrain)

    # Clone environment
    scene.clone_environments(copy_from_source=False)

    # Same DomeLight as play
    sky_light_cfg = sim_utils.DomeLightCfg(
        intensity=750.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    )
    sky_light_cfg.func("/World/skyLight", sky_light_cfg)

    sim_ctx.reset()

    print(f"[INFO] Scene loaded: 1 EVA02 robot + {'rough' if args_cli.task == 'rough' else 'flat'} terrain + HDR lighting")
    print("[INFO] Controls: Right-drag=rotate, Middle-drag=pan, Scroll=zoom")
    print("[INFO] Close viewer window to exit.")

    while simulation_app.is_running():
        sim_ctx.step()
        scene.write_data_to_sim()
        scene.update(sim_cfg.dt)

    sim_ctx.__class__.clear_instance()


if __name__ == "__main__":
    main()
    simulation_app.close()
