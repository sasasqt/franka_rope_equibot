# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# NOTE: Import here your extension examples to be propagated to ISAAC SIM Extensions startup
from equibot.policies.utils.misc import get_agent
# from equibot.policies.utils.diffusion.conditional_unet1d import ConditionalUnet1D <- this (in get_agent) caused BUG Windows fatal exception: access violation, if imported after simulationApp

# [Warning] [omni.isaac.kit.simulation_app] Modules: ['omni.kit_app'] were loaded before SimulationApp was started and might not be loaded correctly.
# [Warning] [omni.isaac.kit.simulation_app] Please check to make sure no extra omniverse or pxr modules are imported before the call to SimulationApp(...)
# not my fault?!
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import nest_asyncio
nest_asyncio.apply()

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.examples")
from franka_pusht import IsaacUIUtils, VRUIUtils
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles




import hydra
import os
from glob import glob

@hydra.main(config_path="equibot/policies/configs", config_name="franka_base")
def main(cfg):
    # IsaacUIUtils.setUp(cfg.franka_rope)
    VRUIUtils.setUp()
    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()

if __name__ == "__main__":
    main()



















