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
import asyncio

import nest_asyncio
nest_asyncio.apply()

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.examples")
enable_extension("omni.videoencoding") # need to have g_video_encoding_api in this before importing capture
enable_extension("omni.kit.viewport.utility")
# enable_extension("omni.kit.renderer.capture") # this capture the entire omniverse kit.exe window
# enable_extension("omni.kit.capture.viewport") # this caused the timeline to pause after the last frame captured
from franka_rope import IsaacUIUtils, VRUIUtils
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles



# IsaacUIUtils.setUp()
# VRUIUtils.setUp()

import hydra
import os
from glob import glob
import omegaconf
import wandb

async def eval_async(config):
    from .eval_push_etseed_util import EvalUtils
    log_dir = os.getcwd()
    await EvalUtils.eval_async(
            config=config,
            log_dir=log_dir,
            reduce_horizon_dim=config['cfg'].data.dataset.reduce_horizon_dim,
            cfg=config['cfg'].franka_rope,
            simulation_app=simulation_app
        )
        

import torch
from equibot.policies.utils.etseed.model.se3_transformer.equinet import SE3ManiNet_Invariant, SE3ManiNet_Equivariant_Separate
from equibot.policies.utils.etseed.utils.SE3diffusion_scheduler import DiffusionScheduler
import torch.nn as nn


# Initialize the model and optimizer
def init_model(device,config):
    noise_pred_net_in = SE3ManiNet_Invariant()
    noise_pred_net_eq = SE3ManiNet_Equivariant_Separate()
    nets = nn.ModuleDict({
        'invariant_pred_net': noise_pred_net_in,
        'equivariant_pred_net': noise_pred_net_eq
    }).to(device)
    checkpoint = torch.load(config["checkpoint_path"])
    nets.load_state_dict(checkpoint['model_state_dict'])
    nets.eval()
    return nets

@hydra.main(config_path="configs", config_name="etseed")
def main(cfg):

    config = {
        "seed": cfg.seed,
        "mode": cfg.mode,
        "pred_horizon": cfg.pred_horizon,
        "obs_horizon": cfg.obs_horizon,
        "action_horizon": cfg.action_horizon,
        "T_a": cfg.T_a,
        "batch_size": cfg.batch_size,
        "diffusion_steps": cfg.diffusion_steps,
        "diffusion_mode": cfg.diffusion_mode,
        "checkpoint_path": r"C:\Users\Shadow\project\equibot_for_franka_rope\logs\train\2025-03-02_17-32-19_etseed\ckpt00019.pth",  # replace with your checkpoint path
        "cfg":cfg
    }

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    nets = init_model(device,config)

    noise_scheduler = DiffusionScheduler(num_steps=config["diffusion_steps"],mode=config["diffusion_mode"],device=device)

    config['nets']=nets
    config['noise_scheduler']=noise_scheduler
    

    # IsaacUIUtils.setUp()
    # from viztracer import VizTracer
    # tracer = VizTracer(tracer_entries=99999999,output_file="my_trace.json")
    # tracer.start()
    # TODO BUG Windows fatal exception: access violation

    if cfg.use_wandb:
        wandb_config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=False
        )
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            tags=["eval"],
            name=cfg.prefix,
            settings=wandb.Settings(code_dir="."),
            config=wandb_config,
        )
    
    asyncio.ensure_future(eval_async(config))

    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()

if __name__ == "__main__":
    main()




