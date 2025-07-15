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
# from franka_rope import IsaacUIUtils, VRUIUtils
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles



# IsaacUIUtils.setUp()
# VRUIUtils.setUp()

import hydra
import os
from glob import glob
import omegaconf
import wandb

async def eval_async(config):
    from .eval_toy_etseed_util import EvalUtils
    log_dir = os.getcwd()
    await EvalUtils.eval_async(
            config=config,
            log_dir=log_dir,
            reduce_horizon_dim=config['cfg'].data.dataset.reduce_horizon_dim,
            cfg=config['cfg'].franka_rope,
            simulation_app=simulation_app
        )
        

import torch
torch.set_grad_enabled(False)
from equibot.policies.utils.etseed.model.se3_transformer.equinet import  SE3ManiNet_Fused
from equibot.policies.utils.etseed.utils.SE3diffusion_scheduler import DiffusionScheduler
import torch.nn as nn


# # Initialize the model and optimizer
# def init_model(device,config):
#     noise_pred_net_in = SE3ManiNet_Fused(k_neighbours=config['k_neighbours*obs_horizon'],pred_horizon=config['pred_horizon'],config=config)
#     noise_pred_net_eq = SE3ManiNet_Fused(k_neighbours=config['k_neighbours*obs_horizon'],pred_horizon=config['pred_horizon'],config=config)
    
#     nets = nn.ModuleDict({
#         'invariant_pred_net': noise_pred_net_in,
#         'equivariant_pred_net': noise_pred_net_eq
#     }).to(device)
#     checkpoint = torch.load(config["checkpoint_path"])
#     nets.load_state_dict(checkpoint['model_state_dict'])
#     nets.eval()
#     return nets

@hydra.main(config_path="configs", config_name="etseed")
def main(cfg):
    config = {
        "seed": cfg.seed,
        "mode": cfg.mode,
        "pred_horizon": cfg.pred_horizon,
        "obs_horizon": cfg.obs_horizon,
        "pred_horizon*obs_horizon":cfg.pred_horizon*cfg.obs_horizon,
        "action_horizon": cfg.action_horizon,
        "T_a": cfg.T_a,
        "k_neighbours":cfg.k_neighbours,
        "k_neighbours*obs_horizon":cfg.k_neighbours*cfg.obs_horizon,
        "batch_size": cfg.batch_size,
        "diffusion_steps": cfg.diffusion_steps,
        "diffusion_mode": cfg.diffusion_mode,
        'use_ddpm': cfg.dev.use_ddpm,
        'k_option':cfg.dev.k_option,
        'diffusion_option':cfg.dev.diffusion_option,
        'early_return':cfg.dev.early_return,
        "sigma_r":cfg.sigma_r,
        "sigma_t": cfg.sigma_t,
        "checkpoint_path": cfg.training.ckpt,
        'sh_basis_compute_gradients':cfg.dev.sh_basis_compute_gradients,
        'rot_aggregation':cfg.dev.rot_aggregation,
        'trans_aggregation':cfg.dev.trans_aggregation,
        'ddpm_predict_noise':cfg.dev.ddpm_predict_noise,
        'predict_h0': cfg.dev.predict_h0,
        'no_noise':cfg.dev.no_noise,
        'low_memory':cfg.dev.low_memory,
        'se3':cfg.dev.se3,
        'unet':True,
        'unet_film':cfg.dev.unet_film,
        'unet_equivariance': cfg.dev.unet_equivariance,
        'Ho_in_B':cfg.dev.Ho_in_B,
        'diff': cfg.dev.diff,
        'bugfix':cfg.dev.bugfix,
        'sanity_check': cfg.dev.sanity_check,
        'testing': cfg.dev.testing,
        'pc_xyz_feat': cfg.dev.pc_xyz_feat,
        'eef_xyz_feat': cfg.dev.eef_xyz_feat,
        'latent_pc_as_feat': cfg.dev.latent_pc_as_feat,
        'num_degrees':cfg.dev.num_degrees,
        'num_channels':cfg.dev.num_channels,
        'num_heads':cfg.dev.num_heads,
        'channels_div':cfg.dev.channels_div,
        'global_cond':cfg.dev.global_cond,
        'local_cond':cfg.dev.local_cond,
        'arch':cfg.dev.arch,
        "cfg":cfg,
        "loadFromCkpt": False,
        'pc_inv':cfg.dev.pc_inv,
        'trans_norm':cfg.dev.trans_norm,
        'noisy_action_as_k':cfg.dev.noisy_action_as_k,
        'k_target': cfg.dev.k_target,
        'k_on_lie': cfg.dev.k_on_lie,
        'k1_type_1': cfg.dev.k1_type_1,
        'k2_type_1': cfg.dev.k2_type_1,
        'proper_se3_test': cfg.dev.proper_se3_test,
        'rel_gripper_pos': cfg.dev.rel_gripper_pos,
        'nonlinear': cfg.dev.nonlinear,
        'bias': cfg.dev.bias,
        'gate': cfg.dev.gate,
        'robomimic': cfg.robomimic,
        'num_layers': cfg.dev.num_layers,
        'snr': cfg.dev.snr,
        'adaptive_knn':cfg.dev.adaptive_knn,
        'amp': cfg.dev.amp,
    }

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        device = torch.device('cpu')

    if config['arch']==0:
        from equibot.policies.etseed_test import init_model
    # elif config['arch']==1:
    #     pass
    elif config['arch']==2:
        from equibot.policies.etseed_test_sep2 import init_model
        # config['early_return']=False
    elif config['arch']==3:
        from equibot.policies.etseed_test_sep_no_diffusion_no_gripper import init_model
    elif config['arch']==4:
        from equibot.policies.etseed_test_sep2_nounetdiffusion import init_model
        config['unet']=False
    else:
        raise NotImplementedError

    nets = init_model(device,config)

    if config['use_ddpm']:
        from diffusers import DDPMScheduler
        prediction_type='epsilon'
        if not config['ddpm_predict_noise']:
            prediction_type='sample'
        noise_scheduler = DDPMScheduler(num_train_timesteps=config["diffusion_steps"],beta_schedule=config['diffusion_mode'],prediction_type=prediction_type)
    else:
        noise_scheduler = DiffusionScheduler(num_steps=config["diffusion_steps"], sigma_r=config["sigma_r"],sigma_t=config["sigma_t"],mode=config["diffusion_mode"],device=device)

    from diffusers import DDPMScheduler
    prediction_type='epsilon'
    if not config['ddpm_predict_noise']:
        prediction_type='sample'
    gripper_noise_scheduler = DDPMScheduler(num_train_timesteps=config["diffusion_steps"],beta_schedule=config['diffusion_mode'],prediction_type=prediction_type)


    config['nets']=nets
    config['noise_scheduler']=noise_scheduler
    config['gripper_noise_scheduler']=gripper_noise_scheduler

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




