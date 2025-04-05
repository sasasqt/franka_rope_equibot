import os
import numpy as np
import torch
import time
import torch.nn as nn
import wandb
from equibot.policies.utils.etseed.utils.loss_utils import compute_loss
from .etseed_train_sep2 import prepare_model_input1,prepare_model_input2
from tqdm.auto import tqdm

# env import
from equibot.policies.utils.etseed.model.se3_transformer.equinet import SE3ManiNet_Fused, SE3VisionNet_Hierarchical
from equibot.policies.utils.etseed.utils.SE3diffusion_scheduler import DiffusionScheduler

import hydra
import logging
from equibot.policies.utils.misc import get_dataset

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
        'sh_basis_compute_gradients':cfg.dev.sh_basis_compute_gradients,
        'rot_aggregation':cfg.dev.rot_aggregation,
        'trans_aggregation':cfg.dev.trans_aggregation,
        'ddpm_predict_noise':cfg.dev.ddpm_predict_noise,
        'no_noise':cfg.dev.no_noise,
        'early_return':cfg.dev.early_return,
        "sigma_r":cfg.sigma_r,
        "sigma_t": cfg.sigma_t,
        "checkpoint_path": cfg.training.ckpt,
        'low_memory':cfg.dev.low_memory,
        'se3':cfg.dev.se3,
        'unet':cfg.dev.unet,
        'unet_film':cfg.dev.unet_film,
        'Ho_in_B':cfg.dev.Ho_in_B,
        'bugfix':cfg.dev.bugfix,
        'arch':cfg.dev.arch,
    }


    assert config["mode"] == "eval"
    np.random.seed(config["seed"])

    logging.basicConfig(level=logging.INFO)

    # initialize parameters
    batch_size = config["batch_size"]

    # setup logging
    log_dir = os.getcwd()
    num_workers = min(os.cpu_count(),cfg.data.dataset.num_workers)

    if os.name == 'nt': # if windows
        num_workers=0 

    valid_dataset = get_dataset(cfg, "train", valid=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False, # was True
        pin_memory=True,
    )
    checkpoint_dir = log_dir

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        device = torch.device('cpu') # compile dgl w/ cuda in windows is as easy as compiling pytorch cuda from source:)
        # micromamba further complicates it by not introducing proper sys envs for cmakelists
    
    nets = init_model(device,config)

    if config['use_ddpm']:
        from diffusers import DDPMScheduler
        noise_scheduler = DDPMScheduler(num_train_timesteps=config["diffusion_steps"],beta_schedule=config['diffusion_mode'])
    else:
        noise_scheduler = DiffusionScheduler(num_steps=config["diffusion_steps"], sigma_r=config["sigma_r"],sigma_t=config["sigma_t"],mode=config["diffusion_mode"],device=device)

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        tags=["eval"],
        name=cfg.prefix,
        settings=wandb.Settings(code_dir="."),
        config={
            "pred_horizon": config["pred_horizon"],
            "obs_horizon": config["obs_horizon"],
            "batch_size": config["batch_size"],
            "diffusion_num_steps": config["diffusion_steps"],
            "diffusion_mode": config["diffusion_mode"],
            'use_ddpm': config["use_ddpm"],
            'k_option':config["k_option"],
            'diffusion_option':config["diffusion_option"],
            "diffusion_sigma_r": config["sigma_r"],
            "diffusion_sigma_t": config["sigma_t"],
        }
    )
    test_losses = []
    global g_step
    g_step=-1
    with tqdm(valid_loader, desc='Test Batch') as tepoch:
        for nbatch in tepoch:
            loss_cpu = test_batch(nets, noise_scheduler, nbatch, device,config)
            test_losses.append(loss_cpu)
            tepoch.set_postfix(loss=loss_cpu)
    avg_test_loss = np.mean(test_losses)
    wandb.log({'test_loss': avg_test_loss},step=g_step)
    print(f"Test Done! Average Test Loss: {avg_test_loss}")


def init_model(device,config):
    pointcloud_encoder = SE3VisionNet_Hierarchical(hierarchy_layers=config['pred_horizon*obs_horizon'],output_type_1_feat=3,config=config)
    if config['se3']==0:
        noise_pred_net=SE3ManiNet_Fused(k_neighbours=8,pred_horizon=config['pred_horizon'],config=config,no_tgt_nxyz=True)
    elif config['se3']==1:
        from equibot.policies.utils.etseed.model.se3_transformer.equinet import SE3ManiNet_ori_pos_sep
        noise_pred_net=SE3ManiNet_ori_pos_sep(k_neighbours=8,pred_horizon=config['pred_horizon'],config=config,no_tgt_nxyz=True)
    else:
        raise NotImplementedError(f"k_option {config['se3']} not implemented")
    
    unet=None
    if  config['unet']:
        from equibot.policies.utils.diffusion.conditional_unet1d import ConditionalUnet1D
        unet = ConditionalUnet1D(
            input_dim=9, # ori pos
            diffusion_step_embed_dim=config['pred_horizon*obs_horizon']*9, #hierarchy_layers*output_type_1_feat*3
            global_cond_dim=config['pred_horizon*obs_horizon']*9,
        )

    nets = nn.ModuleDict({
        'pointcloud_encoder': pointcloud_encoder,
        'equivariant_pred_net': noise_pred_net,
        'unet': unet,
    }).to(device)


    checkpoint = torch.load(config["checkpoint_path"])
    nets.load_state_dict(checkpoint['model_state_dict'])
    nets.eval()
    return nets


# test a single batch of data
def test_batch(nets, noise_scheduler, nbatch, device,config,isVisualEval=False):
    nets.eval()
    if 'g_step' not in globals():
        global g_step
        g_step=-1

    with torch.no_grad():
        nxyz = nbatch['pc'][:, :, :, :3].to(device) # [B,Ho,num_pts,3]
        tgt_nxyz = nbatch['pc'][:, :, :, 3:6].to(device)
        if not isVisualEval:
            naction = nbatch['action'].to(device) # [B,Ho,4by4]
        neefpose = nbatch['eef_pos'].to(device)
        bz = nxyz.shape[0]
        ho = nxyz.shape[1]

        if not isVisualEval:
            naction = naction.view(naction.size(0),naction.size(1),4,4) # naction: torch.Size([B, Ho, 4, 4])
        num_point = nxyz.shape[2]
        nxyz = nxyz.view(bz, -1, 3) # [B,Ho*num_pts,3]
        tgt_nxyz = tgt_nxyz.view(bz, -1, 3) # [B,Ho*num_pts,3]
        neefpose=neefpose.view(bz,ho,-1) # ([B, Ho, num_eef * (pose gripper action etc)])



        H_Identity = torch.eye(4)[None].expand(bz,config["pred_horizon"], -1, -1).to(device) # H_T: [B,Ho,4,4]
        k=torch.full((bz,), config['diffusion_steps'] - 1).long().to(device)

        if config['use_ddpm']:
            noise = torch.randn(H_Identity.shape, device=device)
            noise[:, :,:3, :3]=noise[:, :,:3, :3]*config["sigma_r"]
            noise[:, :, :3, 3]=noise[:, :, :3, 3]*config["sigma_t"]
            noisy_actions = noise_scheduler.add_noise(H_Identity, noise, k)
        else:
            noisy_actions, noise=noise_scheduler.add_noise(H_Identity, k, device=device)
        if os.name == 'nt': # mock actions on windows 
            #actions=prepare_model_output(noisy_actions)
            return noisy_actions

        if config['use_ddpm']:
            g_step+=1
            # ddpm, the huggingface diffuser way
            noise_scheduler.set_timesteps(num_inference_steps=config['diffusion_steps'],device=device)
            for k in noise_scheduler.timesteps: # shape [1]

                pc= prepare_model_input1(nxyz, tgt_nxyz)
                latent_pc=nets["pointcloud_encoder"](pc) # b,l,f (l:x*Hp; f:3x)

                num_point = config['pred_horizon']

                model_input = prepare_model_input2(latent_pc, neefpose, k, num_point,config)
                model_output = nets["equivariant_pred_net"](model_input,num_point)

                if config['unet']:
                    model_output= nets['unet'](model_output.reshape(model_output.shape[0],model_output.shape[1],9), k, global_cond=latent_pc.reshape(latent_pc.shape[0],-1))

                if not config['early_return']:
                    noisy_actions = noise_scheduler.step(model_output.view(model_output.shape[0],model_output.shape[1],4,4), k, noisy_actions).prev_sample      
                else:
                    noisy_actions = noise_scheduler.step(model_output.view(model_output.shape[0],model_output.shape[1],4,4), k, noisy_actions).pred_original_sample

                if not isVisualEval:
                    loss, dist_R, dist_T = compute_loss(noisy_actions.view(-1,4,4), naction.view(-1,4,4))

                    loss_cpu = loss.item()

                    wandb.log({"test_dist_R": dist_R},step=g_step)
                    wandb.log({"test_dist_T": dist_T},step=g_step)
                    wandb.log({"test_loss_cpu": loss_cpu},step=g_step)

                if config['early_return']:
                    break

            if isVisualEval:
                actions=noisy_actions
                return actions
            else:
                return loss_cpu
            
        else:
                                
            # predict action instead of noise might due to https://github.com/lucidrains/denoising-diffusion-pytorch/issues/58#issuecomment-2676085515
            # but why does the predicted action at denoise_idx=num_steps already good, if not the best action?
            for denoise_idx in range(config['diffusion_steps'] - 1, -1, -1):
                g_step+=1

                # Options
                if config['diffusion_option']==0 or config['diffusion_option']==1:
                    # 0: the default, predict the gt H0
                    # 1: predict relative transformation from Ht to H0
                    k=torch.full((bz,), denoise_idx).long().to(device)
                elif config['diffusion_option']==2:
                    # 2: no diffusion, no denoising
                    k = torch.zeros((bz,)).long().to(device)
                else:
                    raise NotImplementedError(f"diffusion_option {config['diffusion_option']} not implemented")
                                
                pc= prepare_model_input1(nxyz, tgt_nxyz)
                latent_pc=nets["pointcloud_encoder"](pc) # b,l,f (l:x*Hp; f:3x)

                num_point = config['pred_horizon']

                model_input = prepare_model_input2(latent_pc, neefpose, k, num_point,config)
                model_output = nets["equivariant_pred_net"](model_input,num_point)

                if config['unet']:
                    model_output=nets['unet'](model_output.reshape(model_output.shape[0],model_output.shape[1],9), k, global_cond=latent_pc.reshape(latent_pc.shape[0],-1))
                    #process_action 
                    raise NotImplementedError # TODO
                
                # TODO assert last 2 dim 4x4

                # Options
                if config['diffusion_option']==0:
                    # 0: the default, predict the gt H0
                    reconstructed_H_0 = model_output
                    noisy_actions = noise_scheduler.denoise(
                        reconstructed_H_0=reconstructed_H_0,
                        timestep = k,
                        sample = noisy_actions,
                        device = device
                    )                
                elif config['diffusion_option']==1:
                    # 1: predict relative transformation from Ht to H0
                    reconstructed_H_0 = torch.einsum('bhij,bhjk->bhjk',model_output,noisy_actions)
                    noisy_actions = noise_scheduler.denoise(
                        reconstructed_H_0=reconstructed_H_0,
                        timestep = k,
                        sample = noisy_actions,
                        device = device
                    )
                elif config['diffusion_option']==2:
                    # 2: no diffusion, no denoising
                    reconstructed_H_0=model_output
                else:
                    raise NotImplementedError(f"diffusion_option {config['diffusion_option']} not implemented")

                assert not torch.any(torch.isnan(model_output)), model_output
                assert not torch.any(torch.isnan(noisy_actions)), noisy_actions

                if not isVisualEval:
                    loss, dist_R, dist_T = compute_loss(reconstructed_H_0.reshape(-1,4,4), naction.reshape(-1,4,4))
                    # print("loss: ", loss)
                    loss_cpu = loss.item()
                    # if test_equiv:
                    #     dist_equiv_r = dist_R
                    #     dist_equiv_t = dist_T
                    # else:
                    #     dist_invar_r = dist_R
                    #     dist_invar_t = dist_T

                    wandb.log({"test_dist_R": dist_R},step=g_step)
                    wandb.log({"test_dist_T": dist_T},step=g_step)
                    wandb.log({"test_loss_cpu": loss_cpu},step=g_step)
                    # if test_equiv:
                    #     wandb.log({"test_dist_R_eq": dist_equiv_r},step=g_step)
                    #     wandb.log({"test_dist_T_eq": dist_equiv_t},step=g_step)
                    # else:
                    #     wandb.log({"test_dist_R_in": dist_invar_r},step=g_step)
                    #     wandb.log({"test_dist_T_in": dist_invar_t},step=g_step)

                if config['early_return']:
                    break

            if isVisualEval:
                actions=noisy_actions
                return actions
            else:
                return loss_cpu




if __name__ == "__main__":
    main()