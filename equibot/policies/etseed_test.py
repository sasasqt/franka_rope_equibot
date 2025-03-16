import os
import numpy as np
import torch
import time
import torch.nn as nn
import wandb
from equibot.policies.utils.etseed.utils.loss_utils import compute_loss
from .etseed_train import prepare_model_input
from tqdm.auto import tqdm

# env import
from equibot.policies.utils.etseed.model.se3_transformer.equinet import SE3ManiNet_Invariant_Separate, SE3ManiNet_Equivariant_Separate, SE3ManiNet_Fused_Separate, SE3ManiNet_Fused
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
        "action_horizon": cfg.action_horizon,
        "T_a": cfg.T_a,
        "batch_size": cfg.batch_size,
        "diffusion_steps": cfg.diffusion_steps,
        "diffusion_mode": cfg.diffusion_mode,
        "checkpoint_path": cfg.training.ckpt,
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
    noise_scheduler = DiffusionScheduler(num_steps=config["diffusion_steps"],mode=config["diffusion_mode"],device=device)
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
            "diffusion_num_steps": noise_scheduler.num_steps,
            "diffusion_mode": noise_scheduler.mode,
            "diffusion_sigma_r": noise_scheduler.sigma_r,
            "diffusion_sigma_t": noise_scheduler.sigma_t
        }
    )
    test_losses = []
    global g_step
    g_step=-1
    with tqdm(valid_loader, desc='Test Batch') as tepoch:
        for nbatch in tepoch:
            g_step+=1
            loss_cpu = test_batch(nets, noise_scheduler, nbatch, device,config)
            test_losses.append(loss_cpu)
            tepoch.set_postfix(loss=loss_cpu)
    avg_test_loss = np.mean(test_losses)
    wandb.log({'test_loss': avg_test_loss},step=g_step)
    print(f"Test Done! Average Test Loss: {avg_test_loss}")


def init_model(device,config):
    noise_pred_net_in = SE3ManiNet_Fused()
    noise_pred_net_eq = SE3ManiNet_Fused()
    nets = nn.ModuleDict({
        'invariant_pred_net': noise_pred_net_in,
        'equivariant_pred_net': noise_pred_net_eq
    }).to(device)
    checkpoint = torch.load(config["checkpoint_path"])
    nets.load_state_dict(checkpoint['model_state_dict'])
    nets.eval()
    return nets


# test a single batch of data
def test_batch(nets, noise_scheduler, nbatch, device,config):
    nets.eval()
    global g_step

    with torch.no_grad():
        nxyz = nbatch['pc'][:, :, :, :3].to(device) # [B,Ho,num_pts,3]
        tgt_nxyz = nbatch['pc'][:, :, :, 3:6].to(device)
        naction = nbatch['action'].to(device) # [B,Ho,4by4]
        #neefpose = nbatch['eef_pos'].to(device)
        bz = nxyz.shape[0]
        naction = naction.view(naction.size(0),naction.size(1),4,4) # naction: torch.Size([B, Ho, 4, 4])
        num_point = nxyz.shape[2]
        nxyz = nxyz.view(bz, -1, 3) # [B,Ho*num_pts,3]
        tgt_nxyz = tgt_nxyz.view(bz, -1, 3) # [B,Ho*num_pts,3]
        
        H_t_noise = torch.eye(4)[None].expand(bz,config["pred_horizon"], -1, -1).to(device) # H_T: [B,Ho,4,4]
        if os.name == 'nt': # mock actions on windows 
            #actions=prepare_model_output(H_t_noise)
            return H_t_noise
        
        for denoise_idx in range(noise_scheduler.num_steps - 1, -1, -1):
            g_step+=1
            k = torch.zeros((bz,)).long().to(device)
            k = k.repeat(config["T_a"], 1).transpose(0, 1).reshape(-1)
            k[:] = denoise_idx
            model_input = prepare_model_input(nxyz, tgt_nxyz, H_t_noise, k, num_point,config)
            
            if (denoise_idx == 0): 
                test_equiv = True 
            else: 
                test_equiv = False
            
            with torch.no_grad():
                if (test_equiv):
                    pred = nets["equivariant_pred_net"](model_input)
                else:
                    pred = nets["invariant_pred_net"](model_input)
    
            noise_pred = pred
            H_t_noise, H_0 = noise_scheduler.denoise(
                model_output = noise_pred,
                timestep = k,
                sample = H_t_noise,
                device = device
            )
        
            loss, dist_R, dist_T = compute_loss(H_0.view(-1,4,4), naction.view(-1,4,4))
            # print("loss: ", loss)
            loss_cpu = loss.item()
            if test_equiv:
                dist_equiv_r = dist_R
                dist_equiv_t = dist_T
            else:
                dist_invar_r = dist_R
                dist_invar_t = dist_T
            wandb.log({"test_dist_R": dist_R},step=g_step)
            wandb.log({"test_dist_T": dist_T},step=g_step)
            wandb.log({"test_loss_cpu": loss_cpu},step=g_step)
            if test_equiv:
                wandb.log({"test_dist_R_eq": dist_equiv_r},step=g_step)
                wandb.log({"test_dist_T_eq": dist_equiv_t},step=g_step)
            else:
                wandb.log({"test_dist_R_in": dist_invar_r},step=g_step)
                wandb.log({"test_dist_T_in": dist_invar_t},step=g_step)
    return loss_cpu




if __name__ == "__main__":
    main()