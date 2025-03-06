import os
import numpy as np
import torch
import time
import torch.nn as nn
import wandb
from equibot.policies.utils.etseed.utils.loss_utils import compute_loss
from tqdm.auto import tqdm

# env import
from equibot.policies.utils.etseed.model.se3_transformer.equinet import SE3ManiNet_Invariant, SE3ManiNet_Equivariant_Separate
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
    with tqdm(valid_loader, desc='Test Batch') as tepoch:
        cnt=0
        for nbatch in tepoch:
            loss_cpu = test_batch(nets, noise_scheduler, nbatch, device,cnt,config)
            test_losses.append(loss_cpu)
            tepoch.set_postfix(loss=loss_cpu)
            cnt+=1
    avg_test_loss = np.mean(test_losses)
    wandb.log({'test_loss': avg_test_loss})
    print(f"Test Done! Average Test Loss: {avg_test_loss}")


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

# Prepare the input for the model
def prepare_model_input(nxyz, nrgb, noisy_actions, k, num_point,config):
    B = nxyz.shape[0]
    nxyz = nxyz.repeat(config["T_a"] // config["obs_horizon"], 1, 1)
    nrgb = nrgb.repeat(config["T_a"] // config["obs_horizon"], 1, 1)
    indices = [(0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 3), (2, 0), (2, 1), (2, 3)]
    selected_elements_action = [noisy_actions[:, :, i, j] for i, j in indices]
    noisy_actions = torch.stack(selected_elements_action, dim=-1).reshape(-1, 9).unsqueeze(1).expand(-1, num_point, -1)
    tensor_k = k.clone().detach().unsqueeze(1).unsqueeze(2).expand(-1, num_point, -1)
    feature = torch.cat((nrgb, noisy_actions, tensor_k), dim=-1)
    model_input = {
        'xyz': nxyz,
        'feature': feature
    }
    return model_input


# test a single batch of data
def test_batch(nets, noise_scheduler, nbatch, device,cnt,config):
    nets.eval()
    wandb.log({"diffusing": cnt})

    with torch.no_grad():
        nxyz = nbatch['pc'][:, :, :, :3].to(device)
        tgt_nxyz = nbatch['pc'][:, :, :, 3:6].to(device)
        naction = nbatch['action'].to(device)
        bz = nxyz.shape[0]
        naction = naction.view(naction.size(0),naction.size(1),4,4) # naction: torch.Size([B, Ho, 4, 4])
        num_point = nxyz.shape[2]
        nxyz = nxyz.view(-1, num_point, 3)
        tgt_nxyz = tgt_nxyz.view(-1, num_point, 3)
        
        H_t_noise = torch.eye(4)[None].expand(bz,config["pred_horizon"], -1, -1).to(device) # H_T: [B,Ho,4,4]
        if os.name == 'nt': # mock actions on windows 
            #actions=prepare_model_output(H_t_noise)
            return H_t_noise
        
        for denoise_idx in range(noise_scheduler.num_steps - 1, -1, -1):
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
            wandb.log({"test_dist_R": dist_R})
            wandb.log({"test_dist_T": dist_T})
            wandb.log({"test_loss_cpu": loss_cpu})
            if test_equiv:
                wandb.log({"test_dist_R_eq": dist_equiv_r})
                wandb.log({"test_dist_T_eq": dist_equiv_t})
            else:
                wandb.log({"test_dist_R_in": dist_invar_r})
                wandb.log({"test_dist_T_in": dist_invar_t})
    return loss_cpu




if __name__ == "__main__":
    main()