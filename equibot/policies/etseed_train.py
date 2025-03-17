import os
import numpy as np
import torch
import time
import torch.nn as nn
import wandb
from equibot.policies.utils.etseed.utils.loss_utils import compute_loss
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# env import
from equibot.policies.utils.etseed.model.se3_transformer.equinet import SE3ManiNet_Invariant_Separate, SE3ManiNet_Equivariant_Separate, SE3ManiNet_Fused_Separate, SE3ManiNet_Fused
from equibot.policies.utils.etseed.utils.SE3diffusion_scheduler import DiffusionScheduler

import hydra
import logging
import omegaconf
from equibot.policies.utils.misc import get_dataset

import kornia

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
        "num_epochs": cfg.num_epochs,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "betas": cfg.betas,
        "eps": cfg.eps,
        "equiv_frac": cfg.equiv_frac,
        "save_freq": cfg.save_freq,
        "diffusion_steps": cfg.diffusion_steps,
        "diffusion_mode": cfg.diffusion_mode,
    }


    assert config["mode"] == "train"
    np.random.seed(config["seed"])

    logging.basicConfig(level=logging.INFO)

    # initialize parameters
    batch_size = config["batch_size"]

    # setup logging
    log_dir = os.getcwd()
    num_workers = min(os.cpu_count(),cfg.data.dataset.num_workers)
    if os.name == 'nt': # if windows
        num_workers=0 
    # init dataloader
    train_dataset = get_dataset(cfg, "train")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False, # was True
        pin_memory=True,
    )
    config["num_training_steps"]=cfg.data.dataset.num_training_steps = (
        max(1,2 * len(train_dataset) // (batch_size))
    )

    valid_dataset = get_dataset(cfg, "train", valid=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=64,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False, # was True
        pin_memory=True,
    )

    checkpoint_dir = log_dir

    train_dataloader = train_loader
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        device = torch.device('cpu') # compile dgl w/ cuda in windows is as easy as compiling pytorch cuda from source:)
        # micromamba further complicates it by not introducing proper sys envs for cmakelists
    
    nets, optimizer, lr_scheduler = init_model_and_optimizer(device,config)
    noise_scheduler = DiffusionScheduler(num_steps=config["diffusion_steps"],mode=config["diffusion_mode"],device=device)
    
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        tags=["train"],
        name=cfg.prefix,
        settings=wandb.Settings(code_dir="."),
        config={
            "learning_rate": config["learning_rate"],
            "pred_horizon": config["pred_horizon"],
            "obs_horizon": config["obs_horizon"],
            "batch_size": config["batch_size"],
            "epochs": config["num_epochs"],
            "diffusion_num_steps": noise_scheduler.num_steps,
            "diffusion_mode": noise_scheduler.mode,
            "diffusion_sigma_r": noise_scheduler.sigma_r,
            "diffusion_sigma_t": noise_scheduler.sigma_t
        }
    )
    global g_step
    g_step=-1
    with tqdm(range(config["num_epochs"]), desc='Epoch', position=0) as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []
            with tqdm(train_dataloader, desc='Batch', position=1, leave=False) as tepoch:
                for nbatch in tepoch:
                    loss_cpu = train_batch(nets, optimizer, lr_scheduler, noise_scheduler, nbatch,epoch_idx, device,config=config)
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            wandb.log({'train_loss_avg': np.mean(epoch_loss), 'epoch': epoch_idx},step=g_step)
            
            if (epoch_idx + 1) % config["save_freq"] == 0 or epoch_idx == cfg["num_epochs"] - 1:
                checkpoint_path = os.path.join(checkpoint_dir, f'ckpt{epoch_idx:05d}.pth')
                torch.save({
                    'epoch': epoch_idx,
                    'model_state_dict': nets.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_cpu,
                    'lr_scheduler_state_dict': lr_scheduler.state_dict()
                }, checkpoint_path)
    print("Training Done!")




# Initialize the model and optimizer
def init_model_and_optimizer(device,config):
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.version.cuda)

    # noise_pred_net_in = SE3ManiNet_Fused_Separate()
    # noise_pred_net_eq = SE3ManiNet_Fused_Separate()
    noise_pred_net_in = SE3ManiNet_Fused()
    noise_pred_net_eq = SE3ManiNet_Fused()
    
    nets = nn.ModuleDict({
        'invariant_pred_net': noise_pred_net_in,
        'equivariant_pred_net': noise_pred_net_eq
    }).to(device)
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"],
        betas=config["betas"], 
        eps=config["eps"]
    )
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=config["num_training_steps"]
    )
    return nets, optimizer, lr_scheduler



# Prepare the input for the model
def prepare_model_input(nxyz, tgt_nxyz, noisy_actions, k, num_point,config):
    B = nxyz.shape[0]
    nxyz = nxyz.repeat(1,config["T_a"] // config["obs_horizon"], 1) #[B,Ho*num_pts,3]
    tgt_nxyz = tgt_nxyz.repeat(1,config["T_a"] // config["obs_horizon"], 1) #[B,Ho*num_pts,3]
    ori_indices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    selected_ori_actions = [noisy_actions[:, :, i, j] for i, j in ori_indices]

    trans_indices = [(0, 3), (1, 3), (2, 3)]
    selected_trans_actions = [noisy_actions[:, :, i, j] for i, j in trans_indices]
    noisy_ori_actions = torch.stack(selected_ori_actions, dim=-1).repeat_interleave(num_point,dim=1) # [B,Ho*num_pts,6]
    noisy_trans_actions = torch.stack(selected_trans_actions, dim=-1).repeat_interleave(num_point,dim=1) # [B,Ho*num_pts,3]
    # k: [B]        
    tensor_k = k.clone().detach().unsqueeze(-1).unsqueeze(-1).expand(-1,nxyz.shape[1], -1) # [B,Ho*num_pts,1]

    feature = torch.cat((tensor_k,noisy_ori_actions,noisy_trans_actions,tgt_nxyz), dim=-1)

    model_input = {
        'xyz': nxyz.to(dtype=torch.float32),
        'feature': feature.to(dtype=torch.float32)
    }
    assert model_input["xyz"].dtype == torch.float32
    assert model_input["feature"].dtype == torch.float32

    return model_input

# Prepare the output of the model
def prepare_model_output(actions):
    pass
    # B = actions.shape[0]
    # Ho = actions.shape[1]
    # actions4by4 = torch.zeros((B, Ho, 4, 4), dtype=actions.dtype, device=actions.device)
    # indices = [(0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 3), (2, 0), (2, 1), (2, 3)]
    # for i, (row, col) in enumerate(indices):
    #     actions4by4[:, :, row, col] = actions[:, :, i]
    # col1 = actions4by4[..., :3, 0]
    # col2 = actions4by4[..., :3, 1]
    # col3 = torch.cross(col1, col2, dim=-1)
    # actions4by4[..., :3, 2] = col3

    # translations = actions4by4[:, :, :3, 3]
    # rotations=actions4by4[:, :,:3, :3]
    # quaternions = kornia.geometry.conversions.rotation_matrix_to_quaternion(rotations)
    # zeros = torch.zeros(B, Ho, 1, device=actions.device)
    # stacked = torch.cat([zeros, translations, quaternions], dim=2)

    # return stacked


# Train a single batch of data
def train_batch(nets, optimizer, lr_scheduler, noise_scheduler, nbatch,epoch_idx, device,config):
    global g_step
    g_step+=1
    nets.train()
    nxyz = nbatch['pc'][:, :, :, :3].to(device) # [B,Ho,num_pts,3]
    tgt_nxyz = nbatch['pc'][:, :, :, 3:6].to(device)
    naction = nbatch['action'].to(device) # [B,Ho,4by4]
    #neefpose = nbatch['eef_pos'].to(device)
    bz = nxyz.shape[0]
    naction = naction.view(naction.size(0),naction.size(1),4,4) # naction: torch.Size([B, Ho, 4, 4])
    num_point = nxyz.shape[2]
    nxyz = nxyz.view(bz, -1, 3) # [B,Ho*num_pts,3]
    tgt_nxyz = tgt_nxyz.view(bz, -1, 3) # [B,Ho*num_pts,3]

    # k [B]
    # if epoch_idx==0:
    #     train_equiv = True
    #     k = torch.randint(0, noise_scheduler.num_steps, (bz,), device=device)
    # elif torch.rand(1) < config["equiv_frac"] or not "old_equiv" in globals():
    #     train_equiv = True
    #     k = torch.zeros((bz,)).long().to(device)
    # else:
    #     train_equiv = False
    #     k = torch.randint(1, noise_scheduler.num_steps, (bz,), device=device)
    # train_equiv = True
    k = torch.randint(0, noise_scheduler.num_steps, (bz,), device=device)

    noisy_actions, noise = noise_scheduler.add_noise(naction, k, device=device)
    model_input = prepare_model_input(nxyz, tgt_nxyz, noisy_actions, k, num_point,config)
    # if train_equiv:
    #     pred = nets["equivariant_pred_net"](model_input,num_point,Inv=False)
    # else:
    #     pred = nets["invariant_pred_net"](model_input, num_point,Inv=True)
    pred = nets["equivariant_pred_net"](model_input,num_point,Inv=False)

    noise_pred = pred
    # noise_pred: [B,Ho,4,4]
    # noise: [B,Ho,4,4]
    # # see eq 10 in DiffusionReg paper, (exp are applied to both sides)
    # interpolated, predicted=noise_scheduler.pre_compute_loss(
    #     H_0 = naction,
    #     timestep = k,
    #     H_t = noisy_actions,
    #     predicted=noise_pred,
    #     device = device)
    # loss, dist_r, dist_t = compute_loss(predicted.view(-1,4,4),(interpolated).view(noise.size(0)*noise.size(1),4,4))  

    # see algorithm 1, but no more naction @torch.inverse(noisy_actions)
    loss, dist_r, dist_t = compute_loss(noise_pred.view(-1,4,4),(naction ).view(noise.size(0)*noise.size(1),4,4))  


    # weighted=torch.tensor(max(1.0,1.0 + (5-epoch_idx)/5), device=loss.device)
    # wandb.log({"weighted": weighted},step=g_step)
    # loss=weighted*dist_r + (2.0-weighted)*dist_t

    # if train_equiv:
    #     dist_equiv_r = dist_r
    #     dist_equiv_t = dist_t
    # else:
    #     dist_invar_r = dist_r
    #     dist_invar_t = dist_t
    dist_equiv_r = dist_r
    dist_equiv_t = dist_t

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()
    loss_cpu = loss.item()
    wandb.log({"dist_R": dist_r},step=g_step)
    wandb.log({"dist_T": dist_t},step=g_step)
    wandb.log({"loss_cpu": loss_cpu},step=g_step)
    wandb.log({'learning_rate': optimizer.param_groups[0]['lr']},step=g_step)

    # if train_equiv:
    #     wandb.log({"dist_R_eq": dist_equiv_r})
    #     wandb.log({"dist_T_eq": dist_equiv_t})
    # else:
    #     wandb.log({"dist_R_in": dist_invar_r})
    #     wandb.log({"dist_T_in": dist_invar_t})

    # wandb.log({"dist_R_eq": dist_equiv_r})
    # wandb.log({"dist_T_eq": dist_equiv_t})
    return loss_cpu


if __name__ == "__main__":
    main()