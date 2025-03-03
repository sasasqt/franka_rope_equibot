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
from equibot.policies.utils.etseed.model.se3_transformer.equinet import SE3ManiNet_Invariant, SE3ManiNet_Equivariant_Separate
from equibot.policies.utils.etseed.utils.SE3diffusion_scheduler import DiffusionScheduler

import hydra
import logging
import omegaconf
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
        max(1,cfg.num_epochs * len(train_dataset) // (batch_size * cfg.diffusion_steps))
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
    with tqdm(range(config["num_epochs"]), desc='Epoch', position=0) as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []
            with tqdm(train_dataloader, desc='Batch', position=1, leave=False) as tepoch:
                for nbatch in tepoch:
                    loss_cpu = train_batch(nets, optimizer, lr_scheduler, noise_scheduler, nbatch, device,config=config)
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({'learning_rate': current_lr})
            wandb.log({'train_loss_avg': np.mean(epoch_loss), 'epoch': epoch_idx})
            
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

    noise_pred_net_in = SE3ManiNet_Invariant()
    noise_pred_net_eq = SE3ManiNet_Equivariant_Separate()
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
    nxyz = nxyz.repeat(config["T_a"] // config["obs_horizon"], 1, 1)
    tgt_nxyz = tgt_nxyz.repeat(config["T_a"] // config["obs_horizon"], 1, 1)
    indices = [(0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 3), (2, 0), (2, 1), (2, 3)]
    selected_elements_action = [noisy_actions[:, :, i, j] for i, j in indices]
    noisy_actions = torch.stack(selected_elements_action, dim=-1).reshape(-1, 9).unsqueeze(1).expand(-1, num_point, -1)
    tensor_k = k.clone().detach().unsqueeze(1).unsqueeze(2).expand(-1, num_point, -1)
    feature = torch.cat((nxyz,tgt_nxyz, noisy_actions, tensor_k), dim=-1)
    model_input = {
        'xyz': nxyz,
        'feature': feature
    }
    return model_input

# Prepare the output of the model
def prepare_model_output(actions):
    B = actions.shape[0]
    Ho = actions.shape[1]
    actions4by4 = torch.zeros((B, Ho, 4, 4), dtype=actions.dtype, device=actions.device)
    indices = [(0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 3), (2, 0), (2, 1), (2, 3)]
    for i, (row, col) in enumerate(indices):
        actions4by4[:, :, row, col] = actions[:, :, i]
    col1 = actions4by4[..., :3, 0]
    col2 = actions4by4[..., :3, 1]
    col3 = torch.cross(col1, col2, dim=-1)
    actions4by4[..., :3, 2] = col3
    return actions4by4


# Train a single batch of data
def train_batch(nets, optimizer, lr_scheduler, noise_scheduler, nbatch, device,config,isTrain=True):
    nets.train(isTrain)
    nxyz = nbatch['pc'][:, :, :, :3].to(device)
    tgt_nxyz = nbatch['pc'][:, :, :, 3:6].to(device)
    naction = nbatch['action'].to(device)
    #neefpose = nbatch['eef_pos'].to(device)
    bz = nxyz.shape[0]
    naction = naction.view(naction.size(0),naction.size(1),4,4) # naction: torch.Size([B, Ho, 4, 4])
    num_point = nxyz.shape[2]
    nxyz = nxyz.view(-1, num_point, 3)
    tgt_nxyz = tgt_nxyz.view(-1, num_point, 3)
    if not isTrain:
        H_t_noise = torch.eye(4)[None].expand(bz,config["action_horizon"], -1, -1).to(device) # H_T: [B,Ho,4,4]
        # k = torch.zeros((bz,)).long().to(device)
        # k = k.repeat(config["T_a"], 1).transpose(0, 1).reshape(-1) # k: torch.Size([B*Ho])                                                                                                                                                                                                                      | 0/7 [00:00<?, ?it/s]
        for denoise_idx in range(noise_scheduler.num_steps - 1, -1, -1):
            k = torch.zeros((bz,)).long().to(device)
            k = k.repeat(config["T_a"], 1).transpose(0, 1).reshape(-1)
            k[:] = denoise_idx
            model_input = prepare_model_input(nxyz, tgt_nxyz, noisy_actions, k, num_point,config)
            
            if (denoise_idx == 0): 
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
        actions=prepare_model_output(H_t_noise)
        return actions
    
    if torch.rand(1) < config["equiv_frac"]:
        train_equiv = True
        k = torch.zeros((bz,)).long().to(device)
    else:
        train_equiv = False
        k = torch.randint(1, noise_scheduler.num_steps, (bz,), device=device)
    k = k.repeat(config["T_a"], 1).transpose(0, 1).reshape(-1) # k: torch.Size([B*Ho])                                                                                                                                                                                                                      | 0/7 [00:00<?, ?it/s]
    noisy_actions, noise = noise_scheduler.add_noise(naction, k, device=device)
    model_input = prepare_model_input(nxyz, tgt_nxyz, noisy_actions, k, num_point,config)
    if train_equiv:
        pred = nets["equivariant_pred_net"](model_input)
    else:
        pred = nets["invariant_pred_net"](model_input)
    noise_pred = pred
    # noise_pred: [B*Ho,4,4]
    # noise: [B,Ho,4,4]
    loss, dist_r, dist_t = compute_loss(noise_pred, noise.view(noise.size(0)*noise.size(1),4,4))    
    if train_equiv:
        dist_equiv_r = dist_r
        dist_equiv_t = dist_t
    else:
        dist_invar_r = dist_r
        dist_invar_t = dist_t
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()
    loss_cpu = loss.item()
    wandb.log({"dist_R": dist_r})
    wandb.log({"dist_T": dist_t})
    wandb.log({"loss_cpu": loss_cpu})
    if train_equiv:
        wandb.log({"dist_R_eq": dist_equiv_r})
        wandb.log({"dist_T_eq": dist_equiv_t})
    else:
        wandb.log({"dist_R_in": dist_invar_r})
        wandb.log({"dist_T_in": dist_invar_t})
    return loss_cpu


if __name__ == "__main__":
    main()