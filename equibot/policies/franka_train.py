import os
import sys
import copy
import hydra
import torch
import wandb
import omegaconf
import numpy as np
import getpass as gt
from tqdm import tqdm
from glob import glob
from omegaconf import OmegaConf
from torch import nn

from equibot.policies.utils.misc import get_dataset, get_agent

import logging

@hydra.main(config_path="configs", config_name="fold_synthetic")
def main(cfg):
    assert cfg.mode == "train"
    np.random.seed(cfg.seed)

    logging.basicConfig(level=logging.INFO)

    # initialize parameters
    batch_size = cfg.training.batch_size

    # setup logging
    if cfg.use_wandb:
        wandb_config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=False
        )
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            tags=["train"],
            name=cfg.prefix,
            settings=wandb.Settings(code_dir="."),
            config=wandb_config,
        )
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
    cfg.data.dataset.num_training_steps = (
        cfg.training.num_epochs * len(train_dataset) // batch_size
    )

    valid_dataset = get_dataset(cfg, "train", valid=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False, # was True
        pin_memory=True,
    )


    # init agent
    agent = get_agent(cfg.agent.agent_name)(cfg)

    if cfg.training.ckpt is not None:
        agent.load_snapshot(cfg.training.ckpt)
        start_epoch_ix = int(cfg.training.ckpt.split("/")[-1].split(".")[0][4:])
    else:
        start_epoch_ix = 0

    # train loop
    global_step = 0
    best_n = {}
    for epoch_ix in tqdm(range(start_epoch_ix, cfg.training.num_epochs)):
        batch_ix = 0
        for batch in tqdm(train_loader, leave=False, desc="Batches"):
            # print(batch.keys()) # dict_keys(['pc', 'eef_pos', 'action'])
            # print(batch["pc"].shape) # torch.Size([1024, 4, 44, 3])
            # print(batch["action"].shape) # torch.Size([1024, 32, 7])
            # print(batch["eef_pos"].shape) # torch.Size([1024, 4, 1, 13])
            train_metrics = agent.update(
                batch, vis=False
            )
            if cfg.use_wandb:
                wandb.log(
                    {"train/" + k: v for k, v in train_metrics.items()},
                    step=global_step,
                )
                wandb.log({"epoch": epoch_ix}, step=global_step)
            del train_metrics
            global_step += 1
            batch_ix += 1
        if (
            epoch_ix % cfg.training.save_interval == 0
            or epoch_ix == cfg.training.num_epochs - 1
        ):
            save_path = os.path.join(log_dir, f"ckpt{epoch_ix:05d}.pth")
            num_ckpt_to_keep = 10000 # keep them all
            if len(list(glob(os.path.join(log_dir, "ckpt*.pth")))) > num_ckpt_to_keep:
                # remove old checkpoints
                for fn in list(sorted(glob(os.path.join(log_dir, "ckpt*.pth"))))[
                    :-num_ckpt_to_keep
                ]:
                    os.remove(fn)
            agent.save_snapshot(save_path)

            diff=0.0
            count=0

            for valid_batch in tqdm(valid_loader, leave=False, desc="validations"):
                count+=1
                obs = dict(
                    pc=valid_batch["pc"].permute(1, 0, 2, 3).numpy(), # (obs_horizon, B, N, 3)
                    state=valid_batch["eef_pos"].permute(1, 0, 2, 3).numpy()
                    ) #pc and eef_pos                    
                pred_ac=agent.act(
                    obs
                )
                diff+=nn.functional.mse_loss(torch.from_numpy(pred_ac),valid_batch["action"])
                # if count >= 10:
                #     break
            diff/=count
            wandb.log({"valid_loss":diff},step=global_step)

if __name__ == "__main__":
    main()
