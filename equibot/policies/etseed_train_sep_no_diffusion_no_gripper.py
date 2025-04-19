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
from equibot.policies.utils.etseed.model.se3_transformer.equinet import SE3ManiNet_Fused, SE3VisionNet_Hierarchical
from equibot.policies.utils.etseed.utils.SE3diffusion_scheduler import DiffusionScheduler
from equibot.policies.utils.etseed.utils.group_utils import process_action #, orthogonalization

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
        "pred_horizon*obs_horizon":cfg.pred_horizon*cfg.obs_horizon,
        "action_horizon": cfg.action_horizon,
        "T_a": cfg.T_a,
        "k_neighbours":cfg.k_neighbours,
        "k_neighbours*obs_horizon":cfg.k_neighbours*cfg.obs_horizon,
        "batch_size": cfg.batch_size,
        "num_epochs": cfg.num_epochs,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "betas": cfg.betas,
        "eps": cfg.eps,
        "save_freq": cfg.save_freq,
        'k_option':3,
        'sh_basis_compute_gradients':cfg.dev.sh_basis_compute_gradients,
        'rot_aggregation':cfg.dev.rot_aggregation,
        'trans_aggregation':cfg.dev.trans_aggregation,
        'low_memory':cfg.dev.low_memory,
        'se3':cfg.dev.se3,
        'unet':True,
        'unet_film':cfg.dev.unet_film,
        'Ho_in_B':cfg.dev.Ho_in_B,
        'diff': cfg.dev.diff,
        'bugfix':cfg.dev.bugfix,
        'sanity_check': cfg.dev.sanity_check,
        'testing': cfg.dev.testing,
        # new
        'pc_xyz_feat': cfg.dev.pc_xyz_feat,
        'eef_xyz_feat': cfg.dev.eef_xyz_feat,
        'test_lr_scheduler':cfg.dev.test_lr_scheduler,
        'num_degrees':cfg.dev.num_degrees,
        'num_channels':cfg.dev.num_channels,
        'num_heads':cfg.dev.num_heads,
        'channels_div':cfg.dev.channels_div,
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
        1500 #max(1,2 * len(train_dataset) // (batch_size))
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

    noise_scheduler = None

    _config={
        "learning_rate": config["learning_rate"],
        "pred_horizon": config["pred_horizon"],
        "obs_horizon": config["obs_horizon"],
        "batch_size": config["batch_size"],
        "epochs": config["num_epochs"],
        'dev': cfg.dev,
    }

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        tags=["train"],
        name=cfg.prefix,
        settings=wandb.Settings(code_dir="."),
        config=_config
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
def init_model_and_optimizer(device,config,isNotTrain=False):
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.version.cuda)

    # TODO do not hardcode
    SE3VisionNet_Hierarchical_input_type_1_feat=2 if config['pc_xyz_feat'] else 1
    pointcloud_encoder = SE3VisionNet_Hierarchical(hierarchy_layers=config['pred_horizon*obs_horizon'],input_type_1_feat=SE3VisionNet_Hierarchical_input_type_1_feat,output_type_1_feat=3,config=config)
    if config['se3']==0:
        action_pred_net=SE3ManiNet_Fused(k_neighbours=8,pred_horizon=config['pred_horizon'],config=config,no_tgt_nxyz=True,eef_abs_position_as_node=config['testing']==1,eef_xyz_feat=config['eef_xyz_feat'] and config['testing'])
    elif config['se3']==1:
        from equibot.policies.utils.etseed.model.se3_transformer.equinet import SE3ManiNet_ori_pos_sep
        action_pred_net=SE3ManiNet_ori_pos_sep(k_neighbours=8,pred_horizon=config['pred_horizon'],config=config,no_tgt_nxyz=True,eef_abs_position_as_node=config['testing']==1,eef_xyz_feat=config['eef_xyz_feat'] and config['testing'])
    else:
        raise NotImplementedError(f"se_3 {config['se3']} not implemented")
    
    from equibot.policies.utils.diffusion.conditional_unet1d import ConditionalUnet1D
    unet = ConditionalUnet1D(
        input_dim=9, #cat ori, pos
        diffusion_step_embed_dim=config['pred_horizon*obs_horizon']*9, #hierarchy_layers*output_type_1_feat*3
        global_cond_dim=config['pred_horizon*obs_horizon']*9,
        cond_predict_scale=config['unet_film']
    )

    nets = nn.ModuleDict({
        'pointcloud_encoder': pointcloud_encoder,
        'equivariant_pred_net': action_pred_net,
        'unet': unet,
    }).to(device)

    if isNotTrain:
        checkpoint = torch.load(config["checkpoint_path"])
        nets.load_state_dict(checkpoint['model_state_dict'])
        nets.eval()
        return nets,None,None

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
def prepare_model_input1(nxyz, tgt_nxyz,diff=False,pc_xyz_feat=False):
    B = nxyz.shape[0]
    Ho_num_point=nxyz.shape[1]
    # nxyz[B,Ho*num_pts,3]
    feature=tgt_nxyz
    if diff:
        feature=tgt_nxyz-nxyz
    if pc_xyz_feat:
        # [B,Ho*num_pts,6]
        feature=torch.cat((nxyz, feature), dim=-1)  

    model_input = {
        'xyz': nxyz.to(device='cuda',dtype=torch.float32),
        'feature': feature.to(device='cuda',dtype=torch.float32)
    }
    assert model_input["xyz"].dtype == torch.float32
    assert model_input["feature"].dtype == torch.float32

    return model_input #,ref_output


# Prepare the input for the model
def prepare_model_input2(nxyz, neefpose, k, num_point,config):
    B = nxyz.shape[0]
    Ho_num_point=nxyz.shape[1]
    # nxyz is the latent pc, [B,Ho*num_pts,type_1_feat*3]
    neefpose=neefpose.repeat(1,num_point, 1)# neefpose ([B, Ho*num_point, num_eef * (pose gripper action etc = 13)])
    #   the order: 3d world position, 3d ori col1, 3d ori col2, 3d gravity, 1d gripper action

    if config['Ho_in_B']:
        nxyz=nxyz.reshape(B*Ho_num_point,-1).unsqueeze(1)
        neefpose=neefpose.reshape(B*Ho_num_point,-1).unsqueeze(1)

    right_eef_world_pos=neefpose[...,0:3]
    col1=neefpose[...,3:6]
    col2=neefpose[...,6:9]
    gravity=neefpose[...,9:12]
    gripper_pose=neefpose[...,12:13]

    feature = torch.cat((gripper_pose,right_eef_world_pos,col1,col2,gravity), dim=-1)

    model_input = {
        'xyz': nxyz.to(device='cuda',dtype=torch.float32),
        'feature': feature.to(device='cuda',dtype=torch.float32)
    }
    assert model_input["xyz"].dtype == torch.float32
    assert model_input["feature"].dtype == torch.float32

    return model_input


# Prepare the input for the model
def prepare_model_input3(nxyz,neefpose, k,num_point,config):
    B = nxyz.shape[0]
    Ho_num_point=nxyz.shape[1]
    # nxyz is the latent pc, [B,Ho*num_pts,type_1_feat*3]
    neefpose=neefpose.repeat(1,num_point, 1)# neefpose ([B, Ho*num_point, num_eef * (pose gripper action etc = 13)])

    if config['Ho_in_B']:
        neefpose=neefpose.reshape(B*Ho_num_point,-1).unsqueeze(1)
        k1=k1.reshape(B*Ho_num_point,-1).unsqueeze(1)
        k2=k2.reshape(B*Ho_num_point,-1).unsqueeze(1)
    nxyz=neefpose[...,0:3]
    col1=neefpose[...,3:6]
    col2=neefpose[...,6:9]
    gravity=neefpose[...,9:12]
    gripper_pose=neefpose[...,12:13]
    
    # Options
    if not config['eef_xyz_feat']:
        feature = torch.cat((gripper_pose,col1,col2,gravity), dim=-1)
    else:
        feature = torch.cat((gripper_pose,nxyz,col1,col2,gravity), dim=-1)
    
    model_input = {
        'xyz': nxyz.to(device='cuda',dtype=torch.float32),
        'feature': feature.to(device='cuda',dtype=torch.float32)
    }
    assert model_input["xyz"].dtype == torch.float32
    assert model_input["feature"].dtype == torch.float32

    return model_input 


# Train a single batch of data
def train_batch(nets, optimizer, lr_scheduler, noise_scheduler, nbatch,epoch_idx, device,config):
    global g_step
    g_step+=1
    nets.train()
    nxyz = nbatch['pc'][:, :, :, :3].to(device) # [B,Ho,num_pts,3]
    tgt_nxyz = nbatch['pc'][:, :, :, 3:6].to(device)
    naction = nbatch['action'].to(device) # [B,Hp,4by4]
    neefpose = nbatch['eef_pos'].to(device) # ([B, Ho, num_eef, pose gripper action etc])
    
    bz = nxyz.shape[0]
    ho = nxyz.shape[1]
    naction = naction.view(naction.size(0),naction.size(1),4,4) # naction: torch.Size([B, Hp, 4, 4])
    nxyz = nxyz.view(bz, -1, 3) # [B,Ho*num_pts,3]
    tgt_nxyz = tgt_nxyz.view(bz, -1, 3) # [B,Ho*num_pts,3]
    neefpose=neefpose.view(bz,ho,-1) # ([B, Ho, num_eef * (pose gripper action etc)])
    
    k = torch.zeros((bz,)).long().to(device)

    
    pc= prepare_model_input1(nxyz, tgt_nxyz,diff=config['diff'],pc_xyz_feat=config['pc_xyz_feat'])
    latent_pc=nets["pointcloud_encoder"](pc) # b,l,f (l:x*Hp; f:3x)

    num_point = config['pred_horizon']

    if config['testing']==0:
        model_input = prepare_model_input2(latent_pc, neefpose, k, num_point,config)
    elif config['testing']==1:
        model_input = prepare_model_input3(latent_pc,neefpose, k,num_point,config)

    if config['Ho_in_B']:
        num_point = config['pred_horizon*obs_horizon']

    return_raw=True

    model_output = nets["equivariant_pred_net"](model_input,num_point,return_raw=return_raw,Ho_in_B=config['Ho_in_B'])

    ori=model_output['ori']
    pos=model_output['pos']
    model_output=torch.cat((ori, pos), dim=-1) # [B,Hp,6+3]

    model_output= nets['unet'](model_output, k, global_cond=latent_pc.reshape(latent_pc.shape[0],-1))

    output_ori=model_output[...,0:6].reshape(-1,6)
    output_pos=model_output[...,6:9].reshape(-1,3)
    action=process_action(output_ori, output_pos,follow_rot_trans_convention=True).view(model_output.shape[0],-1,4,4)
    

    loss, dist_r, dist_t, dist_g = compute_loss(action.reshape(-1,4,4),(naction ).reshape(-1,4,4))  


    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()
    loss_cpu = loss.item()
    # if not config['use_ddpm']:
    wandb.log({"dist_R": dist_r},step=g_step)
    wandb.log({"dist_T": dist_t},step=g_step)
    if dist_g is not None:
        wandb.log({"dist_G": dist_g},step=g_step)
    wandb.log({"loss_cpu": loss_cpu},step=g_step)
    wandb.log({'learning_rate': optimizer.param_groups[0]['lr']},step=g_step)

    return loss_cpu


if __name__ == "__main__":
    main()