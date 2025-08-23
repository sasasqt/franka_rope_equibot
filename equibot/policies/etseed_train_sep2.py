import os
import numpy as np
import torch
import time
import torch.nn as nn
import wandb
from equibot.policies.utils.etseed.utils.loss_utils import compute_loss
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import math

# env import
from equibot.policies.utils.etseed.model.se3_transformer.equinet import SE3ManiNet_Fused, SE3VisionNet_Hierarchical
from equibot.policies.utils.etseed.utils.SE3diffusion_scheduler import DiffusionScheduler
from equibot.policies.utils.etseed.utils.group_utils import process_action #, orthogonalization
from equibot.policies.utils.etseed.utils.se_math import se3

import hydra
import logging
import omegaconf
from equibot.policies.utils.misc import get_dataset
import glob

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
        "sigma_r":cfg.sigma_r,
        "sigma_t": cfg.sigma_t,
        #"equiv_frac": cfg.equiv_frac,
        "save_freq": cfg.save_freq,
        "max_ckpts": cfg.max_ckpts,
        "loadFromCkpt": cfg.loadFromCkpt,
        "lr_cycle_scale_factor": cfg.lr_cycle_scale_factor,
        "checkpoint_path": cfg.training.ckpt,
        "diffusion_steps": cfg.diffusion_steps,
        "diffusion_mode": cfg.diffusion_mode,
        'use_ddpm': cfg.dev.use_ddpm,
        'k_option':cfg.dev.k_option,
        'diffusion_option':cfg.dev.diffusion_option,
        'sh_basis_compute_gradients':cfg.dev.sh_basis_compute_gradients,
        'rot_aggregation':cfg.dev.rot_aggregation,
        'trans_aggregation':cfg.dev.trans_aggregation,
        'trans_norm':cfg.dev.trans_norm,
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
        'test_lr_scheduler':cfg.dev.test_lr_scheduler,
        'num_degrees':cfg.dev.num_degrees,
        'num_channels':cfg.dev.num_channels,
        'num_heads':cfg.dev.num_heads,
        'channels_div':cfg.dev.channels_div,
        'global_cond':cfg.dev.global_cond,
        'local_cond':cfg.dev.local_cond,
        'sign_mismatch':cfg.dev.sign_mismatch,
        'pc_inv':cfg.dev.pc_inv,
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
        'gate_weight': cfg.dev.gate_weight,
        'robomimic': cfg.robomimic,
        'num_layers': cfg.dev.num_layers,
        'snr': cfg.dev.snr,
        'adaptive_knn':cfg.dev.adaptive_knn,
        'amp': cfg.dev.amp,
    }


    assert config["mode"] == "train"

    set_seed(config['seed'],torch_deterministic=True)
    
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
        drop_last=True, # was True
        pin_memory=True,
    )

    config["num_training_steps"]=cfg.data.dataset.num_training_steps = (
        1500*200*config['lr_cycle_scale_factor']//config['batch_size']
        #1500 #max(1,2 * len(train_dataset) // (batch_size)) # config["num_epochs"] * len(train_dataset)
    )
    if config['test_lr_scheduler']:
        config["num_training_steps"]=cfg.data.dataset.num_training_steps = (
            config["num_epochs"] * (len(train_dataset)//config['batch_size'])
        )

    if config['loadFromCkpt'] and ".pth" not in config["checkpoint_path"]:
        ckpt = sorted(
            glob.glob(os.path.join(config["checkpoint_path"], 'ckpt*.pth')),
            key=os.path.getmtime
        )
        if ckpt ==[]:
            config['loadFromCkpt']=False
        else:
            config["checkpoint_path"]=ckpt[-1]
        
    # valid_dataset = get_dataset(cfg, "train", valid=True)
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset,
    #     batch_size=4,
    #     num_workers=num_workers,
    #     shuffle=True,
    #     drop_last=True, # was True
    #     pin_memory=False,
    # )

    checkpoint_dir = log_dir

    train_dataloader = train_loader
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        device = torch.device('cpu') # compile dgl w/ cuda in windows is as easy as compiling pytorch cuda from source:)
        # micromamba further complicates it by not introducing proper sys envs for cmakelists
    
    nets, optimizer, lr_scheduler = init_model_and_optimizer(device,config)
    if config['loadFromCkpt']:
        ckpt=torch.load(config["checkpoint_path"])
        config['epoch_offset']=ckpt['epoch'] 
        config['g_step_offset'] = ckpt['g_step']

    if config['use_ddpm']:

        from diffusers import DDPMScheduler
        prediction_type='epsilon'
        if not config['ddpm_predict_noise']:
            prediction_type='sample'

        noise_scheduler = DDPMScheduler(num_train_timesteps=config["diffusion_steps"], beta_schedule=config['diffusion_mode'],prediction_type=prediction_type)
        gripper_noise_scheduler = DDPMScheduler(num_train_timesteps=config["diffusion_steps"],beta_schedule=config['diffusion_mode'],prediction_type=prediction_type)


        # raise NotImplementedError
        # from diffusers import DDPMScheduler
        # prediction_type='epsilon'
        # if not config['ddpm_predict_noise']:
        #     prediction_type='sample'
        # noise_scheduler = DDPMScheduler(num_train_timesteps=config["diffusion_steps"],beta_schedule=config['diffusion_mode'],prediction_type=prediction_type)
    else:
        # noise_scheduler = DiffusionScheduler(num_steps=config["diffusion_steps"], sigma_r=config["sigma_r"],sigma_t=config["sigma_t"],mode=config["diffusion_mode"],device=device)

        noise_scheduler = DiffusionScheduler(num_steps=config["diffusion_steps"], sigma_r=config["sigma_r"],sigma_t=config["sigma_t"],mode=config["diffusion_mode"],device=device)

        from diffusers import DDPMScheduler
        prediction_type='epsilon'
        if not config['ddpm_predict_noise']:
            prediction_type='sample'
        gripper_noise_scheduler = DDPMScheduler(num_train_timesteps=config["diffusion_steps"],beta_schedule=config['diffusion_mode'],prediction_type=prediction_type)

    if config['use_ddpm']:
        # raise NotImplementedError 
        _config={
            "learning_rate": config["learning_rate"],
            "pred_horizon": config["pred_horizon"],
            "obs_horizon": config["obs_horizon"],
            "batch_size": config["batch_size"],
            "epochs": config["num_epochs"],
            "diffusion_num_steps": config["diffusion_steps"],
            "diffusion_mode": config["diffusion_mode"],
            'dev': cfg.dev,
        }
    else:
        _config={
            "learning_rate": config["learning_rate"],
            "pred_horizon": config["pred_horizon"],
            "obs_horizon": config["obs_horizon"],
            "batch_size": config["batch_size"],
            "epochs": config["num_epochs"],
            "diffusion_num_steps": noise_scheduler.num_steps,
            "diffusion_mode": noise_scheduler.mode,
            "diffusion_sigma_r": noise_scheduler.sigma_r,
            "diffusion_sigma_t": noise_scheduler.sigma_t,
            'dev': cfg.dev,
            'cfg':cfg,
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
    if config['loadFromCkpt']:
        g_step+=config['g_step_offset']
    with tqdm(range(config["num_epochs"]), desc='Epoch', position=0) as tglobal:        
        for epoch_idx in tglobal:
            epoch_loss = []
            with tqdm(train_dataloader, desc='Batch', position=1, leave=False) as tepoch:
                for nbatch in tepoch:
                    loss_cpu = train_batch(nets, optimizer, lr_scheduler, noise_scheduler, nbatch,epoch_idx, device,gripper_noise_scheduler,config=config)
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            epoch=epoch_idx
            if config['loadFromCkpt']:
                epoch+=config['epoch_offset']+1
            wandb.log({'train_loss_avg': np.mean(epoch_loss), 'epoch': epoch},step=g_step)
            
            if (epoch + 1) % config["save_freq"] == 0 or epoch_idx == cfg["num_epochs"] - 1:
                if (epoch + 1) % (20*config["save_freq"]) == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f'backup{epoch:05d}.pth')
                    torch.save({
                        'epoch': epoch,
                        'g_step': g_step,
                        'model_state_dict': nets.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_cpu,
                        'lr_scheduler_state_dict': lr_scheduler.state_dict()
                    }, checkpoint_path)

                checkpoint_path = os.path.join(checkpoint_dir, f'ckpt{epoch:05d}.pth')

                torch.save({
                    'epoch': epoch,
                    'g_step': g_step,
                    'model_state_dict': nets.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_cpu,
                    'lr_scheduler_state_dict': lr_scheduler.state_dict()
                }, checkpoint_path)

                ckpts = sorted(
                    glob.glob(os.path.join(checkpoint_dir, 'ckpt*.pth')),
                    key=os.path.getmtime
                )

                # Remove older checkpoints if exceeding max_ckpts
                while len(ckpts) > config['max_ckpts']:
                    os.remove(ckpts[0])
                    ckpts.pop(0)
    print("Training Done!")




# Initialize the model and optimizer
def init_model_and_optimizer(device,config,isNotTrain=False):
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.version.cuda)
    loadFromCkpt=config['loadFromCkpt']
    # TODO do not hardcode
    SE3VisionNet_Hierarchical_input_type_1_feat=2 if config['pc_xyz_feat'] else 1
    if not config['robomimic']:
        hierarchy_layers=config['pred_horizon*obs_horizon']
    else:
        hierarchy_layers=config['num_layers']
        # hierarchy_layers=config['pred_horizon*obs_horizon']

    pointcloud_encoder = SE3VisionNet_Hierarchical(hierarchy_layers=hierarchy_layers,input_type_1_feat=SE3VisionNet_Hierarchical_input_type_1_feat,output_type_1_feat=3,config=config,nonlinear=config['nonlinear'],input_type_1_feat_is_actually_type_0=config['robomimic'],amp=config['amp'])
    if config['se3']==0:
        # action_pred_net=SE3ManiNet_Fused(k_neighbours=8,pred_horizon=config['pred_horizon'],config=config,no_tgt_nxyz=True,eef_abs_position_as_node=config['testing']==1,eef_xyz_feat=config['eef_xyz_feat'] and config['testing'])
        action_pred_net=SE3ManiNet_Fused(k_neighbours=8,pred_horizon=config['pred_horizon'],config=config,no_tgt_nxyz=True,latent_pc_as_feat=config['latent_pc_as_feat'],nonlinear=config['nonlinear'],bias=config['bias'],gate=config['gate'],gravity=not config['robomimic'],num_layers=config['num_layers'],amp=config['amp'])
    elif config['se3']==1:
        action_pred_net=SE3ManiNet_Fused(k_neighbours=8,pred_horizon=config['pred_horizon'],config=config,no_tgt_nxyz=True,eef_abs_position_as_node=config['testing']==1,eef_xyz_feat=config['eef_xyz_feat'] and config['testing'],fused=False,nonlinear=config['nonlinear'],bias=config['bias'],gate=config['gate'],gravity=not config['robomimic'],num_layers=config['num_layers'],amp=config['amp'])
        # from equibot.policies.utils.etseed.model.se3_transformer.equinet import SE3ManiNet_ori_pos_sep
        # action_pred_net=SE3ManiNet_ori_pos_sep(k_neighbours=8,pred_horizon=config['pred_horizon'],config=config,no_tgt_nxyz=True,eef_abs_position_as_node=config['testing']==1,eef_xyz_feat=config['eef_xyz_feat'] and config['testing'])
    else:
        raise NotImplementedError(f"se3 {config['se3']} not implemented")
    
    unet=None
    assert config['unet']==True,'unet needed for diffusion'
    if config['unet']:
        from equibot.policies.utils.diffusion.conditional_unet1d import ConditionalUnet1D
        local_cond_dim=None
        global_cond_dim=None
        diffusion_step_embed_dim=None
        
        if config['local_cond']==0:
            local_cond_dim=None
        elif config['local_cond']==1:
            # proposed actions as local
            local_cond_dim=10 # last dim of action_pred_net
        else:
            raise NotImplementedError
        
        if config['global_cond']==0:
            # proposed actions as global
            global_cond_dim=config['pred_horizon']*10 # last two dim of action_pred_net
            diffusion_step_embed_dim=global_cond_dim
        elif config['global_cond']==1:
            # latent pc as global   
            global_cond_dim=config['pred_horizon']*2*9 # last two dim of pointcloud encoder
            diffusion_step_embed_dim=global_cond_dim
        elif config['global_cond']==2:
            global_cond_dim=None
            diffusion_step_embed_dim=config['pred_horizon']*10
        else: 
            raise NotImplementedError
        
        unet = ConditionalUnet1D(
            input_dim=10, #cat: ori, pos, gripper o/c
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            global_cond_dim=global_cond_dim,
            local_cond_dim=local_cond_dim,
            cond_predict_scale=config['unet_film'],
            equivariance=config['unet_equivariance'],
        )

    nets = nn.ModuleDict({
        'pointcloud_encoder': pointcloud_encoder,
        'equivariant_pred_net': action_pred_net,
        'unet': unet,
    }).to(device)

    # torch.nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
    num_parameters = sum(dict((p.data_ptr(), p.numel()) 
                    for p in nets.parameters()).values())
    logging.info(">>> !!number of unique trainable parameters!! <<<"+num_parameters.__str__())

    # keys_to_remove=[]
    # if config['unet_equivariance']:
    #     keys_to_remove = [
    #         "unet.final_conv.0.block.0.bias",
    #         "unet.final_conv.0.block.1.weight", 
    #         "unet.final_conv.0.block.1.bias"
    #     ]

    if isNotTrain:
        checkpoint = torch.load(config["checkpoint_path"])
        # for key in keys_to_remove:
        #     if key in checkpoint['model_state_dict']:
        #         del checkpoint['model_state_dict'][key]
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
        num_warmup_steps=500*200//config['batch_size'],
        num_training_steps=config["num_training_steps"],
    )

    if loadFromCkpt:
        checkpoint = torch.load(config["checkpoint_path"])
        # for key in keys_to_remove:
        #     if key in checkpoint['model_state_dict']:
        #         del checkpoint['model_state_dict'][key]
        nets.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict']) 
        lr_scheduler.last_epoch = checkpoint['g_step']

    return nets, optimizer, lr_scheduler



# Prepare the input for the model
def prepare_model_input1(nxyz, tgt_nxyz,diff=False,pc_xyz_feat=False,rgb=False):
    B = nxyz.shape[0]
    Ho_num_point=nxyz.shape[1]
    # nxyz[B,Ho*num_pts,3]
    feature=tgt_nxyz
    if not rgb:
        if diff:
            feature=tgt_nxyz-nxyz
        if pc_xyz_feat:
            # [B,Ho*num_pts,6]
            feature=torch.cat((feature, nxyz), dim=-1)  
    if rgb:
        if diff:
            pass
        if pc_xyz_feat:
            # [B,Ho*num_pts,6]
            feature=torch.cat((feature, nxyz), dim=-1)  
    model_input = {
        'xyz': nxyz.to(device='cuda',dtype=torch.float32),
        'feature': feature.to(device='cuda',dtype=torch.float32)
    }
    assert model_input["xyz"].dtype == torch.float32
    assert model_input["feature"].dtype == torch.float32
    return model_input #,ref_output


# Prepare the input for the model
def prepare_model_input2(nxyz, neefpose, k, num_point,config,mean=None):
    B = nxyz.shape[0]
    _dim=nxyz.shape[1]
    Ho_num_point=config['pred_horizon*obs_horizon']
    repeat_times = math.ceil(Ho_num_point / _dim)          # how many complete repeats you need
    nxyz = nxyz.repeat(1, repeat_times, 1)     # [b, a*repeat_times, x]
    nxyz = nxyz[:, :Ho_num_point, :]  
    # nxyz is the latent pc, [B,Ho*num_pts,type_1_feat*3]
    neefpose=neefpose.repeat(1,num_point, 1)# neefpose ([B, Ho*num_point, num_eef * (pose gripper action etc = 13)])
    #   the order: 3d world position, 3d ori col1, 3d ori col2, 3d gravity, 1d gripper action
    # ori_indices = [(0, 0), (1,0), (2,0), (0, 1), (1,1), (2,1)] # first two cols
    # selected_ori_actions = [noisy_actions[:, :, i, j] for i, j in ori_indices]
    # trans_indices = [(0, 3), (1, 3), (2, 3)]
    # selected_trans_actions = [noisy_actions[:, :, i, j] for i, j in trans_indices]
    # noisy_ori_actions = torch.stack(selected_ori_actions, dim=-1) #.repeat_interleave(num_point,dim=1) # [B,Hp,6]
    # noisy_trans_actions = torch.stack(selected_trans_actions, dim=-1)  #.repeat_interleave(num_point,dim=1) # [B,Hp,3]
    device='cuda'

    if not config['noisy_action_as_k']:
        vectors = torch.tensor([[1.0, 1.0, 1.0]] * B,device=nxyz.device)  # Shape: (B, 3)
        angles = k.clone().detach()*torch.pi/(1+config['diffusion_steps']+config['pred_horizon'])
        angles=angles.to(nxyz.device)
        axes = vectors / torch.norm(vectors, dim=1, keepdim=True)  # Shape: (batch_size, 3)

        # Rodrigues' formula
        K = torch.zeros(B, 3, 3,device=nxyz.device)
        K[:, 0, 1] = -axes[:, 2]
        K[:, 0, 2] = axes[:, 1]
        K[:, 1, 0] = axes[:, 2]
        K[:, 1, 2] = -axes[:, 0]
        K[:, 2, 0] = -axes[:, 1]
        K[:, 2, 1] = axes[:, 0]

        I = torch.eye(3,device=nxyz.device).unsqueeze(0).repeat(B, 1, 1)
        angles = angles.unsqueeze(-1).unsqueeze(-1)

        # Compute rotation matrices
        rotation_matrices = I + torch.sin(angles) * K + (1 - torch.cos(angles)) * torch.bmm(K, K)
        k1=rotation_matrices[:, :, 0] # [B,3] first col
        k2=rotation_matrices[:, :, 1] # [B,3] second col
        k1=k1.unsqueeze(1).expand(-1,nxyz.shape[1], -1) # [B,Ho*num_pts,3]
        k2=k2.unsqueeze(1).expand(-1,nxyz.shape[1], -1) # [B,Ho*num_pts,3]

        k1=k1.to(device)
        k2=k2.to(device)

        tensor_k = k.clone().detach().unsqueeze(-1).unsqueeze(-1).expand(-1,nxyz.shape[1], -1) # [B,Ho*num_pts,1]
    else:
        k1=k[...,0:3]
        k2=k[...,3:6]

    if config['Ho_in_B']:
        nxyz=nxyz.reshape(B*Ho_num_point,-1).unsqueeze(1)
        neefpose=neefpose.reshape(B*Ho_num_point,-1).unsqueeze(1)
        # tensor_k=tensor_k.reshape(B*Ho_num_point,-1).unsqueeze(1)
        k1=k1.reshape(B*Ho_num_point,-1).unsqueeze(1)
        k2=k2.reshape(B*Ho_num_point,-1).unsqueeze(1)

    right_eef_world_pos=neefpose[...,0:3]
    col1=neefpose[...,3:6]
    col2=neefpose[...,6:9]
    gravity=neefpose[...,9:12]
    gripper_pose=neefpose[...,12:13]

    if config['proper_se3_test']:
        new_shape = nxyz.shape[:-1] + (3, 3)
        t_reshaped = nxyz.view(new_shape)
        pred_mean = t_reshaped.mean(dim=-2)

        col1=col1+pred_mean
        col2=col2+pred_mean
        gravity=gravity+pred_mean

    if config['rel_gripper_pos']:
        if not config['Ho_in_B']:
            right_eef_world_pos=right_eef_world_pos-mean.unsqueeze(1) 
        else:
            right_eef_world_pos=right_eef_world_pos-mean.unsqueeze(1).expand(-1,Ho_num_point,-1).reshape(B*Ho_num_point,-1).unsqueeze(1) 
            
    # # Options
    # if config['k_option']==0:
    #     # # 0 diffusion steps as type 0 scalar
    #     # num_fib_in = [2,4] # 14 in total, 2 type0: k; binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity
    #     # k: [B]        
    #     feature = torch.cat((tensor_k,gripper_pose,right_eef_world_pos,col1,col2,gravity), dim=-1)
    # elif config['k_option']==1:
    #     # # 1 diffusion steps as type 0 rotation
    #     # num_fib_in = [7,4] # 19 in total, 7 type0: k1,k2; binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity
    #     feature = torch.cat((k1,k2,gripper_pose,right_eef_world_pos,col1,col2,gravity), dim=-1)
    # elif config['k_option']==2:
    #     # # 2 diffusion steps as type 1 rotation
    #     # num_fib_in = [1,6] # 19 in total, 1 type0: binary gripper_action 6 type1: k1,k2; eef_abs_position, eef_abs_rotation (2cols); gravity
    #     feature = torch.cat((gripper_pose,k1,k2,right_eef_world_pos,col1,col2,gravity), dim=-1)
    # elif config['k_option']==3:
    #     # no k
    #     feature = torch.cat((gripper_pose,right_eef_world_pos,col1,col2,gravity), dim=-1)
    # else:
    #     raise NotImplementedError(f"k_option {config['k_option']} not implemented")

    # ref_output=torch.cat((noisy_ori_actions,noisy_trans_actions), dim=-1)

    latent_pc_as_feat=config['latent_pc_as_feat']
    if config['k_option']==1:
            # # 1 diffusion steps as type 0 rotation
            # num_fib_in = [7,4] # 19 in total, 7 type0: k1,k2; binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity
            if not config['k1_type_1'] and not config['k2_type_1']:
                if latent_pc_as_feat:
                    feature = torch.cat((k1,k2,gripper_pose,nxyz,right_eef_world_pos,col1,col2,gravity), dim=-1)
                else:
                    feature = torch.cat((k1,k2,gripper_pose,right_eef_world_pos,col1,col2,gravity), dim=-1)
            elif config['k1_type_1'] and not config['k2_type_1']:
                if latent_pc_as_feat:
                    feature = torch.cat((k2,gripper_pose,k1,nxyz,right_eef_world_pos,col1,col2,gravity), dim=-1)
                else:
                    feature = torch.cat((k2,gripper_pose,k1,right_eef_world_pos,col1,col2,gravity), dim=-1)
            elif not config['k1_type_1'] and config['k2_type_1']:
                if latent_pc_as_feat:
                    feature = torch.cat((k1,gripper_pose,k2,nxyz,right_eef_world_pos,col1,col2,gravity), dim=-1)
                else:
                    feature = torch.cat((k1,gripper_pose,k2,right_eef_world_pos,col1,col2,gravity), dim=-1)
            elif config['k1_type_1'] and config['k2_type_1']:
                if latent_pc_as_feat:
                    feature = torch.cat((gripper_pose,k1,k2,nxyz,right_eef_world_pos,col1,col2,gravity), dim=-1)
                else:
                    feature = torch.cat((gripper_pose,k1,k2,right_eef_world_pos,col1,col2,gravity), dim=-1)
            else:
                raise NotImplementedError
            
    elif config['k_option']==3:
        # no k
        if latent_pc_as_feat:
            feature = torch.cat((gripper_pose,nxyz,right_eef_world_pos,col1,col2,gravity), dim=-1)
        else:
            feature = torch.cat((gripper_pose,right_eef_world_pos,col1,col2,gravity), dim=-1)
    else:
        raise NotImplementedError

    model_input = {
        'xyz': nxyz.to(device='cuda',dtype=torch.float32),
        'feature': feature.to(device='cuda',dtype=torch.float32)
    }
    assert model_input["xyz"].dtype == torch.float32
    assert model_input["feature"].dtype == torch.float32
    assert not model_input["xyz"].isnan().any()
    assert not model_input["feature"].isnan().any()

    return model_input #,ref_output


# Prepare the input for the model
def prepare_model_input3(nxyz,neefpose, k,num_point,config):
    raise NotImplementedError

    B = nxyz.shape[0]
    Ho_num_point=nxyz.shape[1]
    # nxyz is the latent pc, [B,Ho*num_pts,type_1_feat*3]
    neefpose=neefpose.repeat(1,num_point, 1)# neefpose ([B, Ho*num_point, num_eef * (pose gripper action etc = 13)])

    # device='cuda'
    # vectors = torch.tensor([[1.0, 1.0, 1.0]] * B,device=nxyz.device)  # Shape: (B, 3)
    # angles = k.clone().detach()*torch.pi/(1+config['diffusion_steps']+config['pred_horizon'])
    # angles=angles.to(nxyz.device)
    # axes = vectors / torch.norm(vectors, dim=1, keepdim=True)  # Shape: (batch_size, 3)

    # # Rodrigues' formula
    # K = torch.zeros(B, 3, 3,device=nxyz.device)
    # K[:, 0, 1] = -axes[:, 2]
    # K[:, 0, 2] = axes[:, 1]
    # K[:, 1, 0] = axes[:, 2]
    # K[:, 1, 2] = -axes[:, 0]
    # K[:, 2, 0] = -axes[:, 1]
    # K[:, 2, 1] = axes[:, 0]

    # I = torch.eye(3,device=nxyz.device).unsqueeze(0).repeat(B, 1, 1)
    # angles = angles.unsqueeze(-1).unsqueeze(-1)

    # # Compute rotation matrices
    # rotation_matrices = I + torch.sin(angles) * K + (1 - torch.cos(angles)) * torch.bmm(K, K)
    # k1=rotation_matrices[:, :, 0] # [B,3] first col
    # k2=rotation_matrices[:, :, 1] # [B,3] second col
    # k1=k1.unsqueeze(1).expand(-1,nxyz.shape[1], -1) # [B,Ho*num_pts,3]
    # k2=k2.unsqueeze(1).expand(-1,nxyz.shape[1], -1) # [B,Ho*num_pts,3]

    # k1=k1.to(device)
    # k2=k2.to(device)

    # tensor_k = k.clone().detach().unsqueeze(-1).unsqueeze(-1).expand(-1,nxyz.shape[1], -1) # [B,Ho*num_pts,1]

    if config['Ho_in_B']:
        neefpose=neefpose.reshape(B*Ho_num_point,-1).unsqueeze(1)
        # tensor_k=tensor_k.reshape(B*Ho_num_point,-1).unsqueeze(1)
        # k1=k1.reshape(B*Ho_num_point,-1).unsqueeze(1)
        # k2=k2.reshape(B*Ho_num_point,-1).unsqueeze(1)
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
        
    # if not config['eef_xyz_feat']:
    #     if config['k_option']==0:
    #         # # 0 diffusion steps as type 0 scalar
    #         # num_fib_in = [2,4] # 14 in total, 2 type0: k; binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity
    #         # k: [B]        
    #         feature = torch.cat((tensor_k,gripper_pose,col1,col2,gravity), dim=-1)
    #     elif config['k_option']==1:
    #         # # 1 diffusion steps as type 0 rotation
    #         # num_fib_in = [7,4] # 19 in total, 7 type0: k1,k2; binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity
    #         feature = torch.cat((k1,k2,gripper_pose,col1,col2,gravity), dim=-1)
    #     elif config['k_option']==2:
    #         # # 2 diffusion steps as type 1 rotation
    #         # num_fib_in = [1,6] # 19 in total, 1 type0: binary gripper_action 6 type1: k1,k2; eef_abs_position, eef_abs_rotation (2cols); gravity
    #         feature = torch.cat((gripper_pose,k1,k2,col1,col2,gravity), dim=-1)
    #     elif config['k_option']==3:
    #         # no k
    #         feature = torch.cat((gripper_pose,col1,col2,gravity), dim=-1)
    #     else:
    #         raise NotImplementedError(f"k_option {config['k_option']} not implemented")
    # else:
    #     if config['k_option']==0:
    #         # # 0 diffusion steps as type 0 scalar
    #         # num_fib_in = [2,4] # 14 in total, 2 type0: k; binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity
    #         # k: [B]        
    #         feature = torch.cat((tensor_k,gripper_pose,nxyz,col1,col2,gravity), dim=-1)
    #     elif config['k_option']==1:
    #         # # 1 diffusion steps as type 0 rotation
    #         # num_fib_in = [7,4] # 19 in total, 7 type0: k1,k2; binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity
    #         feature = torch.cat((k1,k2,gripper_pose,nxyz,col1,col2,gravity), dim=-1)
    #     elif config['k_option']==2:
    #         # # 2 diffusion steps as type 1 rotation
    #         # num_fib_in = [1,6] # 19 in total, 1 type0: binary gripper_action 6 type1: k1,k2; eef_abs_position, eef_abs_rotation (2cols); gravity
    #         feature = torch.cat((gripper_pose,k1,k2,nxyz,col1,col2,gravity), dim=-1)
    #     elif config['k_option']==3:
    #         # no k
    #         feature = torch.cat((gripper_pose,nxyz,col1,col2,gravity), dim=-1)
    #     else:
    #         raise NotImplementedError(f"k_option {config['k_option']} not implemented")

    # ref_output=torch.cat((noisy_ori_actions,noisy_trans_actions), dim=-1)

    model_input = {
        'xyz': nxyz.to(device='cuda',dtype=torch.float32),
        'feature': feature.to(device='cuda',dtype=torch.float32)
    }
    assert model_input["xyz"].dtype == torch.float32
    assert model_input["feature"].dtype == torch.float32

    return model_input #,ref_output





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
def train_batch(nets, optimizer, lr_scheduler, noise_scheduler, nbatch,epoch_idx, device,gripper_noise_scheduler=None,config=None):
    global g_step
    g_step+=1
    nets.train()
    nxyz = nbatch['pc'][:, :, :, :3].to(device) # [B,Ho,num_pts,3]
    tgt_nxyz = nbatch['pc'][:, :, :, 3:6].to(device)
    naction = nbatch['action'].to(device) # [B,Hp,4by4]
    gt_gripper_action=naction[...,-1].unsqueeze(-1) # [B,Hp,1] # 0=close 1=open
    naction[...,-1]=1
    neefpose = nbatch['eef_pos'].to(device) # ([B, Ho, num_eef, pose gripper action etc])
    
    bz = nxyz.shape[0]
    ho = nxyz.shape[1]
    naction = naction.view(naction.size(0),naction.size(1),4,4) # naction: torch.Size([B, Hp, 4, 4])
    nxyz = nxyz.view(bz, -1, 3) # [B,Ho*num_pts,3]
    tgt_nxyz = tgt_nxyz.view(bz, -1, 3) # [B,Ho*num_pts,3]
    neefpose=neefpose.view(bz,ho,-1) # ([B, Ho, num_eef * (pose gripper action etc)])

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
    
        

    if config['use_ddpm']:
        # raise NotImplementedError
        # ddpm
        k = torch.randint(0, config["diffusion_steps"], (bz,), device=device)

    else:
        # Options
        if config['diffusion_option']==0 or config['diffusion_option']==1:
            # 0: the default, predict the gt H0
            # 1: predict relative transformation from Ht to H0
            k = torch.randint(0, config["diffusion_steps"], (bz,), device=device)
        elif config['diffusion_option']==2: 
            k = torch.zeros(bz, device=device).long()
        else:
            raise NotImplementedError(f"diffusion_option {config['diffusion_option']} not implemented")
    
    pc= prepare_model_input1(nxyz, tgt_nxyz,diff=config['diff'],pc_xyz_feat=config['pc_xyz_feat'],rgb=config['robomimic'])
    latent_pc=nets["pointcloud_encoder"](pc) # b,l,f (l:x*Hp; f:3x)

    num_point = config['pred_horizon']    
    if not config['use_ddpm'] and  (config['diffusion_option']==0 or config['diffusion_option']==1 or config['diffusion_option']==2):
        noisy_actions, actions_noise,actions,snr = noise_scheduler.add_noise9(naction, k, device=device,no_noise=config['no_noise'])
        if not config['snr']:
            snr=None
        else:
            pass
            # print(snr," >> raw snr <<")
    if config['noisy_action_as_k']:
        if config['k_on_lie']:
            if config['k_target']=='noise':
                _cond=se3.log(actions_noise) #[b,hp,trans+rots=6]
            elif config['k_target']=='noisy_actions':
                _cond=se3.log(noisy_actions) #[b,hp,trans+rots=6]
            elif config['k_target']=='actions':
                _cond=se3.log(actions) #[b,hp,trans+rots=6]
            else:
                raise NotImplementedError
        else:
            def hom2cols(matrix4by4):
                col1=matrix4by4[...,:3, 0]
                col2=matrix4by4[...,:3, 1]
                return torch.concatenate([col1,col2],dim=-1)
            if config['k_target']=='noise':
                _cond=hom2cols(actions_noise) #[b,hp,rot col1 col2=6]
            elif config['k_target']=='noisy_actions':
                _cond=hom2cols(noisy_actions) #[b,hp,rot col1 col2=6]
            elif config['k_target']=='actions':
                _cond=hom2cols(actions) #[b,hp,rot col1 col2=6]
            else:
                raise NotImplementedError
        _cond=_cond.repeat_interleave(config['obs_horizon'],dim=1)
        model_input = prepare_model_input2(latent_pc, neefpose, _cond, num_point,config,mean=nxyz.mean(dim=-2))
    else:
        model_input = prepare_model_input2(latent_pc, neefpose, k, num_point,config,mean=nxyz.mean(dim=-2))

    if config['testing']==1:
        model_input = prepare_model_input3(latent_pc,neefpose, k,num_point,config)

    if config['Ho_in_B']:
        num_point = config['pred_horizon*obs_horizon']

    return_raw=True
    # return_raw=False
    # if config['use_ddpm'] or config['unet']:
    #     return_raw=True
    model_output = nets["equivariant_pred_net"](model_input,num_point,return_raw=return_raw,Ho_in_B=config['Ho_in_B'])
    
    if config['sanity_check']:
        if config['sanity_check']==1: # se3 pc enc + unet only
            pass

    else:
        ori=model_output['ori']
        pos=model_output['pos']
        gripper=model_output['gripper']
        model_output=torch.cat((ori, pos,gripper), dim=-1) # [B,Hp,6+3+1]

        # must use unet, required for diffusion
        # if config['unet']:
        #     model_output= nets['unet'](model_output, k, global_cond=latent_pc.reshape(latent_pc.shape[0],-1))

        if config['use_ddpm']:
            # raise NotImplementedError
            # # TODO
            # ddpm
            noise = torch.randn(naction.shape, device=device)
            noise[:, :,:3, :3]=noise[:, :,:3, :3]
            noise[:, :, :3, 3] = noise[:, :, :3, 3]
            noise[:, :, 3, :3]=0.0
            noise[:, :, 3, 3]=1.0

            noisy_ori=noise[:, :,:3, :2].flatten(start_dim=-2)
            noisy_tran= noise[:, :, :3, 3]
            
            ori=naction[:, :,:3, :2].flatten(start_dim=-2)
            tran= naction[:, :, :3, 3]

            noise=torch.cat((noisy_ori,noisy_tran), dim=-1)
            naction=torch.cat((ori,tran), dim=-1)
        
            noisy_actions = noise_scheduler.add_noise(naction, noise, k)

            gripper_noise = torch.randn(gt_gripper_action.shape, device=device)
            noisy_gripper = gripper_noise_scheduler.add_noise(gt_gripper_action, gripper_noise, k)

            unet_input=torch.cat((noisy_actions,noisy_gripper),dim=-1) # [B,Hp,10]

        else:

            # Options
            if config['diffusion_option']==0 or config['diffusion_option']==1 or config['diffusion_option']==2:
                # 0: the default, predict the gt H0
                # 1: predict relative transformation from Ht to H0
                # [B,Ho,4,4]

                gripper_noise = torch.randn(gt_gripper_action.shape, device=device)
                noisy_gripper = gripper_noise_scheduler.add_noise(gt_gripper_action, gripper_noise, k)

                ori_indices = [(0, 0), (1,0), (2,0), (0, 1), (1,1), (2,1)] # first two cols
                selected_ori_actions = [noisy_actions[:, :, i, j] for i, j in ori_indices]
                trans_indices = [(0, 3), (1, 3), (2, 3)]
                selected_trans_actions = [noisy_actions[:, :, i, j] for i, j in trans_indices]
                noisy_ori_actions = torch.stack(selected_ori_actions, dim=-1)
                noisy_trans_actions = torch.stack(selected_trans_actions, dim=-1)
                unet_input=torch.cat((noisy_ori_actions,noisy_trans_actions,noisy_gripper),dim=-1) # [B,Hp,10]
            else:
                raise NotImplementedError(f"diffusion_option {config['diffusion_option']} not implemented")
            

        local_cond=None
        global_cond=None

        if config['local_cond']==0:
            local_cond=None
        elif config['local_cond']==1:
            # proposed actions as local
            local_cond=model_output
        else:
            raise NotImplementedError
        
        if config['global_cond']==0:
            # proposed actions as global
            global_cond=model_output.reshape(model_output.shape[0],-1)
        elif config['global_cond']==1:
            # latent pc as global
            global_cond=latent_pc.reshape(latent_pc.shape[0],-1)
        elif config['global_cond']==2:
            global_cond=None
        else:
            raise NotImplementedError

        unet_output= nets['unet'](unet_input, k, global_cond=global_cond,local_cond=local_cond)

        if config['use_ddpm']:
            # raise NotImplementedError
            # # TODO
            # ddpm
            target=noise
            if not config['ddpm_predict_noise']:
                target=noisy_actions
            target=torch.cat((target,gt_gripper_action),dim=-1)
            # # loss=torch.nn.functional.mse_loss(unet_output.reshape(target.shape[0],target.shape[1],-1), target.reshape(target.shape[0],target.shape[1],-1))
            # # TODO 2 cols and 1 trans from target


            # if isinstance(unet_output,dict):
            #     ori=unet_output['ori']
            #     pos=unet_output['pos']
            #     gripper=unet_output['gripper']

            #     unet_output=torch.cat((ori, pos,gripper), dim=-1) # [B,Hp,6+3+1]

            loss = nn.functional.mse_loss(unet_output,target)
            reconstructed_ori=unet_output[...,0:6].reshape(-1,6)
            reconstructed_pos=unet_output[...,6:9].reshape(-1,3)
            reconstructed_unet_output=process_action(reconstructed_ori, reconstructed_pos,follow_rot_trans_convention=True).view(unet_output.shape[0],-1,4,4)
            reconstructed_gripper_action=unet_output[...,9:10]
            reconstructed_ori=target[...,0:6].reshape(-1,6)
            reconstructed_pos=target[...,6:9].reshape(-1,3)
            reconstructed_target=process_action(reconstructed_ori, reconstructed_pos,follow_rot_trans_convention=True).view(unet_output.shape[0],-1,4,4)
            _, dist_r, dist_t,dist_g = compute_loss(reconstructed_unet_output.reshape(-1,4,4),reconstructed_target.reshape(-1,4,4),reconstructed_gripper_action,gt_gripper_action,snr=snr,gt_gripper_zero_one=not config['robomimic'])  
            # if dist_g is not None:
            #     loss=loss+dist_g
        else:
            if return_raw:
                output_ori=unet_output[...,0:6].reshape(-1,6)
                output_pos=unet_output[...,6:9].reshape(-1,3)
                final_action=process_action(output_ori, output_pos,follow_rot_trans_convention=True).view(unet_output.shape[0],-1,4,4)
                output_gripper_action=unet_output[...,9:10]
            # Options
            if config['predict_h0']:
                target=naction
            else:
                target=noisy_actions
            if config['diffusion_option']==0 or config['diffusion_option']==2:
                # 0: the default, predict the gt H0
                # see algorithm 1, but no more naction @torch.inverse(noisy_actions)
                loss, dist_r, dist_t, dist_g = compute_loss(final_action.reshape(-1,4,4),(target ).reshape(-1,4,4),output_gripper_action,gt_gripper_action,sign_mismatch=config['sign_mismatch'],snr=snr,gt_gripper_zero_one=not config['robomimic'])  
            elif config['diffusion_option']==1:
                # 1: predict relative transformation from Ht to H0
                loss, dist_r, dist_t, dist_g = compute_loss(torch.einsum('bhij,bhjk->bhjk',final_action,noisy_actions).reshape(-1,4,4),(target ).reshape(-1,4,4),output_gripper_action,gt_gripper_action,sign_mismatch=config['sign_mismatch'],snr=snr,gt_gripper_zero_one=not config['robomimic'])  
            else:
                raise NotImplementedError(f"diffusion_option {config['diffusion_option']} not implemented")

    if config['gate']:
        _losses=[m.matched_loss() for key in nets.keys() for m in nets[key].modules() if hasattr(m, "matched_loss") and m.matched_loss() is not None]
        penalty_loss=sum(_losses)/len(_losses)
        loss=loss+config['gate_weight']*penalty_loss
        print(penalty_loss.item(),">>> penalty loss <<<")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()
    loss_cpu = loss.item()
    wandb.log({"dist_R": dist_r},step=g_step)
    wandb.log({"dist_T": dist_t},step=g_step)
    if dist_g is not None:
        wandb.log({"dist_G": dist_g},step=g_step)
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

def set_seed(seed, torch_deterministic=False):
    """set seed across modules"""
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # wp.rand_init(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed

if __name__ == "__main__":
    main()