import os
import numpy as np
import torch
import time
import torch.nn as nn
import wandb
from equibot.policies.utils.etseed.utils.loss_utils import compute_loss
from .etseed_train_sep2 import prepare_model_input1,prepare_model_input2,prepare_model_input3
from tqdm.auto import tqdm

# env import
from equibot.policies.utils.etseed.model.se3_transformer.equinet import SE3ManiNet_Fused, SE3VisionNet_Hierarchical
from equibot.policies.utils.etseed.utils.SE3diffusion_scheduler import DiffusionScheduler
from equibot.policies.utils.etseed.utils.group_utils import process_action #, orthogonalization
from equibot.policies.utils.etseed.utils.se_math import se3

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
        'unet_equivariance': cfg.dev.unet_equivariance,
        'Ho_in_B':cfg.dev.Ho_in_B,
        'diff': cfg.dev.diff,
        'bugfix':cfg.dev.bugfix,
        'testing': cfg.dev.testing,
        'arch':cfg.dev.arch,
        'pc_xyz_feat': cfg.dev.pc_xyz_feat,
        'eef_xyz_feat': cfg.dev.eef_xyz_feat,
        'num_degrees':cfg.dev.num_degrees,
        'num_channels':cfg.dev.num_channels,
        'num_heads':cfg.dev.num_heads,
        'channels_div':cfg.dev.channels_div,
        'global_cond':cfg.dev.global_cond,
        'local_cond':cfg.dev.local_cond,
        "loadFromCkpt": cfg.loadFromCkpt,

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
            loss_cpu = test_batch(nets, noise_scheduler,gripper_noise_scheduler,nbatch, device,config)
            test_losses.append(loss_cpu)
            tepoch.set_postfix(loss=loss_cpu)
    avg_test_loss = np.mean(test_losses)
    wandb.log({'test_loss': avg_test_loss},step=g_step)
    print(f"Test Done! Average Test Loss: {avg_test_loss}")


def init_model(device,config):
    from equibot.policies.etseed_train_sep2 import init_model_and_optimizer
    nets,_,_=init_model_and_optimizer(device,config,isNotTrain=True)
    return nets

# test a single batch of data
def test_batch(nets, noise_scheduler,gripper_noise_scheduler, nbatch, device,config,isVisualEval=False):
    nets.eval()
    if 'g_step' not in globals():
        global g_step
        g_step=-1

    with torch.no_grad():
        nxyz = nbatch['pc'][:, :, :, :3].to(device) # [B,Ho,num_pts,3]
        tgt_nxyz = nbatch['pc'][:, :, :, 3:6].to(device)
        if not isVisualEval:
            naction = nbatch['action'].to(device) # [B,Ho,4by4]
            gt_gripper_action=naction[...,-1].unsqueeze(-1) # [B,Hp,1] # 0=close 1=open
            naction[...,-1]=1                              
        neefpose = nbatch['eef_pos'].to(device) # ([B, Ho, num_eef, pose gripper action etc])
        bz = nxyz.shape[0]
        ho = nxyz.shape[1]
        hp=config["pred_horizon"]
        if not isVisualEval:
            naction = naction.view(naction.size(0),naction.size(1),4,4) # naction: torch.Size([B, Ho, 4, 4])
        num_point = nxyz.shape[2]
        nxyz = nxyz.view(bz, -1, 3) # [B,Ho*num_pts,3]
        tgt_nxyz = tgt_nxyz.view(bz, -1, 3) # [B,Ho*num_pts,3]
        neefpose=neefpose.view(bz,ho,-1) # ([B, Ho, num_eef * (pose gripper action etc)])

        H_Identity = torch.eye(4)[None].expand(bz,hp, -1, -1).to(device) # H_T: [B,Ho,4,4]
        if config['diffusion_option']==0 or config['diffusion_option']==1:
            k=torch.full((bz,), config['diffusion_steps'] - 1).long().to(device)
        elif config['diffusion_option']==2: 
            k=torch.full((bz,), 0).long().to(device)
        else:
            raise NotImplementedError(f"diffusion_option {config['diffusion_option']} not implemented")
    
        if config['use_ddpm']:
            noise = torch.randn(H_Identity.shape, device=device)
            noise[:, :,:3, :3]=noise[:, :,:3, :3]
            noise[:, :, :3, 3] = noise[:, :, :3, 3]
            noise[:, :, 3, :3]=0.0
            noise[:, :, 3, 3]=1.0
            noisy_ori=noise[:, :,:3, :2].flatten(start_dim=-2)
            noisy_tran= noise[:, :, :3, 3]
            
            noisy_actions=torch.cat((noisy_ori,noisy_tran),dim=-1)
        else:
            noisy_actions, noise=noise_scheduler.add_noise(H_Identity, k, device=device,no_noise=config['no_noise'])
        
        gripper_noise = torch.randn((bz,hp,1), device=device)
        noisy_gripper = gripper_noise # gripper_noise_scheduler.add_noise(gt_gripper_action, gripper_noise, k)

        if os.name == 'nt': # mock actions on windows 
            #actions=prepare_model_output(noisy_actions)
            return noisy_actions


        pc= prepare_model_input1(nxyz, tgt_nxyz,diff=config['diff'],pc_xyz_feat=config['pc_xyz_feat'])
        latent_pc=nets["pointcloud_encoder"](pc) # b,l,f (l:x*Hp; f:3x)

                
        if config['use_ddpm']:
            # # ddpm, the huggingface diffuser way
            noise_scheduler.set_timesteps(num_inference_steps=config['diffusion_steps'],device=device)
            for denoise_idx in noise_scheduler.timesteps: # shape [1]
                g_step+=1

                if (denoise_idx.item() +1==config['diffusion_steps'] or not config['k_option']==3):
                    k=torch.full((bz,), denoise_idx).long().to(device)

                    num_point = config['pred_horizon']
                    model_input = prepare_model_input2(latent_pc, neefpose, k, num_point,config)

                    if config['testing']==1:
                        model_input = prepare_model_input3(latent_pc,neefpose, k,num_point,config)

                    if config['Ho_in_B']:
                        num_point = config['pred_horizon*obs_horizon']

                    return_raw=True
                    model_output = nets["equivariant_pred_net"](model_input,num_point,return_raw=return_raw,Ho_in_B=config['Ho_in_B'])
                    ori=model_output['ori']
                    pos=model_output['pos']
                    gripper=model_output['gripper']
                    model_output=torch.cat((ori, pos,gripper), dim=-1) # [B,Hp,6+3+1]
                    
                unet_input=torch.cat((noisy_actions,noisy_gripper),dim=-1) # [B,Hp,10]

                    
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
                
                unet_output= nets['unet'](unet_input, denoise_idx, global_cond=global_cond,local_cond=local_cond)

                output_ori_pos=unet_output[...,0:9]
                output_gripper_action=unet_output[...,9:10]


                # rot=model_outpudev.k_option=1 dev.unet=True dev.use_ddpm=True dev.low_memory=True dev.arch=2t.reshape(-1,4,4)[...,:3,:3]
                # print(torch.det(rot),'MMMMMMMMMM')
                # assert torch.all(torch.det(rot)>=0.0), rot # actually allclose 1.0


                if not config['early_return']:
                    #noisy_actions = noise_scheduler.step(model_output.view(model_output.shape[0],model_output.shape[1],4,4), denoise_idx, noisy_actions).prev_sample      
                    noisy_actions = noise_scheduler.step(output_ori_pos, denoise_idx, noisy_actions).prev_sample      
                    noisy_gripper = gripper_noise_scheduler.step(output_gripper_action, denoise_idx, output_gripper_action).prev_sample      

                else:
                    #noisy_actions = noise_scheduler.step(model_output.view(model_output.shape[0],model_output.shape[1],4,4), denoise_idx, noisy_actions).pred_original_sample
                    noisy_actions = noise_scheduler.step(output_ori_pos, denoise_idx, noisy_actions).pred_original_sample
                    noisy_gripper = gripper_noise_scheduler.step(output_gripper_action, denoise_idx, output_gripper_action).pred_original_sample      


                # rot=noisy_actions.reshape(-1,4,4)[...,:3,:3]
                # print(torch.det(rot),'AAAAAAAAAAA')
                #assert torch.all(torch.det(rot)>=0.0), rot

                if not isVisualEval:
                    pred_original_sample=noise_scheduler.step(model_output, denoise_idx, noisy_actions).pred_original_sample
                    output_ori=pred_original_sample[...,0:6].reshape(-1,6)
                    output_pos=pred_original_sample[...,6:9].reshape(-1,3)
                    final_output=process_action(output_ori, output_pos,follow_rot_trans_convention=True).view(bz,-1,4,4)
                    
                    loss, dist_R, dist_T = compute_loss(final_output.view(-1,4,4), naction.view(-1,4,4))

                    loss_cpu = loss.item()

                    wandb.log({"test_dist_R": dist_R},step=g_step)
                    wandb.log({"test_dist_T": dist_T},step=g_step)
                    wandb.log({"test_loss_cpu": loss_cpu},step=g_step)

                if config['early_return']:
                    break

            # output_ori=noisy_actions[...,0:6].reshape(-1,6)
            # output_pos=noisy_actions[...,6:9].reshape(-1,3)
            # final_output=process_action(output_ori, output_pos,follow_rot_trans_convention=True).view(bz,-1,4,4)

            assert not torch.any(torch.isnan(noisy_actions)), noisy_actions
            if isVisualEval:
                output_ori=noisy_actions[...,0:6].reshape(-1,6)
                output_pos=noisy_actions[...,6:9].reshape(-1,3)
                final_output=process_action(output_ori, output_pos,follow_rot_trans_convention=True).view(bz,-1,4,4)
                
                actions=final_output
                actions[...,3,3]=noisy_gripper.squeeze(-1)

                return actions
            else:
                return loss_cpu
            
        else:    
            gripper_noise_scheduler.set_timesteps(num_inference_steps=config['diffusion_steps'],device=device)

            # predict action instead of noise might due to https://github.com/lucidrains/denoising-diffusion-pytorch/issues/58#issuecomment-2676085515
            # but why does the predicted action at denoise_idx=num_steps already good, if not the best action?
            for denoise_idx in range(config['diffusion_steps'] - 1, 0, -1):
                g_step+=1

                if config['diffusion_option']==0 or config['diffusion_option']==1:
                    k=torch.full((bz,), denoise_idx).long().to(device)
                elif config['diffusion_option']==2: 
                    k=torch.full((bz,), 0).long().to(device)
                else:
                    raise NotImplementedError(f"diffusion_option {config['diffusion_option']} not implemented")
            
                if (denoise_idx +1==config['diffusion_steps'] or not config['k_option']==3):
                # if (denoise_idx +1==config['diffusion_steps'] ):
            
                    num_point = config['pred_horizon']

                    if config['noisy_action_as_k']:
                        if config['k_on_lie']:
                            # TODO: is this correct? do we need noise in denoise?
                            if config['k_target']=='noise':
                                _cond=se3.log(noise) #[b,hp,trans+rots=6]
                            elif config['k_target']=='noisy_actions':
                                if config['diffusion_steps'] - 1==denoise_idx:
                                    _cond=se3.log(noisy_actions) #[b,hp,rot col1 col2=6]
                                else:
                                    _cond=se3.log(noise@noisy_actions) #[b,hp,rot col1 col2=6]
                                    # _cond=se3.log(noise)+se3.log(noisy_actions) #[b,hp,rot col1 col2=6] # for lie  algebra

                                    # _cond=se3.log(noisy_actions) #[b,hp,rot col1 col2=6]
                            elif config['k_target']=='actions':
                                _cond=se3.log(noisy_actions) #[b,hp,trans+rots=6]
                            else:
                                raise NotImplementedError
                        else:
                            def hom2cols(matrix4by4):
                                col1=matrix4by4[...,:3, 0]
                                col2=matrix4by4[...,:3, 1]
                                return torch.concatenate([col1,col2],dim=-1)
                            if config['k_target']=='noise':
                                _cond=hom2cols(noise) #[b,hp,rot col1 col2=6]
                            elif config['k_target']=='noisy_actions':
                                if config['diffusion_steps'] - 1==denoise_idx:
                                    _cond=hom2cols(noisy_actions) #[b,hp,rot col1 col2=6]
                                else:
                                    _cond=hom2cols(noise@noisy_actions) #[b,hp,rot col1 col2=6]
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
                    model_output = nets["equivariant_pred_net"](model_input,num_point,return_raw=return_raw,Ho_in_B=config['Ho_in_B'])
                    ori=model_output['ori']
                    pos=model_output['pos']
                    gripper=model_output['gripper']
                    model_output=torch.cat((ori, pos,gripper), dim=-1) # [B,Hp,6+3+1]

                ori_indices = [(0, 0), (1,0), (2,0), (0, 1), (1,1), (2,1)] # first two cols
                selected_ori_actions = [noisy_actions[:, :, i, j] for i, j in ori_indices]
                trans_indices = [(0, 3), (1, 3), (2, 3)]
                selected_trans_actions = [noisy_actions[:, :, i, j] for i, j in trans_indices]
                noisy_ori_actions = torch.stack(selected_ori_actions, dim=-1)
                noisy_trans_actions = torch.stack(selected_trans_actions, dim=-1)
                unet_input=torch.cat((noisy_ori_actions,noisy_trans_actions,noisy_gripper),dim=-1)
                
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



                output_ori=unet_output[...,0:6].reshape(-1,6)
                output_pos=unet_output[...,6:9].reshape(-1,3)
                action=process_action(output_ori, output_pos,follow_rot_trans_convention=True).view(model_output.shape[0],-1,4,4)
                
                    
                output_gripper_action=unet_output[...,9:10]
                noisy_gripper = gripper_noise_scheduler.step(output_gripper_action, denoise_idx, output_gripper_action).prev_sample      

                # Options
                if config['diffusion_option']==0:
                    # 0: the default, predict the gt H0
                    reconstructed_H_0 = action
                    noisy_actions,noise = noise_scheduler.denoise9(
                        reconstructed_H_0=reconstructed_H_0,
                        timestep = k,
                        sample = noisy_actions,
                        device = device
                    )      
                    # noisy_actions=noise@noisy_actions          
                elif config['diffusion_option']==1:
                    # 1: predict relative transformation from Ht to H0
                    reconstructed_H_0 = torch.einsum('bhij,bhjk->bhjk',action,noisy_actions)
                    noisy_actions = noise_scheduler.denoise9(
                        reconstructed_H_0=reconstructed_H_0,
                        timestep = k,
                        sample = noisy_actions,
                        device = device
                    )
                elif config['diffusion_option']==2:
                    # 2: no diffusion, no denoising
                    reconstructed_H_0=action
                elif config['diffusion_option']==3:
                    # no diffusion during inference
                    noisy_actions=action
                    config['early_return']=True
                else:
                    raise NotImplementedError(f"diffusion_option {config['diffusion_option']} not implemented")

                assert not torch.any(torch.isnan(model_output)), model_output
                assert not torch.any(torch.isnan(noisy_actions)), noisy_actions
                assert not torch.any(torch.isnan(action)), action

                rot=noisy_actions.reshape(-1,4,4)[...,:3,:3]
                tran=noisy_actions.reshape(-1,4,4)[...,:3,3]
                # print()
                # print(torch.det(rot),'RRRRRRRRRR')
                # print(tran,'TTTTTTTTTT')
                # print(noisy_actions)
                assert torch.any(torch.det(rot)>=0.0), rot
                assert not torch.any(torch.isnan(tran)), tran

                if not isVisualEval:
                    loss, dist_R, dist_T, dist_G = compute_loss(reconstructed_H_0.reshape(-1,4,4), naction.reshape(-1,4,4))
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
                    if dist_G is not None:
                        wandb.log({"test_dist_G": dist_G},step=g_step)

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
                # actions=noisy_actions#reconstructed_H_0#noisy_actions# for lie algebra

                actions=reconstructed_H_0#reconstructed_H_0#noisy_actions
                # TODO BUG?
                actions[...,3,3]=output_gripper_action.squeeze(-1)
                return actions
            else:
                return loss_cpu




if __name__ == "__main__":
    main()