# import os
# import sys
# import copy
# import hydra
# import torch
# import wandb
# import omegaconf
# import numpy as np
# import getpass as gt
# from tqdm import tqdm
# from glob import glob
# from omegaconf import OmegaConf

# from equibot.policies.utils.media import save_video
# from equibot.policies.utils.misc import get_env_class, get_dataset, get_agent
# from equibot.policies.vec_eval import run_eval
# from equibot.envs.subproc_vec_env import SubprocVecEnv

# import logging
# from torch.utils.tensorboard import SummaryWriter
# import torch.onnx

# @hydra.main(config_path="configs", config_name="fold_synthetic")
# def main(cfg):
#     assert cfg.mode == "train"
#     np.random.seed(cfg.seed)

#     logging.basicConfig(level=logging.INFO)

#     # initialize parameters
#     batch_size = cfg.training.batch_size

#     # setup logging
#     if cfg.use_wandb:
#         wandb_config = omegaconf.OmegaConf.to_container(
#             cfg, resolve=True, throw_on_missing=False
#         )
#         wandb.init(
#             entity=cfg.wandb.entity,
#             project=cfg.wandb.project,
#             tags=["train"],
#             name=cfg.prefix,
#             settings=wandb.Settings(code_dir="."),
#             config=wandb_config,
#         )
#     log_dir = os.getcwd()

#     # init dataloader
#     train_dataset = get_dataset(cfg, "train")
#     num_workers = min(os.cpu_count(),cfg.data.dataset.num_workers)
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         num_workers=0, # faster and workaround for pickle in windows
#         shuffle=True,
#         drop_last=False, # was True
#         pin_memory=True,
#     )
#     cfg.data.dataset.num_training_steps = (
#         cfg.training.num_epochs * len(train_dataset) // batch_size
#     )

#     # init env
#     env_fns = []
#     env_class = get_env_class(cfg.env.env_class)
#     env_args = dict(OmegaConf.to_container(cfg.env.args, resolve=True))
#     logging.info(env_args) 

#     def create_env(env_args, i):
#         # env_args.seed = cfg.seed * 100 + i
#         env_args['seed']=cfg.seed * 100 + i
#         return env_class(OmegaConf.create(env_args))

#     if cfg.training.eval_interval <= cfg.training.num_epochs:
#         envs = []
#         # oversubscription alittle, decouple #cores and #eval_episodes
#         n_envs=int(os.cpu_count()*1.5)

#         for i in range((cfg.training.num_eval_episodes-1)//n_envs+1):
#             if (i == (cfg.training.num_eval_episodes-1)//n_envs+1):
#                 n_envs = cfg.training.num_eval_episodes % n_envs or n_envs
#             envs.append(SubprocVecEnv(
#                 [
#                     lambda seed=i: create_env(env_args, seed)
#                     for i in range(n_envs)
#                 ]
#             ))
#         # envs.append(SubprocVecEnv(
#         #     [
#         #         lambda seed=i: create_env(env_args, seed)
#         #         for i in range(cfg.training.num_eval_episodes)
#         #     ]
#         # ))
#     else:
#         envs = None

#     # init agent
#     agent = get_agent(cfg.agent.agent_name)(cfg)

#     # for architecture visualization
#     # if (cfg.agent.agent_name == 'dp'):
#     #     # visualize encoder w/ tensorboard/onnx
#     #     batch=next(iter(train_loader))
#     #     pc =batch["pc"]
#     #     pc_shape = pc.shape
#     #     flattened_pc = pc.reshape(batch_size * agent.obs_horizon, *pc_shape[-2:])
#     #     writer = SummaryWriter('encoder')     
#     #     writer.add_graph(agent.actor.encoder, flattened_pc.permute(0, 2, 1).to(agent.device),use_strict_trace=False)
#     #     writer.close()
#     #     agent.actor.encoder.eval()
#     #     torch.onnx.export(agent.actor.encoder,
#     #                 flattened_pc.permute(0, 2, 1).to(agent.device),
#     #                 "encoder.onnx",
#     #                 export_params=True,
#     #                 opset_version=10,
#     #                 do_constant_folding=True,
#     #                 input_names=['input'],
#     #                 output_names=['output'],
#     #                 dynamic_axes={'input': {0: 'batch_size'},
#     #                             'output': {0: 'batch_size'}})

#     #     # visualize diffusion_unet w/ tensorboard/onnx
#     #     z = agent.actor.encoder(flattened_pc.permute(0, 2, 1).to(agent.device))["global"]
#     #     z = z.reshape(batch_size, agent.obs_horizon, -1).to(agent.device)
#     #     batch["eef_pos"]=batch["eef_pos"].reshape(
#     #             tuple(batch["eef_pos"].shape[:2]) + (-1,)
#     #         )
#     #     state = batch["eef_pos"].to(agent.device)
#     #     logging.info(z.shape)
#     #     logging.info(state.shape)
#     #     z = torch.cat([z, state], dim=-1)
#     #     obs_cond = z.reshape(batch_size, -1)  # (BS, obs_horizion * obs_dim)
#     #     gt_action = batch["action"]
#     #     noise = torch.randn(gt_action.shape).to(agent.device)
#     #     timesteps = torch.randint(
#     #         0,
#     #         agent.actor.noise_scheduler.config.num_train_timesteps,
#     #         (batch_size,),
#     #         device=agent.device,
#     #     ).long()
#     #     noisy_actions = agent.actor.noise_scheduler.add_noise(
#     #     gt_action.to(agent.device), noise.to(agent.device), timesteps.to(agent.device)
#     #     )

#     #     """
#     #     x: (B,T,input_dim)
#     #     timestep: (B,) or int, diffusion step
#     #     local_cond: (B,T,local_cond_dim)
#     #     global_cond: (B,global_cond_dim)
#     #     output: (B,T,input_dim)
#     #     """
#     #     inputs = (
#     #         noisy_actions.to(agent.device), # [1536, 16, 14]
#     #         timesteps.to(agent.device),     # [1536,]
#     #         obs_cond.to(agent.device)       # [1536, 116,]
#     #     )
#     #     writer = SummaryWriter('diffusion_unet')
#     #     writer.add_graph(agent.actor.noise_pred_net, inputs,use_strict_trace=False)
#     #     writer.close()
#     #     agent.actor.noise_pred_net.eval()
#     #     torch.onnx.export(agent.actor.noise_pred_net,
#     #                 inputs,
#     #                 "diffusion_unet.onnx",
#     #                 export_params=True,
#     #                 opset_version=10,
#     #                 do_constant_folding=True,
#     #                 input_names=['input'],
#     #                 output_names=['output'],
#     #                 dynamic_axes={'input': {0: 'batch_size'},
#     #                             'output': {0: 'batch_size'}})
#     # else: 
#     #     # visualize SIM(3) encoder w/ tensorboard/onnx
#     #     batch=next(iter(train_loader))
#     #     pc = batch["pc"]
#     #     # rgb = batch["rgb"]
#     #     state = batch["eef_pos"].to(agent.device)
#     #     gt_action = batch["action"]  # torch.Size([32, 16, agent.num_eef * agent.dof])
#     #     pc_shape = pc.shape
#     #     batch_size = B = pc.shape[0]
#     #     Ho = agent.obs_horizon
#     #     Hp = agent.pred_horizon
#     #     writer = SummaryWriter('SIM3_encoder')     
#     #     writer.add_graph(agent.actor.encoder, pc.to(agent.device),use_strict_trace=False)
#     #     writer.close()
#     #     agent.actor.encoder.eval()
#     #     torch.onnx.export(agent.actor.encoder,
#     #                 pc.to(agent.device),
#     #                 "SIM3_encoder.onnx",
#     #                 export_params=True,
#     #                 opset_version=10,
#     #                 do_constant_folding=True,
#     #                 input_names=['input'],
#     #                 output_names=['output'],
#     #                 dynamic_axes={'input': {0: 'batch_size'},
#     #                             'output': {0: 'batch_size'}})
        
#     #     # visualize SIM(3) unet w/ tensorboard/onnx
#     #     feat_dict = agent.actor.encoder_handle(pc.to(agent.device))
#     #     center = (
#     #         feat_dict["center"].reshape(B, Ho, 1, 3)[:, [-1]].repeat(1, Ho, 1, 1)
#     #     )
#     #     scale = feat_dict["scale"].reshape(B, Ho, 1, 1)[:, [-1]].repeat(1, Ho, 1, 1)
#     #     z_pos, z_dir, z_scalar = agent.actor._convert_state_to_vec(state)
#     #     z_pos = (z_pos - center) / scale
#     #     z = feat_dict["so3"]
#     #     z = z.reshape(B, Ho, -1, 3)
#     #     if agent.dof > 4:
#     #         z = torch.cat([z, z_pos, z_dir], dim=-2)
#     #     else:
#     #         z = torch.cat([z, z_pos], dim=-2)
#     #     obs_cond_vec, obs_cond_scalar = z.reshape(B, -1, 3), (
#     #         z_scalar.reshape(B, -1) if z_scalar is not None else None
#     #     )
#     #     if agent.obs_mode.startswith("pc"):
#     #         if agent.ac_mode == "abs":
#     #             center = (
#     #                 feat_dict["center"]
#     #                 .reshape(B, Ho, 1, 3)[:, [-1]]
#     #                 .repeat(1, Hp, 1, 1)
#     #             )
#     #         else:
#     #             center = 0
#     #         scale = feat_dict["scale"].reshape(B, Ho, 1, 1)[:, [-1]].repeat(1, Hp, 1, 1).to('cpu')
#     #         gt_action = gt_action.reshape(B, Hp, agent.num_eef, agent.dof)
#     #         # logging.info(gt_action.device)
#     #         # logging.info(center.device)
#     #         # logging.info(scale.device)
            
#     #         if agent.dof == 4:
#     #             gt_action = torch.cat(
#     #                 [gt_action[..., :1], (gt_action[..., 1:] - center) / scale], dim=-1
#     #             )
#     #             gt_action = gt_action.reshape(B, Hp, -1)
#     #         elif agent.dof == 3:
#     #             gt_action = (gt_action - center) / scale
#     #         elif agent.dof == 7:
#     #             gt_action = torch.cat(
#     #                 [
#     #                     gt_action[..., :1],
#     #                     (gt_action[..., 1:4] - center) / scale,
#     #                     gt_action[..., 4:],
#     #                 ],
#     #                 dim=-1,
#     #             )
#     #             gt_action = gt_action.reshape(B, Hp, -1)
#     #         else:
#     #             raise ValueError(f"Dof {agent.dof} not supported.")
#     #     vec_eef_action, vec_gripper_action = agent.actor._convert_action_to_vec(
#     #         gt_action, batch
#     #     )
#     #     vec_eef_action=vec_eef_action.to(agent.device)
#     #     vec_gripper_action=vec_gripper_action.to(agent.device)
#     #     if agent.dof != 7:
#     #         noise = torch.randn(gt_action.shape, device=agent.device)
#     #         vec_eef_noise, vec_gripper_noise = agent.actor._convert_action_to_vec(
#     #             noise, batch
#     #         )  # to debug
#     #     else:
#     #         vec_eef_noise = torch.randn_like(vec_eef_action, device=agent.device)
#     #         vec_gripper_noise = torch.randn_like(vec_gripper_action, device=agent.device)

#     #     timesteps = torch.randint(
#     #         0,
#     #         agent.actor.noise_scheduler.config.num_train_timesteps,
#     #         (B,),
#     #         device=agent.device,
#     #     ).long()

#     #     noisy_eef_actions = agent.actor.noise_scheduler.add_noise(
#     #         vec_eef_action, vec_eef_noise, timesteps
#     #     )
#     #     noisy_gripper_actions = agent.actor.noise_scheduler.add_noise(
#     #         vec_gripper_action, vec_gripper_noise, timesteps
#     #     )
#     #     inputs=(            
#     #         noisy_eef_actions.permute(0, 3, 1, 2).to(agent.device),
#     #         timesteps.to(agent.device),
#     #             noisy_gripper_actions.permute(0, 2, 1).to(agent.device),
#     #         obs_cond_vec.to(agent.device),
#     #         obs_cond_scalar.to(agent.device),
#     #         )
#     #     writer = SummaryWriter('SIM3_diffusion_unet')
#     #     writer.add_graph(agent.actor.noise_pred_net, inputs,use_strict_trace=False)
#     #     writer.close()
#     #     agent.actor.noise_pred_net.eval()
#     #     torch.onnx.export(agent.actor.noise_pred_net,
#     #                 inputs,
#     #                 "SIM3_diffusion_unet.onnx",
#     #                 export_params=True,
#     #                 opset_version=10,
#     #                 do_constant_folding=True,
#     #                 input_names=['input'],
#     #                 output_names=['output'],
#     #                 dynamic_axes={'input': {0: 'batch_size'},
#     #                             'output': {0: 'batch_size'}})
        
#     # del agent
#     # agent = get_agent(cfg.agent.agent_name)(cfg)

#     if cfg.training.ckpt is not None:
#         agent.load_snapshot(cfg.training.ckpt)
#         start_epoch_ix = int(cfg.training.ckpt.split("/")[-1].split(".")[0][4:])
#     else:
#         start_epoch_ix = 0

#     # train loop
#     global_step = 0
#     best_n = {}
#     for epoch_ix in tqdm(range(start_epoch_ix, cfg.training.num_epochs)):
#         batch_ix = 0
#         for batch in tqdm(train_loader, leave=False, desc="Batches"):
#             train_metrics = agent.update(
#                 batch, vis=epoch_ix % cfg.training.vis_interval == 0 and batch_ix == 0
#             )
#             if cfg.use_wandb:
#                 wandb.log(
#                     {"train/" + k: v for k, v in train_metrics.items()},
#                     step=global_step,
#                 )
#                 wandb.log({"epoch": epoch_ix}, step=global_step)
#             del train_metrics
#             global_step += 1
#             batch_ix += 1
#         if (
#             (
#                 epoch_ix % cfg.training.eval_interval == 0
#                 or epoch_ix == cfg.training.num_epochs - 1
#             )
#             and epoch_ix > 0
#             and envs is not None
#         ):
#             # ~~TODO SAVE BEST????~~
#             # better: do not eval during training, eval afterwards
#             rewards=[]
#             for i in range(len(envs)):
#                 env = envs[i]
#                 eval_metrics = run_eval(
#                     env,
#                     agent,
#                     vis=True,
#                     num_episodes=len(envs[i]),
#                     reduce_horizon_dim=cfg.data.dataset.reduce_horizon_dim,
#                     use_wandb=cfg.use_wandb,
#                 )
#                 rewards.append(eval_metrics['rew_values'])
#                 if cfg.use_wandb:
#                     # if epoch_ix > cfg.training.eval_interval and "vis_pc" in eval_metrics:
#                     #     # only save one pc per run to save space
#                     #     del eval_metrics["vis_pc"]
#                     wandb.log(
#                         {
#                             "eval/" + k: v
#                             for k, v in eval_metrics.items()
#                             if not k in ["vis_rollout", "rew_values"]
#                         },
#                         step=global_step,
#                     )
#                     if "vis_rollout" in eval_metrics:
#                         for idx, eval_video in enumerate(eval_metrics["vis_rollout"]):
#                             eval_idx = i*len(envs[0])+idx
#                             video_path = os.path.join(
#                                 log_dir,
#                                 f"eval{epoch_ix:05d}_ep{eval_idx}_rew{eval_metrics['rew_values'][eval_idx]}.mp4",
#                             )
#                             save_video(eval_video, video_path)
#                             print(f"Saved eval video to {video_path}")
#                 logging.info(eval_metrics['rew_values'])
#             # #success is more important (zero reward is unacceptable)
#             avg_score = geometric_mean_excluding_zero(rewards)+non_zero_elements(rewards)
#             best_n[epoch_ix]=avg_score
#             best_n=keep_best(_best=getattr(cfg.training, 'best', 10), _dict=best_n)
#             save_path = os.path.join(log_dir, f"BEST_ckpt{epoch_ix:05d}_avg-score{avg_score}.pth")
#             agent.save_snapshot(save_path)
#             del eval_metrics
        
#         if (
#             epoch_ix % cfg.training.save_interval == 0
#             or epoch_ix == cfg.training.num_epochs - 1
#         ):
#             save_path = os.path.join(log_dir, f"ckpt{epoch_ix:05d}.pth")
#             num_ckpt_to_keep = 10000 # keep them all
#             if len(list(glob(os.path.join(log_dir, "ckpt*.pth")))) > num_ckpt_to_keep:
#                 # remove old checkpoints
#                 for fn in list(sorted(glob(os.path.join(log_dir, "ckpt*.pth"))))[
#                     :-num_ckpt_to_keep
#                 ]:
#                     os.remove(fn)
#             agent.save_snapshot(save_path)

#     # with open(f"{log_dir}/best_n.txt", 'w') as f:
#     #     f.write(str(best_n))


# def geometric_mean_excluding_zero(numbers):
#     return np.exp(np.mean(np.log(numbers[numbers != 0])))
# def non_zero_elements(elements):
#     _arr = np.array(elements)
#     return sum(x != 0 for x in _arr)
# def keep_best(_best, _dict):
#     _sorted_items = sorted(_dict.items(), key=lambda x: x[1], reverse=True)
#     return dict(_sorted_items[:_best])

# if __name__ == "__main__":
#     main()
