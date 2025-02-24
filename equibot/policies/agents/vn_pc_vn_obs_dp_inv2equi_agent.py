import torch
from torch import nn
import numpy as np
import time

from equibot.policies.utils.norm import Normalizer
from equibot.policies.utils.misc import to_torch
from equibot.policies.agents.dp_agent import DPAgent
from equibot.policies.agents.dp_policy import DPPolicy
from equibot.policies.utils.VN_transformer.VN_transformer import VNTransformer
from equibot.policies.vision.vec_layers import VecLinear
from equibot.policies.vision.sim3_encoder import SIM3Vec4Latent
from equibot.policies.utils.diffusion.conditional_unet1d import ConditionalUnet1D
from equibot.policies.utils.diffusion.ema_model import EMAModel

import copy
import hydra

class WTFAgent(DPAgent):
    
    def _convert_state_to_vec(self, state):
        # state format for 3d and 4d actions: eef_pos
        # state format for 7d actions: eef_pos, eef_rot_x, eef_rot_z, gravity_dir, gripper_pose, [optional] goal_pos
        # input: (B, H, E * eef_dim)
        # output: (B, H, ?, 3) [need norm] + (B, H, ?, 3) [does not need norm] + maybe (B, H, E)
        if self.dof == 3:
            return state.view(state.shape[0], state.shape[1], -1, 3), None, None
        elif self.dof == 4:
            state = state.view(state.shape[0], state.shape[1], self.num_eef, -1)
            assert state.shape[-1] in [4, 7]
            eef_pos = state[:, :, :, :3]
            scalar_state = state[:, :, :, 3]
            if state.shape[-1] == 7:
                goal_pos = state[:, :, :, -3:]
                vec_state_pos = torch.cat([eef_pos, goal_pos], dim=2)
            else:
                vec_state_pos = eef_pos
            return vec_state_pos, None, scalar_state
        else:
            state = state.view(state.shape[0], state.shape[1], self.num_eef, -1)
            assert state.shape[-1] in [13, 16]
            eef_pos = state[:, :, :, :3]
            dir1 = state[:, :, :, 3:6]
            dir2 = state[:, :, :, 6:9]
            gravity_dir = state[:, :, :, 9:12]
            gripper_pose = state[:, :, :, 12]

            if state.shape[-1] > 13:
                goal_pos = state[:, :, :, 13:16]
                vec_state_pos = torch.cat([eef_pos, goal_pos], dim=2)
            else:
                vec_state_pos = eef_pos
            vec_state_dir = torch.cat([dir1, dir2, gravity_dir], dim=2)
            scalar_state = gripper_pose
            return vec_state_pos, vec_state_dir, scalar_state

    def _convert_action_to_vec(self, ac, batch=None):
        # input: (B, H, E * dof); output: (B, ac_dim, 3, H) + maybe (B, E, H)
        # rotation actions are always treated as relative axis-angle rotation
        ac = ac.view(ac.shape[0], ac.shape[1], -1, self.dof)
        if self.dof in [4, 7]:
            gripper_ac = ac[:, :, :, 0]  # (B, H, E)
            eef_ac = ac[:, :, :, 1:]  # (B, H, E, 3)
            if self.dof == 7:
                eef_ac = eef_ac.reshape(
                    ac.shape[0], ac.shape[1], -1, 3
                )  # (B, H, E * 2, 3)
            return eef_ac.permute(0, 2, 3, 1), gripper_ac.permute(0, 2, 1)
        elif self.dof == 3:
            return ac.permute(0, 2, 3, 1), None
        else:
            raise ValueError(f"Cannot handle dof = {self.dof}")
        
    def _init_actor(self):
        self.actor = DPPolicy(self.cfg, device=self.cfg.device).to(self.cfg.device)
        del self.actor.encoder
        self.actor.encoder =self.actor.encoder_handle = SIM3Vec4Latent(**self.cfg.model.encoder).to(self.cfg.device)

        obs_horizon=self.cfg.model.obs_horizon
        obs_dim=self.obs_dim=(self.cfg.model.hidden_dim+4) # if dof == 7

        del self.actor.noise_pred_net 
        self.actor.noise_pred_net = ConditionalUnet1D(
            input_dim=self.cfg.env.dof * self.cfg.env.num_eef,
            diffusion_step_embed_dim=obs_dim * obs_horizon*3,
            global_cond_dim=obs_dim * obs_horizon*3,
        ).to(self.cfg.device)

        del self.actor.nets
        self.actor.nets = nn.ModuleDict(
            {"encoder": self.actor.encoder, "noise_pred_net": self.actor.noise_pred_net}
        )
        self.actor.ema = EMAModel(model=copy.deepcopy(self.actor.nets), power=0.75)

        self.actor.ema.averaged_model.to(self.cfg.device)
        self.obs_dim=None
        self.pc_scale = None
        self.flow_scale=None
        self.fc_z_inv = None
        self.fc_gt_ac_inv = None
        self.inv2equi=None
        self.final_resize=None

    def _init_normalizers(self, batch):
        if self.ac_normalizer is None:
            gt_action = batch["action"]
            flattened_gt_action = gt_action.view(-1, self.dof)
            if self.dof == 7:
                indices = [[0], [1, 2, 3], [4, 5, 6]]
            elif self.dof == 4:
                indices = [[0], [1, 2, 3]]
            else:
                indices = None
            ac_normalizer = Normalizer(
                flattened_gt_action, symmetric=True, indices=indices
            )
            self.ac_normalizer = Normalizer(
                {
                    "min": ac_normalizer.stats["min"].tile((self.num_eef,)),
                    "max": ac_normalizer.stats["max"].tile((self.num_eef,)),
                }
            )

            print(f"Action normalization stats: {self.ac_normalizer.stats}")
        if self.state_normalizer is None:
            # dof layout: maybe gripper open/close, xyz, maybe rot
            if self.dof == 3:
                self.state_normalizer = ac_normalizer
            else:
                self.state_normalizer = Normalizer(
                    {
                        "min": ac_normalizer.stats["min"][1:4],
                        "max": ac_normalizer.stats["max"][1:4],
                    }
                )
            self.actor.state_normalizer = self.state_normalizer
        if self.pc_normalizer is None:
            self.pc_normalizer = self.state_normalizer
            self.actor.pc_normalizer = self.pc_normalizer
            if self.flow:
                self.flow_normalizer = self.state_normalizer
                self.actor.flow_normalizer = self.pc_normalizer

        # compute action scale relative to point cloud scale
        _cnt=3
        if self.flow:
            _cnt=6
        pc = batch["pc"].reshape(-1, self.num_points, _cnt)
        xyz=pc[:,:,:3]
        centroid = xyz.mean(1, keepdim=True)
        centered_xyz = xyz - centroid
        pc_scale = centered_xyz.norm(dim=-1).mean()
        ac_scale = ac_normalizer.stats["max"].max()
        self.pc_scale = pc_scale / ac_scale
        self.actor.pc_scale = self.pc_scale

        if self.flow:
            flow=pc[:,:,3:]
            flow_scale=flow.norm(dim=-1).mean()
            self.flow_scale=flow_scale / self.pc_scale
            self.actor.flow_scale=self.flow_scale

    def update(self, batch, vis=False, train =True):
        self.train(train)

        batch = to_torch(batch, self.device)
        pc = batch["pc"]
        # rgb = batch["rgb"]
        state = batch["eef_pos"]
        if train:
            gt_action = batch["action"]  # torch.Size([32, 16, self.num_eef * self.dof])
        if self.pc_scale is None:
            self._init_normalizers(batch)

        pc_shape = pc.shape
        if not self.flow:
            pc = self.pc_normalizer.normalize(pc)
        else:
            _pc=self.pc_normalizer.normalize(pc.view(-1, 6)[:,::2])
            _flow=self.flow_normalizer.normalize(pc.view(-1, 6)[:,1::2])
            pc=torch.cat((_pc, _flow), dim=-1) #  torch.Size([81920, 6])
            pc=pc.reshape(pc_shape) # torch.Size([1024, 2, 40, 6])
        if train:
            gt_action = self.ac_normalizer.normalize(gt_action)

        batch_size = B = pc.shape[0]
        Ho = self.obs_horizon
        Hp = self.pred_horizon

        if self.obs_mode == "state":
            z_pos, z_dir, z_scalar = self._convert_state_to_vec(state)
            z_pos = self.state_normalizer.normalize(z_pos)
            if self.dof > 4:
                z = torch.cat([z_pos, z_dir], dim=-2)
            else:
                z = z_pos
        else:
            feat_dict = self.actor.encoder_handle(pc, ret_perpoint_feat=self.per_point,flow=self.flow, target_norm=self.pc_scale,flow_norm=self.flow_scale)

            center = (
                feat_dict["center"].reshape(B, Ho, 1, 3)[:, [-1]].repeat(1, Ho, 1, 1)
            )
            scale = feat_dict["scale"].reshape(B, Ho, 1, 1)[:, [-1]].repeat(1, Ho, 1, 1)
            z_pos, z_dir, z_scalar = self._convert_state_to_vec(state) # torch.Size([1024, Ho, 1, 3]) torch.Size([1024, Ho, 3, 3]) torch.Size([1024, Ho, 1])
            z_pos = self.state_normalizer.normalize(z_pos)
            z_pos = (z_pos - center) / scale
            z = feat_dict["so3"] # torch.Size([1024*Ho, h, 3])
            z = z.reshape(B, Ho, -1, 3) # torch.Size([1024, Ho, h, 3])
            if self.dof > 4:
                z = torch.cat([z, z_pos, z_dir], dim=-2) # torch.Size([1024, Ho, h+4, 3])
            else:
                z = torch.cat([z, z_pos], dim=-2)
            if self.fc_z_inv is None:
                self.fc_z_inv=VecLinear(z.size(-2), z.size(-2),device=self.device)
            # if self.per_point:
            #     _per_point=feat_dict["per_point_so3"] #  torch.Size([4096, 8, 3, 8])
            #     _hidden_dim=_per_point.shape[-1]
            #     _per_point=_per_point.reshape(B, Ho, -1, 3) # torch.Size([1024, 4, 64, 3])
            #     z = torch.cat([z, _per_point], dim=-2)

        obs_cond_vec, obs_cond_scalar = z.reshape(B, -1, 3), ( # torch.Size([1024, 48/304, 3]) torch.Size([1024, 4])
            z_scalar.reshape(B, -1) if z_scalar is not None else None
        )

        if self.obs_mode.startswith("pc"):
            if self.ac_mode == "abs":
                center = (
                    feat_dict["center"]
                    .reshape(B, Ho, 1, 3)[:, [-1]]
                    .repeat(1, Hp, 1, 1)
                )
            else:
                center = 0
            scale = feat_dict["scale"].reshape(B, Ho, 1, 1)[:, [-1]].repeat(1, Hp, 1, 1)

            if train:
                gt_action = gt_action.reshape(B, Hp, self.num_eef, self.dof)
                if self.dof == 4:
                    gt_action = torch.cat(
                        [gt_action[..., :1], (gt_action[..., 1:] - center) / scale], dim=-1
                    )
                    gt_action = gt_action.reshape(B, Hp, -1)
                elif self.dof == 3:
                    gt_action = (gt_action - center) / scale
                elif self.dof == 7:
                    gt_action = torch.cat(
                        [
                            gt_action[..., :1],
                            (gt_action[..., 1:4] - center) / scale,
                            gt_action[..., 4:],
                        ],
                        dim=-1,
                    )
                    gt_action = gt_action.reshape(B, Hp, -1) # B,Hp,num_eef*eef_dof
                else:
                    raise ValueError(f"Dof {self.dof} not supported.")
        # vec_eef_action, vec_gripper_action = self.actor._convert_action_to_vec(
        #     gt_action, batch
        # )


        # TODO equivariant latent z ([B, obs, pc+state, 3]) to invariant latent z'

        _z=z.view(z.size(0)*z.size(1),-1,z.size(-1)) #? torch.Size([1024*Ho, h+4, 3])
        z_inv_dual, _ = self.fc_z_inv(_z[..., None])
        z_inv_dual = z_inv_dual.squeeze(-1)
        z_inv = (z_inv_dual * _z).sum(-1,keepdim=True).expand_as(_z)  # torch.Size([1024*Ho, h+4, 3])
        

        # TODO invariant parts in vanilla diffusion
        #   how to make a inv of gt action?
        #   diffusion output = action = (batch_size, self.pred_horizon, self.action_dim)

        obs_cond = z_inv.reshape(batch_size, -1)  # torch.Size([1024,Ho*(h+4)*3])

        if not train:
            initial_noise_scale = 0.0
            noisy_action = (
                torch.randn((batch_size, self.pred_horizon, self.cfg.env.dof * self.cfg.env.num_eef)).to(
                    self.device
                )
                * initial_noise_scale
            )
            curr_action = noisy_action
                    
            if hasattr(self.cfg.model, "num_diffusion_iters"):
                self.num_diffusion_iters = self.cfg.model.num_diffusion_iters
            else:
                self.num_diffusion_iters = self.cfg.model.noise_scheduler.num_train_timesteps

            self.noise_scheduler = hydra.utils.instantiate(self.cfg.model.noise_scheduler)
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            ema_nets=self.actor.ema.averaged_model

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = ema_nets["noise_pred_net"](
                    sample=curr_action, timestep=k, global_cond=obs_cond
                )

                # inverse diffusion step
                curr_action = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=curr_action
                ).prev_sample

            #ret = dict(ac=curr_action)
            gripper_action=curr_action[...,:1]
            vec_action=curr_action[...,1:].reshape(batch_size,-1,3)
            #return vec_eef_action

            _z_shape=z.view(batch_size,-1,3).size(1)

            _min_c=min(_z_shape,vec_action.size(-2))
            _max_c=max(_z_shape,vec_action.size(-2))
            
            if _min_c == vec_action.size(-2):
                resized_ac,_=self.final_resize(vec_action)
                rots=torch.einsum('bhi,bhj->bij',z.view(batch_size,-1,3),resized_ac)
            else:
                resized_z,_=self.final_resize(z.view(batch_size,-1,3))
                rots=torch.einsum('bhi,bhj->bij',resized_z,vec_action)

            det=torch.linalg.det(rots) # 1024 3 3

            correction = torch.where(det < 0, -1.0, 1.0).view(-1, 1, 1)
            det=det.view(-1,1,1)*correction
            reconstructed_action=torch.einsum('bcd,bde->bce', vec_action, rots/(det+1e-8))
            reconstructed_action=reconstructed_action.view(batch_size,-1,6)
            reconstructed_action=torch.cat((gripper_action,reconstructed_action),dim=-1)
            self.ac_normalizer.unnormalize(reconstructed_action)       
            return reconstructed_action.cpu().detach().numpy()


        # print(gt_action.shape) # torch.Size([1024, Hp, 7=(1+3*2)])
        action = gt_action.view(gt_action.shape[0], gt_action.shape[1], -1, self.dof) # torch.Size([1024, Hp, num_eef, 7=(1+3*2)])
        gt_gripper_action=action[:,:,:,:1].view(gt_action.size(0),-1,1) # torch.Size([1024, Hp*num_eef, 1])
        vec_eef_action=action[:,:,:,1:].view(gt_action.size(0),gt_action.size(1),-1,3) 
        vec_eef_action=vec_eef_action.reshape(vec_eef_action.size(0),-1,3) #  torch.Size([1024, Hp*num_eef*2, 3])

    
        if self.fc_gt_ac_inv is None:
            self.fc_gt_ac_inv=VecLinear(vec_eef_action.size(-2), vec_eef_action.size(-2),device=self.device)

        gt_ac_inv_dual, _ = self.fc_gt_ac_inv(vec_eef_action[..., None])
        gt_ac_inv_dual = gt_ac_inv_dual.squeeze(-1)
        gt_ac_inv = (gt_ac_inv_dual * vec_eef_action).sum(-1,keepdim=True).expand_as(vec_eef_action)  #  torch.Size([1024, Hp*num_eef*2, 3])

        _gt_gripper_action=gt_gripper_action
        _gt_ac_inv=gt_ac_inv.reshape(gt_gripper_action.size(0),gt_gripper_action.size(1),-1) #  #  torch.Size([1024, Hp*num_eef, 3*2])

        _cat=torch.cat((_gt_gripper_action,_gt_ac_inv),dim=-1) # torch.Size([1024, Hp*num_eef, 1+3*2])

        noise = torch.randn(_cat.shape, device=self.device)

        timesteps = torch.randint(
            0,
            self.actor.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()

        noisy_actions = self.actor.noise_scheduler.add_noise(
            _cat, noise, timesteps
        )


        # print(noisy_actions.shape,timesteps.shape,obs_cond.shape) # torch.Size([1024, 16, 7]) torch.Size([1024]) torch.Size([1024, 132])
        # if self.cfg.model.use_torch_compile:
        #     noise_pred = self.actor.noise_pred_net_handle(
        #         noisy_actions, timesteps, global_cond=obs_cond
        #     )
        # else:
        #     noise_pred = self.actor.noise_pred_net(
        #         noisy_actions, timesteps, global_cond=obs_cond
        #     )
            
        noise_pred = self.actor.noise_pred_net(
            noisy_actions, timesteps, global_cond=obs_cond
        )

        # loss = nn.functional.mse_loss(noise_pred, noise)

        # TODO invariant diffused output and equivariant latent z in vn-transformer


        _z_shape=z.view(batch_size,-1,3).size(1)

        _min_c=min(_z_shape,vec_eef_action.size(-2))
        _max_c=max(_z_shape,vec_eef_action.size(-2))
        
        final_resize=self.final_resize=VecLinear(_min_c,_max_c,device=self.device)

        if _min_c == vec_eef_action.size(-2):
            resized_ac,_=final_resize(vec_eef_action)
            rots=torch.einsum('bhi,bhj->bij',z.view(batch_size,-1,3),resized_ac)
        else:
            resized_z,_=final_resize(z.view(batch_size,-1,3))
            rots=torch.einsum('bhi,bhj->bij',resized_z,vec_eef_action)

        det=torch.linalg.det(rots) # 1024 3 3

        noise_loss = nn.functional.mse_loss(noise_pred, noise)

        correction = torch.where(det < 0, -1.0, 1.0).view(-1, 1, 1)
        det=det.view(-1,1,1)*correction
        reconstructed_action=torch.einsum('bcd,bde->bce', gt_ac_inv, rots/(det+1e-8))
        rot_loss= nn.functional.mse_loss(reconstructed_action, vec_eef_action)
                
        loss=noise_loss + rot_loss
        print(noise_loss.item(),rot_loss.item())
        # if self.inv2equi is None:
        #     inv2equi=VNTransformer(        
        #     dim=gt_action.size(-1),
        #     depth=12,
        #     num_tokens = None,
        #     dim_feat = None,
        #     dim_head = 64,
        #     heads = 4,
        #     dim_coor = 3,
        #     reduce_dim_out = True,
        #     bias_epsilon = 0.,
        #     l2_dist_attn = False,
        #     flash_attn = True,)

        # inv2equi(coors=z, feats=gt_ac_inv)

        # vn diffusion
        # if self.dof != 7:
        #     noise = torch.randn(gt_action.shape, device=self.device)
        #     vec_eef_noise, vec_gripper_noise = self.actor._convert_action_to_vec(
        #         noise, batch
        #     )  # to debug
        # else:
        #     vec_eef_noise = torch.randn_like(vec_eef_action, device=self.device)
        #     vec_gripper_noise = torch.randn_like(vec_gripper_action, device=self.device)

        # timesteps = torch.randint(
        #     0,
        #     self.actor.noise_scheduler.config.num_train_timesteps,
        #     (B,),
        #     device=self.device,
        # ).long()

        # if vec_gripper_action is not None:
        #     noisy_eef_actions = self.actor.noise_scheduler.add_noise(
        #         vec_eef_action, vec_eef_noise, timesteps
        #     )
        #     noisy_gripper_actions = self.actor.noise_scheduler.add_noise(
        #         vec_gripper_action, vec_gripper_noise, timesteps
        #     )

        #     vec_eef_noise_pred, vec_gripper_noise_pred = (
        #         self.actor.noise_pred_net_handle(
        #             noisy_eef_actions.permute(0, 3, 1, 2),
        #             timesteps,
        #             scalar_sample=noisy_gripper_actions.permute(0, 2, 1),
        #             cond=obs_cond_vec,
        #             scalar_cond=obs_cond_scalar,
        #         )
        #     )
        #     vec_eef_noise_pred = vec_eef_noise_pred.permute(0, 2, 3, 1)
        #     vec_gripper_noise_pred = vec_gripper_noise_pred.permute(0, 2, 1)
        #     if self.dof != 7:
        #         noise_pred = self.actor._convert_action_to_scalar(
        #             vec_eef_noise_pred, vec_gripper_noise_pred, batch=batch
        #         ).view(noise.shape)
        # else:
        #     noisy_eef_actions = self.actor.noise_scheduler.add_noise(
        #         vec_eef_action, vec_eef_noise, timesteps
        #     )

        #     vec_noise_pred = self.actor.noise_pred_net_handle(
        #         noisy_eef_actions.permute(0, 3, 1, 2),
        #         timesteps,
        #         cond=obs_cond_vec,
        #         scalar_cond=obs_cond_scalar,
        #     )[0].permute(0, 2, 3, 1)
        #     if self.dof != 7:
        #         noise_pred = self.actor._convert_action_to_scalar(
        #             vec_noise_pred, batch=batch
        #         ).view(noise.shape)

        # if self.dof == 7:
        #     n_vec = np.prod(vec_eef_noise_pred.shape)
        #     n_sca = np.prod(vec_gripper_noise_pred.shape)
        #     k = (n_vec) / (n_vec + n_sca)
        #     loss = nn.functional.mse_loss(
        #         vec_eef_noise_pred, vec_eef_noise
        #     ) * k + nn.functional.mse_loss(
        #         vec_gripper_noise_pred, vec_gripper_noise
        #     ) * (
        #         1 - k
        #     )
        # else:
        #     loss = nn.functional.mse_loss(noise_pred, noise)
        # if torch.isnan(loss):
        #     print(f"Loss is nan, please investigate.")
        #     import pdb

        #     pdb.set_trace()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.actor.step_ema()

        metrics = {
            "loss": loss,
            "normalized_gt_ac_max": np.max(
                np.abs(vec_eef_action.reshape(-1, 3).detach().cpu().numpy()), axis=0
            ).mean(),
        }
        # if self.dof == 7:
        #     metrics.update(
        #         {
        #             "mean_gt_eef_noise_norm": np.linalg.norm(
        #                 vec_eef_noise.detach().cpu().numpy(), axis=1
        #             ).mean(),
        #             "mean_pred_eef_noise_norm": np.linalg.norm(
        #                 vec_eef_noise_pred.detach().cpu().numpy(), axis=1
        #             ).mean(),
        #             "mean_gt_gripper_noise_norm": np.linalg.norm(
        #                 vec_gripper_noise.detach().cpu().numpy(), axis=1
        #             ).mean(),
        #             "mean_pred_gripper_noise_norm": np.linalg.norm(
        #                 vec_gripper_noise_pred.detach().cpu().numpy(), axis=1
        #             ).mean(),
        #         }
        #     )
        # else:
        #     metrics.update(
        #         {
        #             "mean_gt_noise_norm": np.linalg.norm(
        #                 noise.reshape(gt_action.shape[0], -1).detach().cpu().numpy(),
        #                 axis=1,
        #             ).mean(),
        #             "mean_pred_noise_norm": np.linalg.norm(
        #                 noise_pred.reshape(gt_action.shape[0], -1)
        #                 .detach()
        #                 .cpu()
        #                 .numpy(),
        #                 axis=1,
        #             ).mean(),
        #         }
        #     )

        return metrics




    def save_snapshot(self, save_path):
        state_dict = dict(
            actor=self.actor.state_dict(),
            ema_model=self.actor.ema.averaged_model.state_dict(),
            pc_scale=self.pc_scale,
            flow_scale=self.flow_scale,
            pc_normalizer=self.pc_normalizer.state_dict(),
            state_normalizer=self.state_normalizer.state_dict(),
            ac_normalizer=self.ac_normalizer.state_dict(),
            flow_normalizer=None if self.flow_normalizer is None else self.flow_normalizer.state_dict(), # AttributeError: 'NoneType' object has no attribute 'state_dict'

        )
        torch.save(state_dict, save_path)

    def fix_checkpoint_keys(self, state_dict):
        fixed_state_dict = dict()
        for k, v in state_dict.items():
            if "encoder.encoder" in k:
                fixed_k = k.replace("encoder.encoder", "encoder")
            else:
                fixed_k = k
            if "handle" in k:
                continue
            fixed_state_dict[fixed_k] = v
        return fixed_state_dict

    def load_snapshot(self, save_path):
        state_dict = torch.load(save_path)
        self.state_normalizer = Normalizer(state_dict["state_normalizer"])
        self.actor.state_normalizer = self.state_normalizer
        self.ac_normalizer = Normalizer(state_dict["ac_normalizer"])
        if self.obs_mode.startswith("pc"):
            self.pc_normalizer = self.state_normalizer
            self.actor.pc_normalizer = self.pc_normalizer
            if self.flow:
                self.flow_normalizer = self.state_normalizer
                self.actor.flow_normalizer = self.flow_normalizer
        del self.actor.encoder_handle
        del self.actor.noise_pred_net_handle
        self.actor.load_state_dict(self.fix_checkpoint_keys(state_dict["actor"]))
        self.actor._init_torch_compile()
        self.actor.ema.averaged_model.load_state_dict(
            self.fix_checkpoint_keys(state_dict["ema_model"])
        )
        self.pc_scale = state_dict["pc_scale"]
        self.actor.pc_scale = self.pc_scale
        self.flow_scale = state_dict["flow_scale"]
        self.actor.flow_scale = self.flow_scale



    def act(self, obs, return_dict=False, debug=False):
        self.train(False)
        assert isinstance(obs["pc"][0][0], np.ndarray)
        if len(obs["state"].shape) == 3:
            assert len(obs["pc"][0].shape) == 2  # (obs_horizon, N, 3)
            obs["pc"] = [[x] for x in obs["pc"]]
            for k in obs:
                if k != "pc" and isinstance(obs[k], np.ndarray):
                    obs[k] = obs[k][:, None]
            has_batch_dim = False
        elif len(obs["state"].shape) == 4:
            assert len(obs["pc"][0][0].shape) == 2  # (obs_horizon, B, N, 3)
            has_batch_dim = True
        else:
            raise ValueError("Input format not recognized.")

        ac_dim = self.num_eef * self.dof
        batch_size = len(obs["pc"][0])

        state = obs["state"].reshape(tuple(obs["state"].shape[:2]) + (-1,))

        # process the point clouds
        # some point clouds might be invalid
        # if this occurs, exclude these batch items
        xyzs = []
        ac = np.zeros([batch_size, self.pred_horizon, ac_dim])
        if return_dict:
            ac_dict = []
            for i in range(batch_size):
                ac_dict.append(None)
        forward_idxs = list(np.arange(batch_size))
        _cnt=3
        if self.flow:
            _cnt=6

        for pcs in obs["pc"]:
            xyzs.append([])
            for batch_idx, xyz in enumerate(pcs):
                if not batch_idx in forward_idxs:
                    xyzs[-1].append(np.zeros((self.num_points, _cnt)))
                elif xyz.shape[0] == 0:
                    # no points in point cloud, return no-op action
                    forward_idxs.remove(batch_idx)
                    xyzs[-1].append(np.zeros((self.num_points, _cnt)))
                elif self.shuffle_pc:
                    choice = np.random.choice(
                        xyz.shape[0], self.num_points, replace=True
                    )
                    xyz = xyz[choice, :]
                    xyzs[-1].append(xyz)
                else:
                    step = xyz.shape[0] // self.num_points
                    xyz = xyz[::step, :][: self.num_points]
                    xyzs[-1].append(xyz)

        if len(forward_idxs) > 0:
            torch_obs = dict(
                pc=torch.tensor(np.array(xyzs).swapaxes(0, 1)[forward_idxs])
                .to(self.device)
                .float(),
                state=torch.tensor(state.swapaxes(0, 1)[forward_idxs])
                .to(self.device)
                .float(),
            )
            for k in obs:
                if not k in ["pc", "state"] and isinstance(obs[k], np.ndarray):
                    torch_obs[k] = (
                        torch.tensor(obs[k].swapaxes(0, 1)[forward_idxs])
                        .to(self.device)
                        .float()
                    )

            raw_ac_dict = self.actor(torch_obs, debug=debug)
        else:
            raw_ac_dict = torch.zeros(
                (batch_size, self.actor.pred_horizon, self.actor.action_dim)
            ).to(self.actor.device)
        for i, idx in enumerate(forward_idxs):
            if return_dict:
                ac_dict[idx] = {k: v[i] for k, v in raw_ac_dict.items()}
            unnormed_action = (
                self.ac_normalizer.unnormalize(raw_ac_dict["ac"][i])
                .detach()
                .cpu()
                .numpy()
            )
            ac[idx] = unnormed_action
        if not has_batch_dim:
            ac = ac[0]
            if return_dict:
                ac_dict = ac_dict[0]
        if return_dict:
            return ac, ac_dict
        else:
            return ac
