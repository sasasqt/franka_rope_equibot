import numpy as np
import torch
from torch import nn

from equibot.policies.utils.norm import Normalizer
from equibot.policies.utils.misc import to_torch
from equibot.policies.agents.etseed_policy import ETSEEDPolicy
from equibot.policies.utils.diffusion.lr_scheduler import get_scheduler

class ETSEEDAgent():
    def __init__(self, cfg):
        print(f"Initializing ETSEED agent.")
        self.cfg = cfg
        self._init_actor()

        self.device = cfg.device
        self.num_eef = cfg.env.num_eef
        self.dof = cfg.env.dof
        self.per_point=eval(str(cfg.env.per_point).title())
        self.flow=eval(str(cfg.franka_rope.flow).title())
        self.num_points = cfg.data.dataset.num_points
        self.obs_mode = cfg.model.obs_mode
        self.ac_mode = cfg.model.ac_mode
        self.obs_horizon = cfg.model.obs_horizon
        self.pred_horizon = cfg.model.pred_horizon
        self.ac_horizon = cfg.model.ac_horizon
        self.shuffle_pc = cfg.data.dataset.shuffle_pc
        self.pc_normalizer = None
        self.flow_normalizer=None
        self.state_normalizer = None
        self.ac_normalizer = None

    def _init_actor(self):
        self.actor = ETSEEDPolicy(self.cfg, device=self.cfg.device).to(self.cfg.device)


    # Initialize the model and optimizer
    def init_model_and_optimizer(device):
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
            num_training_steps=config["num_epochs"] * len(create_dataloader())
        )
        return nets, optimizer, lr_scheduler


    def update(self, batch, vis=False, train =True):
        self.train(train)

        batch = to_torch(batch, self.device)
        pc = batch["pc"]
        state = batch["eef_pos"]
        if train:
            gt_action = batch["action"]  # torch.Size([32, 16, self.num_eef * self.dof])

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
        
        return metrics



    def save_snapshot(self, save_path):
        state_dict = dict(
            actor=self.actor.state_dict(),
        )
        torch.save(state_dict, save_path)
        
    def _fix_state_dict_keys(self, state_dict):
        return {k: v for k, v in state_dict.items() if not "handle" in k}

    def load_snapshot(self, save_path):
        state_dict = torch.load(save_path)
        # if hasattr(self, "encoder_handle"):
        #     del self.encoder_handle
        #     del self.noise_pred_net_handle
        self.actor.load_state_dict(self._fix_state_dict_keys(state_dict["actor"]))
