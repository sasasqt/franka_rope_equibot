import torch
import os
from .se3_backbone import SE3Backbone, ExtendedModule
from .se3_transformer.model.fiber import Fiber
from ...utils.group_utils import process_action #, orthogonalization

class SE3ManiNet_Equivariant_Separate(ExtendedModule):
    def __init__(self, voxelize=False):
        super().__init__()
        ''' 
        input features:
        rgb: 3 type0
        action: 9 type0 (the first 2 columns of rotation matrix(2x3), and a translation vector(1x3))
        pose: 9 type0 (the first 2 columns of rotation matrix(2x3), and a translation vector(1x3))
        k: 1 type0 (denote currently is the k-th denoising step)

        output features:
        action: 2 type1(rotation) + 3 type0(translation)
        '''
        num_fib_in = [7,2] # 13 in total, 7 type0:tensor_k,noisy_ori_actions, 2 type1: noisy_trans_actions,tgt_nxyz
        num_fib_out = [1,2]
        self.pos_net = SE3Backbone(
            fiber_in=Fiber({
                "0": num_fib_in[0], 
                "1": num_fib_in[1], 
            }),
            fiber_out=Fiber({
                "1": 1, # offset/translation
            }),
            num_layers= 4,
            num_degrees= 3,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
        )
        self.ori_net = SE3Backbone(
            fiber_in=Fiber({
                "0": num_fib_in[0], 
                "1": num_fib_in[1], 
            }),
            fiber_out=Fiber({
                "0": 6, # 2 cols of rotation
            }),
            num_layers= 4,
            num_degrees= 4,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
        )

    def forward(self, inputs,return_raw=False,**kwargs):
        bs = inputs["xyz"].shape[0] 
        pos_output = self.pos_net(inputs)    
        ori_output = self.ori_net(inputs)

        # process translation
        feature_list = list()
        for i in range(bs):
            batchi_feature = pos_output["feature"][i] # [Horizon, 3]
            #actioni_raw = torch.mean(batchi_feature,dim = 0)
            feature_list.append(batchi_feature)
        output_pos = torch.stack(feature_list,dim = 0) # [B, Horizon, 3]
        
        # process orientation
        feature_list = list()
        for i in range(bs):
            batchi_feature = ori_output["feature"][i] # [Horizon, 6]
            feature_list.append(batchi_feature)
        action = torch.stack(feature_list,dim = 0) # [B, Horizon, 6]

        if return_raw:
            return{
                'pos':output_pos,
                'ori':action
            }
    
        action = process_action(action.view(-1,6), output_pos.view(-1,3)).view(bs,-1,4,4) # orthogonalization
        return action # [B, Ho, 4, 4]  


class SE3ManiNet_Invariant_Separate(ExtendedModule):
    def __init__(self, voxelize=False):
        super().__init__()
        num_fib_in = [7,2] # 13 in total, 7 type0:tensor_k,noisy_ori_actions, 2 type1: noisy_trans_actions,tgt_nxyz
        num_fib_out = [6]
        self.pos_net = SE3Backbone(
            fiber_in=Fiber({
                "0": num_fib_in[0], 
                "1": num_fib_in[1], 
            }),
            fiber_out=Fiber({
                "0": 3, # offset/translation
            }),
            num_layers= 4,
            num_degrees= 3,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
        )
        self.ori_net = SE3Backbone(
            fiber_in=Fiber({
                "0": num_fib_in[0], 
                "1": num_fib_in[1], 
            }),
            fiber_out=Fiber({
                "0": 6, # 2 cols of rotation
            }),
            num_layers= 4,
            num_degrees= 4,
            num_channels= 8,
            num_heads= 1,
            channels_div= 2,
            voxelize = voxelize,
        )

    def forward(self, inputs,return_raw=False,**kwargs):
        bs = inputs["xyz"].shape[0] 
        pos_output = self.pos_net(inputs)        
        ori_output = self.ori_net(inputs)

        # process translation
        feature_list = list()
        for i in range(bs):
            batchi_feature = pos_output["feature"][i] # [Horizon, 3]
            #actioni_raw = torch.mean(batchi_feature,dim = 0)
            feature_list.append(batchi_feature)
        output_pos = torch.stack(feature_list,dim = 0) # [B, Horizon, 3]
        
        # process orientation
        feature_list = list()
        for i in range(bs):
            batchi_feature = ori_output["feature"][i] # [Horizon, 6]
            feature_list.append(batchi_feature)
        action = torch.stack(feature_list,dim = 0) # [B, Horizon, 6]

        if return_raw:
            return{
                'pos':output_pos,
                'ori':action
            }
    
        action = process_action(action.view(-1,6), output_pos.view(-1,3)).view(bs,-1,4,4) # orthogonalization
        return action # [B, Ho, 4, 4]
    



# class SE3ManiNet_Fused_Separate(ExtendedModule):
#     def __init__(self, voxelize=False):
#         super().__init__()
#         num_fib_in = [7,2] # 13 in total, 7 type0:tensor_k,noisy_ori_actions, 2 type1: noisy_trans_actions,tgt_nxyz
#         num_fib_out = [6]
#         self.pos_net = SE3Backbone(
#             fiber_in=Fiber({
#                 "0": num_fib_in[0], 
#                 "1": num_fib_in[1], 
#             }),
#             fiber_out=Fiber({
#                 "0": 3, # offset/translation
#                 "1": 1, # offset/translation
#             }),
#             num_layers= 4,
#             num_degrees= 4,
#             num_channels= 8,
#             num_heads= 2,
#             channels_div= 2,
#             voxelize = voxelize,
#         )
#         self.ori_net = SE3Backbone(
#             fiber_in=Fiber({
#                 "0": num_fib_in[0], 
#                 "1": num_fib_in[1], 
#             }),
#             fiber_out=Fiber({
#                 "0": 6, # 2 cols of rotation
#             }),
#             num_layers= 4,
#             num_degrees= 4,
#             num_channels= 8,
#             num_heads= 2,
#             channels_div= 2,
#             voxelize = voxelize,
#         )

#     def forward(self, inputs,num_point,return_raw=False,Inv=True):
#         bs = inputs["xyz"].shape[0]
#         pos_output = self.pos_net(inputs)        
#         ori_output = self.ori_net(inputs)

#         # process translation
#         feature_list = list()
#         for i in range(bs):
#             if Inv:
#                 batchi_feature = pos_output["feature"][i][:,:3] # [Horizon, 3]
#             else: # Equiv
#                 batchi_feature = pos_output["feature"][i][:,3:] # [Horizon, 3]

#             #actioni_raw = torch.mean(batchi_feature,dim = 0)
#             feature_list.append(batchi_feature)
#         output_pos = torch.stack(feature_list,dim = 0) 
#         output_pos=torch.mean(output_pos.view(bs,-1,num_point,3),dim=2) # [B, Horizon, 3]

#         # process orientation
#         feature_list = list()
#         for i in range(bs):
#             batchi_feature = ori_output["feature"][i] # [Horizon, 6]
#             feature_list.append(batchi_feature)
#         action = torch.stack(feature_list,dim = 0)
#         action=torch.mean(action.view(bs,-1,num_point,6),dim=2) # [B, Horizon, 6]

#         if return_raw:
#             return{
#                 'pos':output_pos,
#                 'ori':action
#             }
    
#         action = process_action(action.view(-1,6), output_pos.view(-1,3),follow_rot_trans_convention=True).view(bs,-1,4,4) # orthogonalization
#         return action # [B, Ho, 4, 4]
    


class SE3ManiNet_Fused(ExtendedModule):
    def __init__(
            self, 
            voxelize=False,
            k_neighbours=8,
            pred_horizon=8,
            config=None,
            no_tgt_nxyz=False,
            eef_abs_position_as_node=False,
            latent_pc_as_feat=False,
            eef_xyz_feat=False,
            fused=True,
            ):
        assert not config==None
        super().__init__()
        self.pred_horizon=pred_horizon
        self.config=config
        self.fused=fused

        num_degrees= config['num_degrees']
        num_channels= config['num_channels']
        num_heads= config['num_heads']
        channels_div= config['channels_div']
        
        type1_cnt=0
        if eef_abs_position_as_node:
            type1_cnt+=1

        if latent_pc_as_feat:
            type1_cnt-=3

        if no_tgt_nxyz:
            type1_cnt+=1

        if eef_xyz_feat:
            type1_cnt-=1

        # # Options
        # if config['k_option']==0:
        #     # 0 diffusion steps as type 0 scalar
        #     num_fib_in = [2,5-type1_cnt] # 17 in total, 2 type0: k; binary gripper_action 5 type1: tgt_nxyz; eef_abs_position, eef_abs_rotation (2cols); gravity
        # elif config['k_option']==1:
        #     # 1 diffusion steps as type 0 rotation
        #     num_fib_in = [7,5-type1_cnt] # 22 in total, 7 type0: k1,k2; binary gripper_action 5 type1: tgt_nxyz; eef_abs_position, eef_abs_rotation (2cols); gravity
        # elif config['k_option']==2:
        #     # 2 diffusion steps as type 1 rotation
        #     num_fib_in = [1,7-type1_cnt] # 22 in total, 1 type0: binary gripper_action 7 type1: k1,k2; tgt_nxyz; eef_abs_position, eef_abs_rotation (2cols); gravity
        # elif config['k_option']==3:
        #     # no k
        #     num_fib_in = [1,5-type1_cnt] # 16 in total, 1 type0: binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity
        # else:
        #     raise NotImplementedError(f"k_option {config['k_option']} not implemented")
        # num_fib_in = [1,5-type1_cnt] # 16 in total, 1 type0: binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity
        if config['k_option']==1:
            # 1 diffusion steps as type 0 rotation
            num_fib_in = [7,5-type1_cnt] # 22 in total, 7 type0: k1,k2; binary gripper_action 5 type1: tgt_nxyz; eef_abs_position, eef_abs_rotation (2cols); gravity
        elif config['k_option']==3:
            # no k
            num_fib_in = [1,5-type1_cnt] # 16 in total, 1 type0: binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity
        else:
            raise NotImplementedError(f"k_option {config['k_option']} not implemented")
        
        # if no_tgt_nxyz:
        #     if config['k_option']==0:
        #         # 0 diffusion steps as type 0 scalar
        #         num_fib_in = [2,4] # 14 in total, 2 type0: k; binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity
        #     elif config['k_option']==1:
        #         # 1 diffusion steps as type 0 rotation
        #         num_fib_in = [7,4] # 19 in total, 7 type0: k1,k2; binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity
        #     elif config['k_option']==2:
        #         # 2 diffusion steps as type 1 rotation
        #         num_fib_in = [1,6] # 19 in total, 1 type0: binary gripper_action 6 type1: k1,k2; eef_abs_position, eef_abs_rotation (2cols); gravity
        #     elif config['k_option']==3:
        #     # no k
        #         num_fib_in = [1,4] # 16 in total, 1 type0: binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity
        #     else:
        #         raise NotImplementedError(f"k_option {config['k_option']} not implemented")

        if fused:
            self.pos_ori_net = SE3Backbone(
                fiber_in=Fiber({
                    "0": num_fib_in[0], 
                    "1": num_fib_in[1], 
                }),
                fiber_out=Fiber({
                    "0": (6+1+1+1)*pred_horizon, # 2 cols of rotation + magnitude of offset + weights of each rot cand. + gripper open(1)/close(0)
                    "1": (1)*pred_horizon, # offset/translation (not unit direction)
                }),
                num_layers= 8,
                num_degrees= num_degrees,
                num_channels= num_channels,
                num_heads= num_heads,
                channels_div= channels_div,
                voxelize = voxelize,
                k_neighbours=k_neighbours,
                compute_gradients=config['sh_basis_compute_gradients'],
                low_memory=config['low_memory'],
            )
        else:
            self.ori_net=SE3Backbone(
                fiber_in=Fiber({
                    "0": num_fib_in[0], 
                    "1": num_fib_in[1], 
                }),
                fiber_out=Fiber({
                    "0": (6+1)*pred_horizon, # 2 cols of rotation + weights of each rot cand.
                }),
                num_layers= 8,
                num_degrees= num_degrees,
                num_channels= num_channels,
                num_heads= num_heads,
                channels_div= channels_div,
                voxelize = voxelize,
                k_neighbours=k_neighbours,
                compute_gradients=config['sh_basis_compute_gradients'],
                low_memory=config['low_memory'],
            )

            self.pos_net = SE3Backbone(
                fiber_in=Fiber({
                    "0": num_fib_in[0], 
                    "1": num_fib_in[1], 
                }),
                fiber_out=Fiber({
                    "0": (1+1)*pred_horizon, # magnitude of offset + gripper open(1)/close(0)
                    "1": (1)*pred_horizon, # offset/translation (not unit direction)
                }),
                num_layers= 8,
                num_degrees= 6,
                num_channels= 16,
                num_heads= 2,
                channels_div= 2,
                # BUG SHOULD BE:
                # num_degrees= num_degrees,
                # num_channels= num_channels,
                # num_heads= num_heads,
                # channels_div= channels_div,
                voxelize = voxelize,
                k_neighbours=k_neighbours,
                compute_gradients=config['sh_basis_compute_gradients'],
                low_memory=config['low_memory'],
            )

    def forward(self, inputs,num_point,return_raw=False,Ho_in_B=False):
        bs = inputs["xyz"].shape[0]

        if self.fused:
            pos_ori_net_features = self.pos_ori_net(inputs)["feature"]

            if Ho_in_B:
                bs=bs//num_point
                assert bs*num_point==inputs["xyz"].shape[0]
                pos_ori_net_features=torch.stack(pos_ori_net_features, dim=0).view(bs,num_point,-1)

            # process type 0 orientation + type 0 offset/translation magnitude # "0": (6+1+1+1)*pred_horizon, # 2 cols of rotation + magnitude of offset + weights of each rot cand. + gripper open(1)/close(0)
            # process type 1 offset/translation direction

            rot_type0_feature_list = []
            gripper_type0_feature_list = []
            pos_type1_feature_list = []
            for i in range(bs):
                batchi_type0_feature = pos_ori_net_features[i] # [Ho*num_point, Hp*7]
                batchi_type0_feature=batchi_type0_feature.view(batchi_type0_feature.shape[0],self.pred_horizon,-1)
                trans_mag_feature=batchi_type0_feature[:, :, 6:7]
                # mag_feature=torch.mean(mag_feature, dim=0) # [Hp, 1]
                rot_mag_feature=batchi_type0_feature[:, :, 7:8]
                gripper_feature=batchi_type0_feature[:, :, 8:9]
                gripper_feature=torch.mean(gripper_feature, dim=0) # [Hp, 1]
                rot_feature=batchi_type0_feature[:, :, :6]

                if self.config['rot_aggregation']=='mean':
                    rot_feature=torch.mean(rot_feature * rot_mag_feature, dim=0) # [Hp, 6]
                elif self.config['rot_aggregation']=='separate_mean':
                    rot_feature=torch.mean(rot_feature, dim=0) * torch.mean(rot_mag_feature, dim=0) # [Hp, 6]    
                elif self.config['rot_aggregation']=='min':
                    indices = torch.argmin(rot_mag_feature.squeeze(-1), dim=0)  # [Hp]
                    rot_feature = rot_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 6]
                elif self.config['rot_aggregation']=='abs_min':
                    indices = torch.argmin(torch.abs(rot_mag_feature).squeeze(-1), dim=0)  # [Hp]
                    rot_feature = rot_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 6]
                elif self.config['rot_aggregation']=='max':
                    indices = torch.argmax(rot_mag_feature.squeeze(-1), dim=0)  # [Hp]
                    rot_feature = rot_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 6]
                else:
                    raise NotImplementedError(f"rot_aggregation {self.config['rot_aggregation']} not implemented")
                
                rot_type0_feature_list.append(rot_feature)
                gripper_type0_feature_list.append(gripper_feature)

                batchi_type1_feature = pos_ori_net_features[i][:,(6+1+1+1)*self.pred_horizon:(6+1+1+1)*self.pred_horizon+3*(1)*self.pred_horizon] # [Ho*num_point, Hp*3]
                batchi_type1_feature=batchi_type1_feature.view(batchi_type1_feature.shape[0],self.pred_horizon,-1)
                if self.config['trans_norm']:
                    norms=batchi_type1_feature.detach().norm(dim=2,keepdim=True)
                    batchi_type1_feature=0.01*batchi_type1_feature/norms
                trans_feature=batchi_type1_feature

                if self.config['trans_aggregation']=='mean':
                    trans_feature=torch.mean(batchi_type1_feature* trans_mag_feature, dim=0) # [Hp, 3]
                elif self.config['trans_aggregation']=='separate_mean':
                    trans_feature=torch.mean(batchi_type1_feature, dim=0)* torch.mean(trans_mag_feature, dim=0) # [Hp, 3]
                elif self.config['trans_aggregation']=='min':
                    indices = torch.argmin(trans_mag_feature.squeeze(-1), dim=0)  # [Hp]
                    trans_feature = trans_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 3]
                elif self.config['trans_aggregation']=='abs_min':
                    indices = torch.argmin(torch.abs(trans_mag_feature).squeeze(-1), dim=0)  # [Hp]
                    trans_feature = trans_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 3]
                elif self.config['trans_aggregation']=='max':
                    indices = torch.argmax(trans_mag_feature.squeeze(-1), dim=0)  # [Hp]
                    trans_feature = trans_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 3]
                else:
                    raise NotImplementedError(f"trans_aggregation {self.config['trans_aggregation']} not implemented")
                
                pos_type1_feature_list.append(trans_feature)
        else:
            ori_net_features = self.ori_net(inputs)["feature"]
            pos_net_features = self.pos_net(inputs)["feature"]

            if Ho_in_B:
                bs=bs//num_point
                assert bs*num_point==inputs["xyz"].shape[0]
                ori_net_features=torch.stack(ori_net_features, dim=0).view(bs,num_point,-1)
                pos_net_features=torch.stack(pos_net_features, dim=0).view(bs,num_point,-1)

            # ori "0": (6+1)*pred_horizon, # 2 cols of rotation + magnitude of offset + weights of each rot cand.

            rot_type0_feature_list = []
            for i in range(bs):
                batchi_type0_feature = ori_net_features[i] # [Ho*num_point, Hp*7]
                batchi_type0_feature=batchi_type0_feature.view(batchi_type0_feature.shape[0],self.pred_horizon,-1)
                rot_mag_feature=batchi_type0_feature[:, :, 6:7]
                rot_feature=batchi_type0_feature[:, :, :6]

                if self.config['rot_aggregation']=='mean':
                    rot_feature=torch.mean(rot_feature * rot_mag_feature, dim=0) # [Hp, 6]
                elif self.config['rot_aggregation']=='separate_mean':
                    rot_feature=torch.mean(rot_feature, dim=0) * torch.mean(rot_mag_feature, dim=0) # [Hp, 6]    
                elif self.config['rot_aggregation']=='min':
                    indices = torch.argmin(rot_mag_feature.squeeze(-1), dim=0)  # [Hp]
                    rot_feature = rot_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 6]
                elif self.config['rot_aggregation']=='abs_min':
                    indices = torch.argmin(torch.abs(rot_mag_feature).squeeze(-1), dim=0)  # [Hp]
                    rot_feature = rot_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 6]
                elif self.config['rot_aggregation']=='max':
                    indices = torch.argmax(rot_mag_feature.squeeze(-1), dim=0)  # [Hp]
                    rot_feature = rot_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 6]
                else:
                    raise NotImplementedError(f"rot_aggregation {self.config['rot_aggregation']} not implemented")
                
                rot_type0_feature_list.append(rot_feature)


            # "0": (1+1)*pred_horizon, # magnitude of offset + gripper open(1)/close(0)
            # "1": (1)*pred_horizon, # offset/translation (not unit direction)

            gripper_type0_feature_list = []
            pos_type1_feature_list = []
            for i in range(bs):
                batchi_type0_feature = pos_net_features[i] # [Ho*num_point, Hp*2]
                batchi_type0_feature=batchi_type0_feature.view(batchi_type0_feature.shape[0],self.pred_horizon,-1)
                trans_mag_feature=batchi_type0_feature[:, :, 0:1]
                gripper_feature=batchi_type0_feature[:, :, 1:2]
                gripper_feature=torch.mean(gripper_feature, dim=0) # [Hp, 1]

                batchi_type1_feature = pos_net_features[i][:,(1+1)*self.pred_horizon:(1+1)*self.pred_horizon+3*(1)*self.pred_horizon] # [Ho*num_point, Hp*3]
                batchi_type1_feature=batchi_type1_feature.view(batchi_type1_feature.shape[0],self.pred_horizon,-1)
                trans_feature=batchi_type1_feature
                if self.config['trans_aggregation']=='mean':
                    trans_feature=torch.mean(batchi_type1_feature* trans_mag_feature, dim=0) # [Hp, 3]
                elif self.config['trans_aggregation']=='separate_mean':
                    trans_feature=torch.mean(batchi_type1_feature, dim=0)* torch.mean(trans_mag_feature, dim=0) # [Hp, 3]
                elif self.config['trans_aggregation']=='min':
                    indices = torch.argmin(trans_mag_feature.squeeze(-1), dim=0)  # [Hp]
                    trans_feature = trans_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 3]
                elif self.config['trans_aggregation']=='abs_min':
                    indices = torch.argmin(torch.abs(trans_mag_feature).squeeze(-1), dim=0)  # [Hp]
                    trans_feature = trans_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 3]
                elif self.config['trans_aggregation']=='max':
                    indices = torch.argmax(trans_mag_feature.squeeze(-1), dim=0)  # [Hp]
                    trans_feature = trans_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 3]
                else:
                    raise NotImplementedError(f"trans_aggregation {self.config['trans_aggregation']} not implemented")
                
                pos_type1_feature_list.append(trans_feature)
                gripper_type0_feature_list.append(gripper_feature)




        # TODO BUG first dim wont match if voxelized, thus cannot be stacked # [B, Hp, 3]
        output_ori = torch.stack(rot_type0_feature_list,dim = 0) # [B, Hp, 6]
        output_pos = torch.stack(pos_type1_feature_list,dim = 0) # [B, Hp, 3]
        output_gripper = torch.stack(gripper_type0_feature_list,dim = 0) # [B, Hp, 1]
        if return_raw:
            return{
                'pos':output_pos,
                'ori':output_ori,
                'gripper':output_gripper,
            }
    
        output_ori = process_action(output_ori.view(-1,6), output_pos.view(-1,3),follow_rot_trans_convention=True).view(bs,-1,4,4) # orthogonalization
        return output_ori # [B, Ho, 4, 4]







class SE3ManiNet_ori_pos_sep(ExtendedModule):
    def __init__(self, voxelize=False,k_neighbours=8,pred_horizon=8,config=None,no_tgt_nxyz=False,eef_abs_position_as_node=False):
        super().__init__()
        self.pred_horizon=pred_horizon
        self.config=config

        type1_cnt=0
        if eef_abs_position_as_node:
            type1_cnt+=1

        if no_tgt_nxyz:
            type1_cnt+=1

        # # Options
        # if config['k_option']==0:
        #     # 0 diffusion steps as type 0 scalar
        #     num_fib_in = [2,5-type1_cnt] # 17 in total, 2 type0: k; binary gripper_action 5 type1: tgt_nxyz; eef_abs_position, eef_abs_rotation (2cols); gravity
        # elif config['k_option']==1:
        #     # 1 diffusion steps as type 0 rotation
        #     num_fib_in = [7,5-type1_cnt] # 22 in total, 7 type0: k1,k2; binary gripper_action 5 type1: tgt_nxyz; eef_abs_position, eef_abs_rotation (2cols); gravity
        # elif config['k_option']==2:
        #     # 2 diffusion steps as type 1 rotation
        #     num_fib_in = [1,7-type1_cnt] # 22 in total, 1 type0: binary gripper_action 7 type1: k1,k2; tgt_nxyz; eef_abs_position, eef_abs_rotation (2cols); gravity
        # elif config['k_option']==3:
        #     # no k
        #     num_fib_in = [1,5-type1_cnt] # 16 in total, 1 type0: binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity
        # else:
        #    raise NotImplementedError(f"k_option {config['k_option']} not implemented")
        num_fib_in = [1,5-type1_cnt] # 16 in total, 1 type0: binary gripper_action 4 type1: eef_abs_position, eef_abs_rotation (2cols); gravity

        self.ori_net = SE3Backbone(
            fiber_in=Fiber({
                "0": num_fib_in[0], 
                "1": num_fib_in[1], 
            }),
            fiber_out=Fiber({
                "0": (6+1)*pred_horizon, # 2 cols of rotation + magnitude of offset + weights of each rot cand.
            }),
            num_layers= 8,
            num_degrees= 6,
            num_channels= 16,
            num_heads= 2,
            channels_div= 2,
            voxelize = voxelize,
            k_neighbours=k_neighbours,
            compute_gradients=config['sh_basis_compute_gradients'],
            low_memory=config['low_memory'],
        )


        self.pos_net = SE3Backbone(
            fiber_in=Fiber({
                "0": num_fib_in[0], 
                "1": num_fib_in[1], 
            }),
            fiber_out=Fiber({
                "0": (1)*pred_horizon, # magnitude of offset
                "1": (1)*pred_horizon, # offset/translation (not unit direction)
            }),
            num_layers= 8,
            num_degrees= 6,
            num_channels= 16,
            num_heads= 2,
            channels_div= 2,
            voxelize = voxelize,
            k_neighbours=k_neighbours,
            compute_gradients=config['sh_basis_compute_gradients'],
            low_memory=config['low_memory'],
        )

    def forward(self, inputs,num_point,return_raw=False,Ho_in_B=False):
        bs = inputs["xyz"].shape[0]
        ori_net_features = self.ori_net(inputs)["feature"]
        pos_net_features = self.pos_net(inputs)["feature"]
        
        if Ho_in_B:
            bs=bs//num_point
            assert bs*num_point==inputs["xyz"].shape[0]
            ori_net_features=torch.stack(ori_net_features, dim=0).view(bs,num_point,-1)
            pos_net_features=torch.stack(pos_net_features, dim=0).view(bs,num_point,-1)


        # ori "0": (6+1)*pred_horizon, # 2 cols of rotation + magnitude of offset + weights of each rot cand.

        type0_feature_list = []
        type1_feature_list = []
        for i in range(bs):
            batchi_type0_feature = ori_net_features[i] # [Ho*num_point, Hp*n]
            batchi_type0_feature=batchi_type0_feature.view(batchi_type0_feature.shape[0],self.pred_horizon,-1)
            rot_mag_feature=batchi_type0_feature[:, :, 6:7]
            rot_feature=batchi_type0_feature[:, :, :6]

            if self.config['rot_aggregation']=='mean':
                rot_feature=torch.mean(rot_feature * rot_mag_feature, dim=0) # [Hp, 6]
            elif self.config['rot_aggregation']=='separate_mean':
                rot_feature=torch.mean(rot_feature, dim=0) * torch.mean(rot_mag_feature, dim=0) # [Hp, 6]    
            elif self.config['rot_aggregation']=='min':
                indices = torch.argmin(rot_mag_feature.squeeze(-1), dim=0)  # [Hp]
                rot_feature = rot_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 6]
            elif self.config['rot_aggregation']=='abs_min':
                indices = torch.argmin(torch.abs(rot_mag_feature).squeeze(-1), dim=0)  # [Hp]
                rot_feature = rot_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 6]
            elif self.config['rot_aggregation']=='max':
                indices = torch.argmax(rot_mag_feature.squeeze(-1), dim=0)  # [Hp]
                rot_feature = rot_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 6]
            else:
                raise NotImplementedError(f"rot_aggregation {self.config['rot_aggregation']} not implemented")
            
            type0_feature_list.append(rot_feature)

        # pos "0": (1)*pred_horizon, # magnitude of offset
        # pos "1": (1)*pred_horizon, # offset/translation (not unit direction)
        for i in range(bs):
            batchi_type0_feature = pos_net_features["feature"][i] # [Ho*num_point, Hp*n]
            batchi_type0_feature=batchi_type0_feature.view(batchi_type0_feature.shape[0],self.pred_horizon,-1)
            trans_mag_feature=batchi_type0_feature[:, :, 0:1]

            batchi_type1_feature = pos_net_features["feature"][i][:,(1)*self.pred_horizon:(1)*self.pred_horizon+3*(1)*self.pred_horizon] # [Ho*num_point, Hp*3]
            batchi_type1_feature=batchi_type1_feature.view(batchi_type1_feature.shape[0],self.pred_horizon,-1)
            trans_feature=batchi_type1_feature
            if self.config['trans_aggregation']=='mean':
                trans_feature=torch.mean(batchi_type1_feature* trans_mag_feature, dim=0) # [Hp, 3]
            elif self.config['trans_aggregation']=='separate_mean':
                trans_feature=torch.mean(batchi_type1_feature, dim=0)* torch.mean(trans_mag_feature, dim=0) # [Hp, 3]
            elif self.config['trans_aggregation']=='min':
                indices = torch.argmin(trans_mag_feature.squeeze(-1), dim=0)  # [Hp]
                trans_feature = trans_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 3]
            elif self.config['trans_aggregation']=='abs_min':
                indices = torch.argmin(torch.abs(trans_mag_feature).squeeze(-1), dim=0)  # [Hp]
                trans_feature = trans_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 3]
            elif self.config['trans_aggregation']=='max':
                indices = torch.argmax(trans_mag_feature.squeeze(-1), dim=0)  # [Hp]
                trans_feature = trans_feature[indices, torch.arange(self.pred_horizon)]  # Shape: [Hp, 3]
            else:
                raise NotImplementedError(f"trans_aggregation {self.config['trans_aggregation']} not implemented")
            
            type1_feature_list.append(trans_feature)


        # TODO BUG first dim wont match if voxelized, thus cannot be stacked # [B, Hp, 3]
        output_ori = torch.stack(type0_feature_list,dim = 0) # [B, Hp, 6]
        output_pos = torch.stack(type1_feature_list,dim = 0) # [B, Hp, 3]

        if return_raw:
            return{
                'pos':output_pos,
                'ori':output_ori
            }
    
        output_ori = process_action(output_ori.view(-1,6), output_pos.view(-1,3),follow_rot_trans_convention=True).view(bs,-1,4,4) # orthogonalization
        return output_ori # [B, Ho, 4, 4]
    

class SE3VisionNet(ExtendedModule):
    def __init__(self,  
            input_type_1_feat=1,
            extra_input_type_1_feat=0,         
            output_type_1_feat=2,
            num_layers= 2,
            num_degrees= 6,
            num_channels= 16,
            num_heads= 2,
            channels_div= 2,
            voxelize=False,
            config=None
            ):
        super().__init__()
        self.config=config
        self.input_type_1_feat=input_type_1_feat
        self.extra_input_type_1_feat=extra_input_type_1_feat
        self.output_type_1_feat=output_type_1_feat
        k_neighbours=config['k_neighbours']
        if config['bugfix'] % 10 == 1:
            k_neighbours=config['k_neighbours*obs_horizon']

        fiber_out=Fiber({
            "0": 1, # the weights/heatmap
            "1": output_type_1_feat, # the global feature
        })
        if config['pc_inv']:
            fiber_out=Fiber({
                "0": 1+output_type_1_feat*3, # the weights/heatmap# the global feature
                #"1": , 
            })

        self.weights_net = SE3Backbone(
            fiber_in=Fiber({
                #"0": 3, # rgb
                "1": input_type_1_feat+extra_input_type_1_feat, # tgt_xyz + extras
            }),
            fiber_out=fiber_out,
            num_layers= num_layers,
            num_degrees= num_degrees,
            num_channels= num_channels,
            num_heads= num_heads,
            channels_div= channels_div,
            voxelize = voxelize,
            k_neighbours=k_neighbours, # BUG FIXED 1 should be 'k_neighbours*obs_horizon'
            compute_gradients=config['sh_basis_compute_gradients'],
        )

        
    def forward(self, inputs,return_raw=False):

        outputs = self.weights_net(inputs)        
        xyz = inputs["xyz"]
        feature = outputs["feature"]
        
        bs = len(xyz)
        n=xyz[0].shape[0]

        # process translation
        weights = []
        new_xyz = torch.zeros(bs,max(n//2,1),3).to(self.device) # only half of the origin xyzs survived
        new_feat = torch.zeros(bs,max(n//2,1),self.output_type_1_feat*3).to(self.device) # only half of the origin features survived
        
        global_feat = torch.zeros(bs,self.output_type_1_feat*3).to(self.device)
        for i in range(bs):
            batchi_feature = outputs["feature"][i] # [N, fiber_out]
            weight = torch.nn.functional.softmax(batchi_feature[:,:1].reshape(-1, 1), dim=0).squeeze() # [N]
            top_indices = torch.topk(weight, k=max(n//2,1), dim=0).indices
            new_xyz[i] = xyz[i][top_indices]
            new_feat[i] = feature[i][top_indices,1:]
            weights.append(weight)
            global_feat[i]=batchi_feature[:,1:].mean(dim=0)
        if return_raw:
            return {
                'weights':weights,
                'global_feat':global_feat 
            }
        return {'xyz':new_xyz,'feature':new_feat}, global_feat # [B,max(N//2,1),3],[B,max(N//2,1),output_type_1_feat*3], [B,output_type_1_feat*3]



class SE3VisionNet_Hierarchical(ExtendedModule):
    def __init__(
            self,
            hierarchy_layers=8,
            input_type_1_feat=1, 
            output_type_1_feat=3,            
            num_layers= 2,
            num_degrees= 6,
            num_channels= 16,
            num_heads= 8,
            channels_div= 2,
            voxelize=False,
            config=None):
        super().__init__()
        num_degrees= config['num_degrees']
        num_channels= config['num_channels']
        num_heads= config['num_heads']
        channels_div= config['channels_div']
        self.config=config
        self.hierarchy_layers=hierarchy_layers
        self.output_type_1_feat=output_type_1_feat

        weights_nets =[]
        #self.input_type_1_feat=input_type_1_feat
        for hierarchy in range(hierarchy_layers):
            weights_nets.append(SE3VisionNet(
                input_type_1_feat=input_type_1_feat, 
                extra_input_type_1_feat=output_type_1_feat if hierarchy != 0 else 0,
                output_type_1_feat=output_type_1_feat,            
                num_layers= num_layers,
                num_degrees= num_degrees,
                num_channels= num_channels,
                num_heads= num_heads,
                channels_div= channels_div,
                voxelize=voxelize,
                config=config,
            ))

        self.weights_nets = torch.nn.Sequential(*weights_nets)

    def forward(self, inputs,return_raw=False):
        bs = inputs["xyz"].shape[0]
        hierarchy_layers=self.hierarchy_layers
        weights_nets=self.weights_nets
        global_feats=torch.zeros(hierarchy_layers,bs,self.output_type_1_feat*3,device=self.device)
        
        for layer in range(hierarchy_layers):
            outputs,global_feat=weights_nets[layer](inputs,return_raw=False)
            global_feats[layer]=global_feat

            if layer == hierarchy_layers-1 and return_raw==True:
                raw=weights_nets[layer](inputs,return_raw=True)
                return raw,torch.einsum('lbf->blf',global_feats)
    

            # outputs['feature'] [B,max(N//2,1),output_type_1_feat*3]
            # global_feat [B,output_type_1_feat*3]

            _N=outputs['feature'].shape[1]
            _expanded = global_feat.unsqueeze(1).expand(-1, _N, -1)  # [B,max(N//2,1),output_type_1_feat*3]
            # print(outputs['feature'].shape)
            outputs['feature'] = torch.cat((outputs['feature'], _expanded), dim=-1) # [B,max(N//2,1), output_type_1_feat*3 + output_type_1_feat*3 ] 
            # print(layer+1,outputs['feature'].shape[-1],_expanded.shape,)
            inputs=outputs

        return torch.einsum('lbf->blf',global_feats)
        