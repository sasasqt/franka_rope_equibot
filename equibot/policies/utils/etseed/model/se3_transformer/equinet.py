import torch
import os
from .se3_backbone import SE3Backbone, ExtendedModule
from .se3_transformer.model.fiber import Fiber
from ...utils.group_utils import process_action, orthogonalization

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

    def forward(self, inputs,return_raw=False):
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

    def forward(self, inputs,return_raw=False):
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
    



class SE3ManiNet_Fused_Separate(ExtendedModule):
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

    def forward(self, inputs,return_raw=False,Inv=True):
        bs = inputs["xyz"].shape[0] 
        pos_output = self.pos_net(inputs)        
        ori_output = self.ori_net(inputs)

        # process translation
        feature_list = list()
        for i in range(bs):
            if Inv:
                batchi_feature = pos_output["feature"][i][:,:3] # [Horizon, 3]
            else: # Equiv
                batchi_feature = pos_output["feature"][i][:,3:] # [Horizon, 3]

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
    
