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
            num_degrees= 4,
            num_channels= 8,
            num_heads= 2,
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
            num_heads= 2,
            channels_div= 2,
            voxelize = voxelize,
        )

    def forward(self, inputs,num_point,return_raw=False,Inv=True):
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
        output_pos = torch.stack(feature_list,dim = 0) 
        output_pos=torch.mean(output_pos.view(bs,-1,num_point,3),dim=2) # [B, Horizon, 3]

        # process orientation
        feature_list = list()
        for i in range(bs):
            batchi_feature = ori_output["feature"][i] # [Horizon, 6]
            feature_list.append(batchi_feature)
        action = torch.stack(feature_list,dim = 0)
        action=torch.mean(action.view(bs,-1,num_point,6),dim=2) # [B, Horizon, 6]

        if return_raw:
            return{
                'pos':output_pos,
                'ori':action
            }
    
        action = process_action(action.view(-1,6), output_pos.view(-1,3),follow_rot_trans_convention=True).view(bs,-1,4,4) # orthogonalization
        return action # [B, Ho, 4, 4]
    


class SE3ManiNet_Fused(ExtendedModule):
    def __init__(self, voxelize=False):
        super().__init__()
        num_fib_in = [7,2] # 13 in total, 7 type0:tensor_k,noisy_ori_actions, 2 type1: noisy_trans_actions,tgt_nxyz
        num_fib_out = [6]
        self.pos_ori_net = SE3Backbone(
            fiber_in=Fiber({
                "0": num_fib_in[0], 
                "1": num_fib_in[1], 
            }),
            fiber_out=Fiber({
                "0": 3+6+1, # offset/translation (unit direction) + 2 cols of rotation + magnitude of offset
                "1": 1, # offset/translation (unit direction)
            }),
            num_layers= 4,
            num_degrees= 4,
            num_channels= 8,
            num_heads= 2,
            channels_div= 2,
            voxelize = voxelize,
        )

    def forward(self, inputs,num_point,return_raw=False,Inv=True):
        bs = inputs["xyz"].shape[0]
        pos_ori_net = self.pos_ori_net(inputs)        

        # process unit offset/translation direction
        feature_list = list()
        for i in range(bs):
            if Inv:
                batchi_feature = pos_ori_net["feature"][i][:,0:3] # [Horizon, 3]
            else: # Equiv
                batchi_feature = pos_ori_net["feature"][i][:,10:13] # [Horizon, 3]
            #actioni_raw = torch.mean(batchi_feature,dim = 0)
            feature_list.append(batchi_feature)
        output_pos = torch.stack(feature_list,dim = 0) 
        output_pos=torch.mean(output_pos.view(bs,-1,num_point,3),dim=2) # [B, Horizon, 3]

        # process orientation
        feature_list = list()
        for i in range(bs):
            batchi_feature = pos_ori_net["feature"][i][:,3:9] # [Horizon, 6]
            feature_list.append(batchi_feature)
        action = torch.stack(feature_list,dim = 0)
        action=torch.mean(action.view(bs,-1,num_point,6),dim=2) # [B, Horizon, 6]

        # process offset/translation magnitude
        feature_list = list()
        for i in range(bs):
            batchi_feature = pos_ori_net["feature"][i][:,9:10] # [Horizon, 1]
            feature_list.append(batchi_feature)
        magnitude = torch.stack(feature_list,dim = 0)
        magnitude=torch.mean(magnitude.view(bs,-1,num_point,1),dim=2) # [B, Horizon, 1]
        # # maybe normalize without backprop?
        # output_pos=torch.nn.functional.normalize(output_pos, p=2, dim=-1) * magnitude
        # norm = torch.norm(output_pos, p=2, dim=-1, keepdim=True).detach()  
        # output_pos=output_pos * magnitude / norm
        output_pos=output_pos * magnitude # magnitude consists of both inv and equiv part


        if return_raw:
            return{
                'pos':output_pos,
                'ori':action
            }
    
        action = process_action(action.view(-1,6), output_pos.view(-1,3),follow_rot_trans_convention=True).view(bs,-1,4,4) # orthogonalization
        return action # [B, Ho, 4, 4]
    


class SE3VisionNet(ExtendedModule):
    def __init__(self,  
            input_type_1_feat=1,
            extra_input_type_1_feat=0,         
            output_type_1_feat=2,
            num_layers= 2,
            num_degrees= 3,
            num_channels= 8,
            num_heads= 2,
            channels_div= 2,
            voxelize=False):
        super().__init__()
        self.input_type_1_feat=input_type_1_feat
        self.extra_input_type_1_feat=extra_input_type_1_feat
        self.output_type_1_feat=output_type_1_feat
        self.weights_net = SE3Backbone(
            fiber_in=Fiber({
                #"0": 3, # rgb
                "1": input_type_1_feat+extra_input_type_1_feat, # tgt_xyz + extras
            }),
            fiber_out=Fiber({
                "0": 1, # the weights/heatmap
                "1": output_type_1_feat, # the global feature
            }),
            num_layers= num_layers,
            num_degrees= num_degrees,
            num_channels= num_channels,
            num_heads= num_heads,
            channels_div= channels_div,
            voxelize = voxelize,
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
            new_feat[i] = feature[i][:,:1][top_indices]
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
            hierarchy_layers=20,
            input_type_1_feat=1, 
            output_type_1_feat=3,            
            num_layers= 2,
            num_degrees= 3,
            num_channels= 8,
            num_heads= 2,
            channels_div= 2,
            voxelize=False):
        super().__init__()
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
                voxelize=voxelize
            ))

        self.weights_nets = torch.nn.Sequential(*weights_nets)

    def forward(self, inputs,return_raw=False):
        bs = inputs["xyz"].shape[0]
        hierarchy_layers=self.hierarchy_layers
        weights_nets=self.weights_nets
        global_feats=torch.zeros(hierarchy_layers,bs,self.output_type_1_feat*3)
        
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
            print(outputs['feature'].shape)
            outputs['feature'] = torch.cat((outputs['feature'], _expanded), dim=-1) # [B,max(N//2,1), output_type_1_feat*3 + output_type_1_feat*3 ] 
            print(layer+1,outputs['feature'].shape[-1],_expanded.shape,)
            inputs=outputs

        return torch.einsum('lbf->blf',global_feats)
        