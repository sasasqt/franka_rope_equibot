# using the backbone, different encoders

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from equibot.policies.vision.vec_layers import VecLinear
from equibot.policies.vision.vec_pointnet import VecPointNet
from equibot.policies.utils.VN_transformer import VNInvariant, VNTransformer


BACKBONE_DICT = {"vn_pointnet": VecPointNet}

NORMALIZATION_METHOD = {"bn": nn.BatchNorm1d, "in": nn.InstanceNorm1d}


class SIM3Vec4Latent(nn.Module):
    """
    This encoder encode the input point cloud to 4 latents
    Now only support so3 mode
    TODO: se3 and hybrid
    """

    def __init__(
        self,
        c_dim,
        backbone_type,
        backbone_args,
        mode="so3",
        normalization_method=None,
        **kwargs
    ):
        super().__init__()
        assert mode == "so3", NotImplementedError("TODO, add se3")
        if normalization_method is not None:
            backbone_args["normalization_method"] = NORMALIZATION_METHOD[
                normalization_method
            ]

        self.backbone = BACKBONE_DICT[backbone_type](**backbone_args)
        self.fc_inv = VecLinear(c_dim, c_dim, mode=mode)

    def forward(self, pcl, ret_perpoint_feat=False,flow=False, target_norm=1.0,flow_norm=1.0):
        B, T, N, H = pcl.shape
        _pcl = pcl.view(B * T, -1, H).transpose(1, 2)  # [BT, 3 or 6, N]
        xyz=_pcl[:,:3,:] # [BT, 3, N]
        centroid = xyz.mean(-1, keepdim=True)  # B,3,1
        input_xyz = xyz - centroid
        z_scale = input_xyz.norm(dim=1).mean(-1) / target_norm  # B
        z_center = centroid.permute(0, 2, 1)

        input_xyz = input_xyz / z_scale[:, None, None]

        if flow:
            input_flow=_pcl[:,3:,:]
            z_scale = input_flow.norm(dim=1).mean(-1) / target_norm / flow_norm  # B
            input_flow = input_flow / z_scale[:, None, None]
            input_pcl=torch.cat((input_xyz,input_flow), dim=1)
        else:
            input_pcl = input_xyz
        x, x_perpoint = self.backbone(input_pcl)  # B,C,3

        z_so3 = x
        z_inv_dual, _ = self.fc_inv(x[..., None])
        z_inv_dual = z_inv_dual.squeeze(-1)
        z_inv = (z_inv_dual * z_so3).sum(-1) # c_dim, num_pts

        ret = {
            "inv": z_inv,
            "so3": z_so3,
            "scale": z_scale,
            "center": z_center,
        }

        if ret_perpoint_feat:
            ret["per_point_so3"] = x_perpoint  # [B, C, 3, N]

        return ret

def test():
    kwargs = {
        "c_dim": 5,
        "backbone_type": "vn_pointnet",
        "backbone_args": {
            "h_dim": 4,
            "c_dim": 5,
            "num_layers": 3,
            "num_points": 8,
            "knn": 3,
            "per_point": False,
            "flow": False
        }
    }

    r = [
            [0.28, -0.36, -0.89],
            [0.78, 0.62, 0],
            [0.55, -0.7, 0.45]
        ]

    r = torch.tensor(r)

    net=SIM3Vec4Latent(**kwargs)
    pc = torch.randn(7, 6,8, 3) # pc: [BS, obs_horizon, num_pts, 3]
    out=net(pc)
    inv=out["inv"]
    so3=out["so3"]
    scale=out["scale"]
    center=out["center"]


    out=net(pc@r)
    inv2=out["inv"]
    so3=out["so3"]
    scale=out["scale"]
    center=out["center"]

    #assert torch.allclose(inv,inv2,1e-3),'not inv'

    z= so3
    print(so3.shape)
    fc_z_inv=VecLinear(z.size(-2), z.size(-2))
    z_inv_dual, _ = fc_z_inv(z[..., None])#.repeat(1, 1, 1, 3))
    print(z_inv_dual.shape)
    z_inv_dual = z_inv_dual.squeeze(-1)
    print(z_inv_dual.shape)
    print((z_inv_dual* z).shape)

    z_inv = (z_inv_dual * z).sum(-1,keepdim=True).expand_as(z) # BS*obs_horizon, c_dim
    print(z_inv.shape,"?")

    action=torch.randn(7,16,12).view(7,16,-1,3)
    print(action.shape)
    action=action.view(action.size(0),-1,3)
    print(action.shape)
    ttt=VecLinear(64, 64)
    ac_inv_dual,_=ttt(action[..., None])

    print(ac_inv_dual.shape)
    ac_inv_dual = ac_inv_dual.squeeze(-1)
    print(ac_inv_dual.shape)
    ac_inv=ac_inv_dual* action
    print((ac_inv_dual* action).shape)

    _min_c=min(z.view(7,-1,3).size(-2),action.size(-2))
    _max_c=max(z.view(7,-1,3).size(-2),action.size(-2))
    
    final_resize=VecLinear(_min_c,_max_c)

    if _min_c == action.size(-2):
        resized_ac,_=final_resize(action)
        rots=torch.einsum('bhi,bhj->bij',z,resized_ac)
    else:
        resized_z,_=final_resize(z.view(7,-1,3))
        rots=torch.einsum('bhi,bhj->bij',resized_z,action)

    print("---")
    print(rots.shape)


def test_vn_invariant():
    layer = VNInvariant(64)

    coors = torch.randn(1, 32, 64, 3)

    r = [
            [0.28, -0.36, -0.89],
            [0.78, 0.62, 0],
            [0.55, -0.7, 0.45]
        ]

    r = torch.tensor(r)
    out1 = layer(coors)
    print(out1.shape)
    out2 = layer(coors @ r)

    # assert torch.allclose(out1, out2, atol = 1)

    print(out1)
    print(out2)
    print(out1-out2)


def test_equivariance(l2_dist_attn):

    model = VNTransformer(
        dim = 64,
        depth = 2,
        dim_head = 64,
        heads = 8,
        l2_dist_attn = l2_dist_attn
    )

    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    r = [
            [0.28, -0.36, -0.89],
            [0.78, 0.62, 0],
            [0.55, -0.7, 0.45]
        ]

    R = torch.tensor(r)
    out1 = model(coors @ R, mask = mask)
    out2 = model(coors, mask = mask) @ R

    assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'

if __name__ == "__main__":
    test()
