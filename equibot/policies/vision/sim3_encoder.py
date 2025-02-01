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
        z_inv = (z_inv_dual * z_so3).sum(-1)

        ret = {
            "inv": z_inv,
            "so3": z_so3,
            "scale": z_scale,
            "center": z_center,
        }

        if ret_perpoint_feat:
            ret["per_point_so3"] = x_perpoint  # [B, C, 3, N]

        return ret
