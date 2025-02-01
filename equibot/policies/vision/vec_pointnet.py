import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
import logging

from equibot.policies.vision.vec_layers import VecLinear
from equibot.policies.vision.vec_layers import VecLinNormAct as VecLNA


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class VecPointNet(nn.Module):
    def __init__(
        self,
        num_points,
        per_point=False,
        flow=False,
        h_dim=128,
        c_dim=128,
        num_layers=4,
        knn=16,        
    ):
        super().__init__()

        self.h_dim = h_dim
        self.c_dim = c_dim
        self.num_layers = num_layers
        self.knn = knn
        self.num_points=num_points
        self.per_point=per_point
        self.flow=flow
        
        self.pool = meanpool

        act_func = nn.LeakyReLU(negative_slope=0.0, inplace=False)
        vnla_cfg = {"mode": "so3", "act_func": act_func}

        self.conv_in = VecLNA(3, h_dim, **vnla_cfg)
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            if i==0 and self.flow:
                self.layers.append(VecLNA(h_dim+1, h_dim, **vnla_cfg))
                self.global_layers.append(VecLNA(h_dim * 2, h_dim, **vnla_cfg))
            else:
                self.layers.append(VecLNA(h_dim, h_dim, **vnla_cfg))
                self.global_layers.append(VecLNA(h_dim * 2, h_dim, **vnla_cfg))
        self.conv_out = VecLinear(h_dim * self.num_layers, c_dim, mode="so3")
        if self.per_point is True:
            self.final_conv=VecLinear(self.num_points, h_dim, mode="so3")
        self.fc_inv = VecLinear(c_dim, 3, mode="so3")

    def get_graph_feature(self, x: torch.Tensor, k: int, knn_idx=None, cross=False):
        # x: B,C,3,N return B,C*2,3,N,K


        B, C, _, N = x.shape
        if knn_idx is None:
            # if knn_idx is not none, compute the knn by x distance; ndf use fixed knn as input topo
            _x = x.reshape(B, -1, N)
            _, knn_idx, neighbors = knn_points(
                _x.transpose(2, 1), _x.transpose(2, 1), K=k, return_nn=True
            )  # B,N,K; B,N,K; B,N,K,D
            neighbors = neighbors.reshape(B, N, k, C, 3).permute(0, -2, -1, 1, 2)
        else:  # gather from the input knn idx
            assert knn_idx.shape[-1] == k, f"input knn gather idx should have k={k}"
            neighbors = torch.gather(
                x[..., None, :].expand(-1, -1, -1, N, -1),
                dim=-1,
                index=knn_idx[:, None, None, ...].expand(-1, C, 3, -1, -1),
            )  # B,C,3,N,K
        x_padded = x[..., None].expand_as(neighbors)

        if cross:
            x_dir = F.normalize(x, dim=2)
            x_dir_padded = x_dir[..., None].expand_as(neighbors)
            cross = torch.cross(x_dir_padded, neighbors, dim=2)
            y = torch.cat([cross, neighbors - x_padded, x_padded], 1)
        else:
            y = torch.cat([neighbors - x_padded, x_padded], 1)
        return y, knn_idx  # B,C*2,3,N,K

    def forward(self, x):
        _x=x # [2048,3 or 6, 40]
        x=x[:,:3,:]
        x = x.unsqueeze(1)  # [B, 1, 3, N]

        x, knn_idx = self.get_graph_feature(x, self.knn, cross=True) # knn idx [2048, 40, 8]
        x, _ = self.conv_in(x) # torch.Size([4096, 8, 3, 35, 8])
        x = self.pool(x) # torch.Size([4096, 8, 3, 35])

        if self.flow:
            flow=_x[:,3:,:] #  # [2048,3, 40]
            flow = flow.unsqueeze(1)  # [2048, 1, 3, 40]
            x=torch.cat((x,flow),dim=1)
        y = x
        feat_list = []
        for i in range(self.num_layers):
            y, _ = self.layers[i](y)
            y_global = y.mean(-1, keepdim=True)
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y, _ = self.global_layers[i](y)
            feat_list.append(y)
        x = torch.cat(feat_list, dim=1) # torch.Size([4096, 32, 3, 35])

        x, _ = self.conv_out(x) # torch.Size([4096, 8, 3, 35])
        x_mean=x.mean(-1)
        if self.per_point is True:
            x,_=self.final_conv(torch.movedim(x, [1, 3], [3, 1]))
            x=torch.movedim(x, [1, 3], [3, 1])
        return x_mean, x
