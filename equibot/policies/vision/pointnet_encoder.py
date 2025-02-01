import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
import logging


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class PointNetEncoder(nn.Module):
    def __init__(self, num_points, per_point,h_dim=128, c_dim=128, num_layers=4,flow=False, **kwargs):
        super().__init__()

        self.h_dim = h_dim
        self.c_dim = c_dim
        self.num_layers = num_layers
        self.num_points=num_points
        self.per_point=per_point

        self.pool = meanpool

        act_func = nn.LeakyReLU(negative_slope=0.0, inplace=False)
        self.act = act_func
        _cnt=3
        if flow:
            _cnt=6
        self.conv_in = nn.Conv1d(_cnt, h_dim, kernel_size=1)
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, c_dim, kernel_size=1)
        if self.per_point is True:
            self.final_conv=nn.Conv1d(self.num_points, h_dim, kernel_size=1)

    def forward(self, x, ret_perpoint_feat=False):
        # x # ([2048, 6 or 3, 40])
        y = self.act(self.conv_in(x))
        feat_list = []
        for i in range(self.num_layers):
            y = self.act(self.layers[i](y))
            y_global = y.max(-1, keepdim=True).values
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)
        x = torch.cat(feat_list, dim=1)
        x = self.conv_out(x) # torch.Size([4096, 32, 35])
        x_global = x.max(-1).values

        if self.per_point is True:
            x=self.final_conv(torch.movedim(x, [1, 2], [2, 1]))
            x=torch.movedim(x, [1, 2], [2, 1])
        ret = {"global": x_global}
        if ret_perpoint_feat:
            ret["per_point"] = x

        return ret
