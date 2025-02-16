
import torch
import torch.nn as nn
import torch.nn.functional as F
from equibot.policies.vision.pfc import SeparableFiberBundleConvFC

class PointNetEncoder(nn.Module):
    def __init__(self, num_points, per_point,h_dim=128, c_dim=128, num_layers=4,flow=False, **kwargs):
        super().__init__()

        self.h_dim = h_dim
        self.c_dim = c_dim
        self.num_layers = num_layers
        self.num_points=num_points
        self.per_point=per_point



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
