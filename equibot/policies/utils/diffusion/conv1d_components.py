import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample1d(nn.Module):
    def __init__(self, dim,equivariance=False):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1,bias=not equivariance)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim,equivariance=False):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1,bias=not equivariance)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8,equivariance=False):
        super().__init__()

        if not equivariance:
            self.block = nn.Sequential(
                # Bias breaks symmetry, but is this a bad thing in unet?
                nn.Conv1d(
                    inp_channels, out_channels, kernel_size, padding=kernel_size // 2,bias=not equivariance
                ),
                # Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels),
                # Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),
            )
        else:
            # GN breaks symmetry, but is this a bad thing in unet?
            self.block = nn.Sequential(
                nn.Conv1d(
                    inp_channels, out_channels, kernel_size, padding=kernel_size // 2,bias=not equivariance
                ),
                # Rearrange('batch channels horizon -> batch channels 1 horizon'),
                # Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),
            )

    def forward(self, x):
        return self.block(x)


def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1, 256, 16))
    o = cb(x)
