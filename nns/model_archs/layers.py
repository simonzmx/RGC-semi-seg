import torch
import torch.nn as nn


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        nn.init.kaiming_normal_(self.single_conv[0].weight.data)

    def forward(self, x):
        return self.single_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        nn.init.kaiming_normal_(self.double_conv[0].weight.data)
        nn.init.kaiming_normal_(self.double_conv[3].weight.data)

    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        nn.init.kaiming_normal_(self.down_conv[0].weight.data)

    def forward(self, x):
        return self.down_conv(x)


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='nearest-exact'),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        nn.init.kaiming_normal_(self.up[1].weight.data)

    def forward(self, x):
        return self.up(x)


class UpSkipConnection(nn.Module):

    def __init__(self, scale_factor):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest-exact')

    def forward(self, x1, x2=None):
        """
        :param x1: encoder feature map
        :param x2: upsampled decoder feature map
        :return: feature map after skip connection
        """
        if x2 is None:
            x = self.up(x1)
        else:
            x = torch.cat([self.up(x1), x2], dim=1)
        return x


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.conv.weight.data)

    def forward(self, x):
        return self.conv(x)
