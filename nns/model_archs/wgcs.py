from nns.model_archs.layers import *


class WeakGCSeg(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(WeakGCSeg, self).__init__()
        self.n_classes = n_classes

        self.inc1 = (SingleConv(1, n_channels))
        self.inc2 = (SingleConv(n_channels, n_channels))
        self.conv1 = (DoubleConv(n_channels * 2, n_channels * 2))
        self.conv2 = (DoubleConv(n_channels * 2 ** 2, n_channels * 2 ** 2))
        self.conv3 = (DoubleConv(n_channels * 2 ** 3, n_channels * 2 ** 3))
        self.down1 = (DownConv(n_channels, out_channels=n_channels * 2))
        self.down2 = (DownConv(n_channels * 2, out_channels=n_channels * 2 ** 2))
        self.down3 = (DownConv(n_channels * 2 ** 2, out_channels=n_channels * 2 ** 3))
        self.up1 = (UpConv(n_channels * 2 ** 3, n_channels * 2 ** 2, 2 ** 3))
        self.up2 = (UpConv(n_channels * 2 ** 2, n_channels * 2, 2 ** 2))
        self.up3 = (UpConv(n_channels * 2, n_channels, 2 ** 1))
        self.outc = (OutConv(n_channels * 2 ** 3, n_classes))

    def forward(self, x):
        """ Encoder"""
        xi1 = self.inc1(x)
        xi2 = self.inc2(xi1) + xi1
        xd1 = self.down1(xi2)
        xc2 = self.conv1(xd1) + xd1
        xd2 = self.down2(xc2)
        xc3 = self.conv2(xd2) + xd2
        xd3 = self.down3(xc3)
        xc4 = self.conv3(xd3) + xd3

        """ Decoder """
        xu1 = self.up1(xc4)
        xu2 = self.up2(xc3)
        xu3 = self.up3(xc2)

        """ Concatenation """
        concat = torch.cat((xu1, xu2, xu3, xi2), 1)

        """ Output """
        logits = self.outc(concat)

        # check nan values
        if xi2.isnan().any():
            print(f'Nan values first detected in xc1.')
        elif xd1.isnan().any():
            print(f'Nan values first detected in xd1.')
        elif xc2.isnan().any():
            print(f'Nan values first detected in xc2.')
        elif xd2.isnan().any():
            print(f'Nan values first detected in xd2.')
        elif xc3.isnan().any():
            print(f'Nan values first detected in xc3.')
        elif xd3.isnan().any():
            print(f'Nan values first detected in xd3.')
        elif xc4.isnan().any():
            print(f'Nan values first detected in xc4.')
        elif xu1.isnan().any():
            print(f'Nan values first detected in xu1.')
        elif xu2.isnan().any():
            print(f'Nan values first detected in xu2.')
        elif xu3.isnan().any():
            print(f'Nan values first detected in xu3.')
        elif logits.isnan().any():
            print(f'Nan values first detected in final output.')

        return logits


if __name__ == '__main__':
    from torchsummary import summary
    net = WeakGCSeg(32, 2)
    summary(net.to('cuda'), (1, 32, 128, 128))
