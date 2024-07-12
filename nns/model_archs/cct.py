import numpy as np
from nns.model_archs.layers import *
from nns.feature_transforms import *
from torch.distributions.uniform import Uniform


class EncoderSimplified(nn.Module):
    def __init__(self, n_channels):
        super(EncoderSimplified, self).__init__()
        self.inc1 = (SingleConv(1, n_channels))
        self.inc2 = (SingleConv(n_channels, n_channels))
        self.conv1 = (DoubleConv(n_channels * 2, n_channels * 2))
        self.conv2 = (DoubleConv(n_channels * 2 ** 2, n_channels * 2 ** 2))
        self.conv3 = (DoubleConv(n_channels * 2 ** 3, n_channels * 2 ** 3))
        self.down1 = (DownConv(n_channels, n_channels * 2))
        self.down2 = (DownConv(n_channels * 2, n_channels * 2 ** 2))
        self.down3 = (DownConv(n_channels * 2 ** 2, n_channels * 2 ** 3))

    def forward(self, x):
        xi1 = self.inc1(x)
        xi2 = self.inc2(xi1) + xi1
        xd1 = self.down1(xi2)
        xc2 = self.conv1(xd1) + xd1
        xd2 = self.down2(xc2)
        xc3 = self.conv2(xd2) + xd2
        xd3 = self.down3(xc3)
        xc4 = self.conv3(xd3) + xd3

        return xc4


class DecoderSimplified(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(DecoderSimplified, self).__init__()
        self.up1 = (UpConv(n_channels * 2 ** 3, n_channels * 2 ** 2, 2))
        self.up2 = (UpConv(n_channels * 2 ** 2, n_channels * 2, 2))
        self.up3 = (UpConv(n_channels * 2, n_channels, 2))
        self.outc = (OutConv(n_channels, n_classes))

    def forward(self, in_features):
        xu1 = self.up1(in_features)
        xu2 = self.up2(xu1)
        xu3 = self.up3(xu2)

        logits = self.outc(xu3)
        return logits


class Encoder(nn.Module):
    def __init__(self, n_channels):
        super(Encoder, self).__init__()
        self.inc1 = (SingleConv(1, n_channels))
        self.inc2 = (SingleConv(n_channels, n_channels))
        self.conv1 = (DoubleConv(n_channels * 2, n_channels * 2))
        self.conv2 = (DoubleConv(n_channels * 2 ** 2, n_channels * 2 ** 2))
        self.conv3 = (DoubleConv(n_channels * 2 ** 3, n_channels * 2 ** 3))
        self.down1 = (DownConv(n_channels, n_channels * 2))
        self.down2 = (DownConv(n_channels * 2, n_channels * 2 ** 2))
        self.down3 = (DownConv(n_channels * 2 ** 2, n_channels * 2 ** 3))

    def forward(self, x):
        xi1 = self.inc1(x)
        xi2 = self.inc2(xi1) + xi1
        xd1 = self.down1(xi2)
        xc2 = self.conv1(xd1) + xd1
        xd2 = self.down2(xc2)
        xc3 = self.conv2(xd2) + xd2
        xd3 = self.down3(xc3)
        xc4 = self.conv3(xd3) + xd3

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
        elif xi2.isinf().any():
            print(f'Inf values first detected in xc1.')
        elif xd1.isinf().any():
            print(f'Inf values first detected in xd1.')
        elif xc2.isinf().any():
            print(f'Inf values first detected in xc2.')
        elif xd2.isinf().any():
            print(f'Inf values first detected in xd2.')
        elif xc3.isinf().any():
            print(f'Inf values first detected in xc3.')
        elif xd3.isinf().any():
            print(f'Inf values first detected in xd3.')
        elif xc4.isinf().any():
            print(f'Inf values first detected in xc4.')

        return xi2, xc2, xc3, xc4


class Decoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Decoder, self).__init__()
        self.up1 = (UpConv(n_channels * 2 ** 3, n_channels * 2 ** 2, 2 ** 3))
        self.up2 = (UpConv(n_channels * 2 ** 2, n_channels * 2, 2 ** 2))
        self.up3 = (UpConv(n_channels * 2, n_channels, 2 ** 1))
        self.outc = (OutConv(n_channels * 2 ** 3, n_classes))

    def forward(self, encoder_features):
        xu1 = self.up1(encoder_features[3])
        xu2 = self.up2(encoder_features[2])
        xu3 = self.up3(encoder_features[1])

        concat = torch.cat((xu1, xu2, xu3, encoder_features[0]), 1)
        logits = self.outc(concat)

        if xu1.isnan().any():
            print(f'Nan values first detected in xu1.')
        elif xu2.isnan().any():
            print(f'Nan values first detected in xu2.')
        elif xu3.isnan().any():
            print(f'Nan values first detected in xu3.')
        elif logits.isnan().any():
            print(f'Nan values first detected in final output.')
        elif xu1.isinf().any():
            print(f'Inf values first detected in xu1.')
        elif xu2.isinf().any():
            print(f'Inf values first detected in xu2.')
        elif xu3.isinf().any():
            print(f'Inf values first detected in xu3.')
        elif logits.isinf().any():
            print(f'Inf values first detected in final output.')

        return logits


class DropOutDecoder(nn.Module):
    def __init__(self, n_channels, n_classes, drop_rate=0.2, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout3d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.basic_decoder = Decoder(n_channels, n_classes)

    def forward(self, input_features, _):
        if type(input_features) == tuple or type(input_features) == list:
            p = [self.dropout(feature) for feature in input_features]
        else:
            p = self.dropout(input_features)
        x = self.basic_decoder(p)
        return x


class FeatureDropDecoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(FeatureDropDecoder, self).__init__()
        self.basic_decoder = Decoder(n_channels, n_classes)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, input_features, _):
        if type(input_features) == tuple or type(input_features) == list:
            p = [self.feature_dropout(feature) for feature in input_features]
        else:
            p = self.feature_dropout(input_features)
        x = self.basic_decoder(p)
        return x


class FeatureNoiseDecoder(nn.Module):
    def __init__(self, n_channels, n_classes, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        self.basic_decoder = Decoder(n_channels, n_classes)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, input_features, _):
        if type(input_features) == tuple or type(input_features) == list:
            p = [self.feature_based_noise(feature) for feature in input_features]
        else:
            p = self.feature_based_noise(input_features)
        x = self.basic_decoder(p)
        return x


class ContextMaskingDecoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ContextMaskingDecoder, self).__init__()
        self.basic_decoder = Decoder(n_channels, n_classes)

    def forward(self, input_features, output_main):
        if type(input_features) == tuple or type(input_features) == list:
            p = [guided_masking(feature, output_main, return_msk_context=True)
                 for feature in input_features]
        else:
            p = guided_masking(input_features, output_main, return_msk_context=True)
        x = self.basic_decoder(p)
        return x


class ObjectMaskingDecoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ObjectMaskingDecoder, self).__init__()
        self.basic_decoder = Decoder(n_channels, n_classes)

    def forward(self, input_features, output_main):
        if type(input_features) == tuple or type(input_features) == list:
            p = [guided_masking(feature, output_main, return_msk_context=False)
                 for feature in input_features]
        else:
            p = guided_masking(input_features, output_main, return_msk_context=False)
        x = self.basic_decoder(p)
        return x
