# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py
#
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from easydict import EasyDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class ConvBlock(torch.nn.Module):
    """Layer to perform a convolution followed by ELU."""
    def __init__(self, in_channels, out_channels, bn=False, dropout=0.0):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            Conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.ELU(inplace=True),
            # Pay attention: 2d version of dropout is used
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity())

    def forward(self, x):
        out = self.block(x)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input with 3x3 kernels."""
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample(x):
    """Upsample input tensor by a factor of 2."""
    return F.interpolate(x, scale_factor=2, mode="nearest")


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, in_channels=3):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, in_channels=3):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        in_channels (int): Number of input channels
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, in_channels=in_channels)

    if pretrained:
        print('loading imagnet weights on resnet...')
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        # loaded['conv1.weight'] = torch.cat(
        #     (loaded['conv1.weight'], loaded['conv1.weight']), 1)
        # diff = model.load_state_dict(loaded, strict=False)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, in_channels=3):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if in_channels > 3:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, in_channels)
        else:
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = resnets[num_layers](weights=weights)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, x):
        self.features = []

        # input_image, normals = xx
        # x = (input_image - 0.45) / 0.225
        # x = torch.cat((input_image, normals),1)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class Decoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True,
        kaiming_init=False, return_feats=False):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.return_feats = return_feats

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # for s in self.scales:
        self.convs[("dispconv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        # self.sigmoid = nn.Sigmoid()

        if kaiming_init:
            print('init weights of decoder')
            for m in self.children():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0.01)

    def forward(self, input_features):
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

        # assert self.scales[0] == 0
        final_conv = self.convs[("dispconv", 0)]
        out = final_conv(x)

        if self.return_feats:
            return out, input_features[-1]
        return out

class MultiHeadDecoder(nn.Module):
    def __init__(self, num_ch_enc, tasks, return_feats, use_skips):
        super().__init__()
        self.decoders = nn.ModuleDict({k:
            Decoder(num_ch_enc=num_ch_enc,
                    num_output_channels=num_ch,
                    scales=[0],
                    kaiming_init=False,
                    use_skips=use_skips,
                    return_feats=return_feats)
            for k, num_ch in tasks.items()})

    def forward(self, x):
        y = EasyDict({k: v(x) for k, v in self.decoders.items()})
        return y

class DenseMTL(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        return self.decoder(self.encoder(x))