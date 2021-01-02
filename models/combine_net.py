import logging

import torch
from torch import nn
from torchvision import models
from torch.nn.functional import interpolate

from utils.initialization import initialize_weights
from models.block_transforms import transform_batchnorm_to_groupnorm
from models.submodels.pyramid_pooling_module import PyramidPoolingModule


def get_backbone(backbone):
    if backbone == "resnet101":
        return models.resnet101(True)
    elif backbone == "resnet50":
        return models.resnet50(True)
    elif backbone == "resnext101":
        return models.resnext101_32x8d(True)
    elif backbone == "resnest50":
        return torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    elif backbone == "resnest101":
        return torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)


class _MergeBlock(nn.Module):

    def __init__(self, in_channels, number_of_classes):
        super(_MergeBlock, self).__init__()
        self._in_channels = in_channels
        moddle_channels = in_channels // 2
        self._num_classes = number_of_classes
        layers = [nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(256, self._num_classes, kernel_size=1, stride=1)]
        self.merge = nn.Sequential(*layers)

    def forward(self, x):
        return self.merge(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(_DecoderBlock, self).__init__()
        middle_channels = int(in_channels / 2)
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]


        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class CombineNet(nn.Module):

    def __init__(self, num_classes, backbone="resnest101", use_groupnorm=True):
        super(CombineNet, self).__init__()
        self._backbone_name = backbone
        self.backbone = get_backbone(backbone)

        self.dec5 = nn.Sequential(
            *([nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2)] +
              [nn.Conv2d(512, 512, kernel_size=3, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True)])
        )
        self.dec4 = _DecoderBlock(int(1536), 128, 4)
        self.dec3 = _DecoderBlock(int(640), 128, 4)
        self.dec2 = _DecoderBlock(int(384), 64, 2)
        self.dec1 = _DecoderBlock(int(192), 64, 2)
        self.ppm1 = PyramidPoolingModule(128, 16, (1, 2, 3, 6))
        self.ppm2 = PyramidPoolingModule(256, 16, (1, 2, 3, 6))
        self.ppm3 = PyramidPoolingModule(512, 16, (1, 2, 3, 6))
        self.ppm4 = PyramidPoolingModule(1024, 16, (1, 2, 3, 6))
        self.ppm5 = PyramidPoolingModule(2048, 16, (1, 2, 3, 6))
        self.merge = _MergeBlock(384, num_classes)
        if use_groupnorm:
            logging.getLogger(self.__class__.__name__).warning("Transforming all BatchNorm layers into GroupNorm layers!")
            transform_batchnorm_to_groupnorm(self, self)
        initialize_weights(self.dec5, self.dec4, self.dec3, self.dec2, self.dec1, self.ppm1,
                           self.ppm2, self.ppm3, self.ppm4, self.ppm5, self.merge)

    def forward(self, x):
        x_size = x.size()
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        enc1 = self.backbone.relu(x)
        x = self.backbone.maxpool(enc1)

        enc2 = self.backbone.layer1(x)
        enc3 = self.backbone.layer2(enc2)
        enc4 = self.backbone.layer3(enc3)
        enc5 = self.backbone.layer4(enc4)
        ppm5 = interpolate(self.ppm5(enc5), x_size[2:], mode="bilinear", align_corners=True)
        ppm4 = interpolate(self.ppm4(enc4), x_size[2:], mode="bilinear", align_corners=True)
        ppm3 = interpolate(self.ppm3(enc3), x_size[2:], mode="bilinear", align_corners=True)
        ppm2 = interpolate(self.ppm2(enc2), x_size[2:], mode="bilinear", align_corners=True)
        ppm1 = interpolate(self.ppm1(enc1), x_size[2:], mode="bilinear", align_corners=True)

        dec5 = self.dec5(enc5)
        dec4 = self.dec4(torch.cat([enc4, dec5], 1))
        dec3 = self.dec3(torch.cat([enc3, dec4], 1))
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))
        dec1 = self.dec1(torch.cat([enc1, dec2], 1))

        return self.merge(torch.cat([dec1, ppm5, ppm4, ppm3, ppm2, ppm1], 1))

    def __str__(self):
        return "CombineNet based on {} backbone.\n".format(self._backbone_name) + super().__str__()