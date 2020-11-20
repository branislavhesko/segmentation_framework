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
        return torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True, dilated=True)
    elif backbone == "resnest101":
        return torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True, dilated=True)


class _MergeBlock(nn.Module):

    def __init__(self, in_channels, number_of_classes):
        super(_MergeBlock, self).__init__()
        self._in_channels = in_channels
        self._num_classes = number_of_classes
        layers = [nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(256, self._num_classes, kernel_size=1, stride=1)]
        self.merge = nn.Sequential(*layers)

    def forward(self, x):
        return self.merge(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class CombineNet(nn.Module):

    def __init__(self, num_classes, backbone="resnest50", use_groupnorm=True):
        super(CombineNet, self).__init__()
        self._backbone_name = backbone
        self.backbone = get_backbone(backbone)
        self.backbone.fc = None

        self.dec5 = nn.Sequential(
            *([nn.Conv2d(2048, 256, kernel_size=3, padding=1),
               nn.BatchNorm2d(256),
               nn.ReLU(inplace=True)]))
        self.dec4 = _DecoderBlock(int(1280), 256)
        self.dec3 = _DecoderBlock(int(768), 256)
        self.dec2 = _DecoderBlock(int(512), 256)
        self.dec1 = _DecoderBlock(int(384), 256)
        self.ppm5 = PyramidPoolingModule(2048, 64, (1, 2, 3, 6))
        self.merge = _MergeBlock(768, num_classes)
        if use_groupnorm:
            print("Transforming all BatchNorm layers into GroupNorm layers!")
            transform_batchnorm_to_groupnorm(self, self)
        initialize_weights(self.dec5, self.dec4, self.dec3, self.dec2, self.dec1, self.ppm5, self.merge)

    def forward(self, x):
        x_size = x.size()
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        enc2 = self.backbone.layer1(x)
        enc3 = self.backbone.layer2(enc2)
        enc4 = self.backbone.layer3(enc3)
        enc5 = self.backbone.layer4(enc4)
        ppm5 = interpolate(self.ppm5(enc5), scale_factor=4., mode="bilinear", align_corners=True)

        dec5 = self.dec5(enc5)
        dec4 = self.dec4(torch.cat([enc4, dec5], 1))
        dec3 = self.dec3(torch.cat([interpolate(enc3, scale_factor=2.), dec4], 1))

        return interpolate(self.merge(torch.cat([interpolate(enc2, scale_factor=2.), dec3, ppm5], 1)), scale_factor=2.)

    def __str__(self):
        return "CombineNet based on {} backbone.\n".format(self._backbone_name) + super().__str__()


if __name__ == "__main__":
    net = CombineNet(2).eval()
    net(torch.rand(1, 3, 512, 512))