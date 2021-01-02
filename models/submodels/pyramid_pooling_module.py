from torch import nn
import torch
from torch.nn.functional import interpolate


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = []
        for f in self.features:
            out.append(interpolate(f(x), size=x_size[2:], mode='bilinear', align_corners=True))
        out = torch.cat(out, 1)
        #print(out.size())
        return out