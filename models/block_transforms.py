import torch
import torch.nn as nn

from models.deeplabv3p.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


def transform_batchnorm_to_groupnorm(model, block):
    for name, p in block.named_children():
        if isinstance(p, torch.nn.Module):
            transform_batchnorm_to_groupnorm(model, p)
        if isinstance(p, nn.BatchNorm2d) or isinstance(p, SynchronizedBatchNorm2d):
            setattr(block, name, nn.GroupNorm(16, p.num_features))
