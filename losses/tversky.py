import torch
from torch import nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    ALPHA = 0.5
    BETA = 0.5
    GAMMA = 1
    EPS = 1e-5

    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=0.5):
        # comment out if your model contains a sigmoid or equivalent activation layer
        targets_layered = torch.zeros_like(inputs)
        for idx in range(inputs.shape[1]):
            targets_layered[:, idx, :, :][targets == idx] = 1
        inputs = F.sigmoid(inputs)
        targets = targets_layered
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth + self.EPS)
        focal_tversky = (1 - tversky + self.EPS) ** gamma

        return focal_tversky


class TverskyLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()

        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

        return 1 - tversky