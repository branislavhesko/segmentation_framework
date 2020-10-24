from typing import Any

import torch

from models.focal_loss import FocalLoss


class CombinedLoss(torch.nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        super(CombinedLoss, self)._forward_unimplemented(input)

    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        self._focal_loss_indices = config.FOCAL_LOSS_INDICES
        self._ce_loss_indices = config.CE_LOSS_INDICES
        self._focal_loss = FocalLoss()
        self._ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, output, labels):
        output_focal_loss = output[:, self._focal_loss_indices, :, :]
        labels_focal_loss = labels[:, self._focal_loss_indices, :, :]
        output_ce_loss = output[:, self._ce_loss_indices, :, :]
        labels_ce_loss = labels[:, self._ce_loss_indices, :, :]
        return self._focal_loss(output_focal_loss, labels_focal_loss), self._ce_loss(output_ce_loss, labels_ce_loss)
