from typing import Any

import torch

from losses.focal_loss_2d import FocalLoss


class CombinedLoss(torch.nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        super(CombinedLoss, self)._forward_unimplemented(input)

    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        self._focal_loss_indices = config.FOCAL_LOSS_INDICES
        self._ce_loss_indices = config.CE_LOSS_INDICES
        self._focal_loss = FocalLoss(apply_nonlin=torch.sigmoid)
        self._ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, output, labels):
        return self._focal_loss(output, labels)
        # labels_layered = torch.zeros_like(output)
        # for idx in range(output.shape[1]):
        #     labels_layered[:, idx, :, :][labels == idx] = 1
        # output_focal_loss = output[:, self._focal_loss_indices, :, :]
        # labels_focal_loss = labels_layered[:, self._focal_loss_indices, :, :]
        # # output_ce_loss = output[:, self._ce_loss_indices, :, :]
        # # labels_ce_loss = labels[:, self._ce_loss_indices, :, :]
        # focal_loss = self._focal_loss(labels_layered[:, 1:, :, :], torch.sigmoid(output)[:, 1:, :, :])
        # print("sum: {}, focal_loss: {}".format(torch.sum(labels_focal_loss), focal_loss.item()))
        # return focal_loss # + self._ce_loss(output, labels)
