import torch


class FocalLoss(torch.nn.Module):

    def __init__(self, alfa=2., beta=4.):
        super().__init__()
        self._alfa = alfa
        self._beta = beta

    def forward(self, labels, output):
        loss_point = torch.mean((1 - output[
            labels == 1.]) ** self._alfa * torch.log(output[labels == 1.] + 1e-7))
        loss_background = torch.mean((1 - labels) ** self._beta * output ** self._alfa * torch.log(1 - output + 1e-7))
        return -1 * (loss_point + 50 * loss_background)
