import torch
from torch import nn


class MAERMSELoss(nn.Module):
    """Balance typical errors and large errors in normalized traffic units."""

    def __init__(self, rmse_weight=0.3):
        super().__init__()
        self.rmse_weight = rmse_weight

    def forward(self, prediction, target):
        error = prediction - target
        return error.abs().mean() + self.rmse_weight * torch.sqrt(error.square().mean() + 1e-8)
