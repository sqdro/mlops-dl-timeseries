"""Módulo con funciones de pérdida personalizadas para modelos de series temporales"""

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """Quantile Loss para modelos de series temporales que predicen cuantiles"""
    
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        
        losses = []
        for index, quantile in enumerate(self.quantiles):
            errors = target - preds[:, :, index]
            loss = torch.max((quantile - 1) * errors, quantile * errors)
            losses.append(loss.unsqueeze(-1))

        return torch.mean(torch.cat(losses, dim=-1))
