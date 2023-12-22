import torch
from torch import nn
from problems import FPU, LennardJones


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss"""
    
    def __init__(self, weights, reduction="mean"):
        super().__init__()
        self.weights = weights
        self.reduction = reduction
    
    def forward(self, input, target):
        self.weights = self.weights.to(input)
        return nn.functional.mse_loss(self.weights*input, self.weights*target, reduction=self.reduction)

    
class MeanEnergyNormSquaredLoss(nn.Module):
    """Mean energy norm squared loss"""

    def __init__(self, problem, problem_kwargs, reduction="mean"):
        super().__init__()
        if problem == "fpu":
            fpu = FPU(**problem_kwargs)
            self.Lambda = lambda u: fpu.Lambda_transform(u[:, :6], u[:, 6:])
        else:
            self.Lambda = None
        self.reduction = reduction
    
    def forward(self, input, target):
        input = self.Lambda(input)
        target = self.Lambda(target)
        return nn.functional.mse_loss(input, target, reduction=self.reduction)
    

class AnchoredMeanEnergyNormSquaredLoss(nn.Module):
    """Anchored mean energy norm squared loss"""

    def __init__(self, problem, problem_kwargs, reduction='mean'):
        super().__init__()
        if problem == "fpu":
            fpu = FPU(**problem_kwargs)
            self.Lambda = lambda u: fpu.Lambda2_transform(u[:, :6], u[:, 6:])
        else:
            self.Lambda = None
        self.reduction = reduction
    
    def forward(self, input, target):
        input = self.Lambda(input)
        target = self.Lambda(target)
        return nn.functional.mse_loss(input, target, reduction=self.reduction)


def get_loss_fn(loss_fn, **kwargs):
    """Define loss function"""
    
    return {
        'MSELoss': nn.MSELoss,
        'WeightedMSELoss': WeightedMSELoss,
        'MeanEnergyNormSquaredLoss': MeanEnergyNormSquaredLoss,
        'AnchoredMeanEnergyNormSquaredLoss': AnchoredMeanEnergyNormSquaredLoss,
        }[loss_fn](**kwargs)

