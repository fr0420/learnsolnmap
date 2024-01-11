from torch import nn
from networks.basics import LambdaLayer

from torch import Tensor
from problems import SeparableHamiltonianSystem


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss"""
    
    def __init__(self, weight: Tensor, reduction: str = "mean") -> None:
        super().__init__()
        self.register_buffer('weight', weight)
        self.weight: Tensor
        self.reduction: str = reduction
    
    def forward(self, input: Tensor, target: Tensor):
        return nn.functional.mse_loss(self.weight*input, self.weight*target, reduction=self.reduction)

    
class MeanEnergyNormSquaredLoss(nn.Module):
    """Mean energy norm squared loss"""

    def __init__(self, problem: SeparableHamiltonianSystem, reduction: str = "mean") -> None:
        super().__init__()

        self.transform: nn.Module = LambdaLayer(
            lambda u: problem.transform_to_energy_components(*u.chunk(2, dim=-1)), 
            "transform_to_energy_components"
        )
        self.reduction: str = reduction
    
    def forward(self, input: Tensor, target: Tensor):
        input = self.transform(input)
        target = self.transform(target)
        return nn.functional.mse_loss(input, target, reduction=self.reduction)
