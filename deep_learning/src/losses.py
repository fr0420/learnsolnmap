import torch
from torch import nn
from networks.basics import LambdaLayer, Scaler
from torch import Tensor
from problems.default import SeparableHamiltonianSystem


class ScaledMSELoss(nn.Module):
    """MSE loss with scaled input and target."""
    
    def __init__(self, scaler: Scaler, reduction: str = "mean") -> None:
        super().__init__()

        self.scaler = scaler
        self.reduction: str = reduction
    
    def forward(self, input: Tensor, target: Tensor):
        return nn.functional.mse_loss(self.scaler(input), self.scaler(target), reduction=self.reduction)


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss."""
    
    def __init__(self, weight: list, reduction: str = "mean") -> None:
        super().__init__()
        
        self.register_buffer("weight", torch.tensor(weight))
        self.reduction: str = reduction
    
    def forward(self, input: Tensor, target: Tensor):

        # Ensure weight has the correct shape
        assert self.weight.shape[0] == input.shape[1], "Weight must have the same dimension as the input and target features"
        
        # Compute the weighted squared errors
        weighted_squared_error = self.weight * (input - target) ** 2  

        # Apply the reduction method
        if self.reduction == "mean": 
            return weighted_squared_error.mean()
        elif self.reduction == "sum":
            return weighted_squared_error.sum()
        else:  # No reduction, return the raw weighted squared errors
            return weighted_squared_error


class NormalizedMSELoss(nn.Module):
    """Normalized MSE loss."""
    
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        
        self.reduction: str = reduction
    
    def forward(self, input: Tensor, target: Tensor):

        # Compute the squared errors
        squared_error = (input - target) ** 2  
        
        # Compute the norm of the target vector along the feature dimension
        norm_target = torch.norm(target, dim=1, keepdim=True)
        
        # Scale each sample's squared error by the squared norm of its corresponding target vector
        normalized_error = squared_error / (norm_target**2 + 1e-10)
        
        # Apply the reduction method
        if self.reduction == "mean": 
            return normalized_error.mean()
        elif self.reduction == "sum":
            return normalized_error.sum()
        else:  # No reduction, return the raw normalized errors
            return normalized_error


class MeanEnergyNormSquaredLoss(nn.Module):
    """Mean energy norm squared loss."""

    def __init__(self, problem: SeparableHamiltonianSystem, reduction: str = "mean") -> None:
        super().__init__()

        self.transform: nn.Module = LambdaLayer(
            problem.transform_to_energy_components, 
            "transform_to_energy_components"
        )
        self.reduction: str = reduction
    
    def forward(self, input: Tensor, target: Tensor):
        input = self.transform(input)
        target = self.transform(target)
        return nn.functional.mse_loss(input, target, reduction=self.reduction)


class AnchoredMeanEnergyNormSquaredLoss(nn.Module):
    """Anchored mean energy norm squared loss."""

    def __init__(self, problem: SeparableHamiltonianSystem, reduction: str = "mean") -> None:
        super().__init__()

        self.transform: nn.Module = LambdaLayer(
            problem.transform_to_energy_components_anchored, 
            "transform_to_energy_components_anchored"
        )
        self.reduction: str = reduction
    
    def forward(self, input: Tensor, target: Tensor):
        input = self.transform(input)
        target = self.transform(target)
        return nn.functional.mse_loss(input, target, reduction=self.reduction)


if __name__ == "__main__":

    import torch

    targets = torch.randn(5, 2) * torch.arange(1, 6).view(-1, 1)
    inputs = targets + 1e-2 * torch.randn(5, 2)

    # test the losses
    weighted_mse_loss = WeightedMSELoss([1., 1.], reduction="mean")
    normalized_mse_loss = NormalizedMSELoss(reduction="mean")

    print(weighted_mse_loss(inputs, targets))
    print(normalized_mse_loss(inputs, targets))