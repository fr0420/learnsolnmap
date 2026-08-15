"""
SympNets: Intrinsic structure-preserving symplectic networks for identifying Hamiltonian systems.

Reference Paper:
    Jin, P., Zhang, Z., Zhu, A., Tang, Y., & Karniadakis, G. E. (2020).
    SympNets: Intrinsic structure-preserving symplectic networks for identifying Hamiltonian systems.
    Neural Networks, 132, 166-179.
    arXiv: https://arxiv.org/abs/2001.03750

Reference Repository:
    https://github.com/jpzxshi/sympnets
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


def _get_activation(activation):
    """
    Helper function to resolve activation function from string, nn.Module, or callable.
    """
    if isinstance(activation, str):
        act_lower = activation.lower()
        if act_lower in ["sigmoid", "sigm"]:
            return torch.sigmoid
        elif act_lower == "tanh":
            return torch.tanh
        elif act_lower == "relu":
            return torch.relu
        elif act_lower == "elu":
            return nn.functional.elu
        elif hasattr(torch, activation):
            return getattr(torch, activation)
        elif hasattr(nn.functional, activation):
            return getattr(nn.functional, activation)
        else:
            raise ValueError(f"Unknown activation function name: '{activation}'")
    elif isinstance(activation, nn.Module):
        return activation
    elif callable(activation):
        return activation
    elif activation is None:
        return torch.sigmoid
    else:
        raise TypeError(f"Unsupported activation type: {type(activation)}")


def _parse_dim(
    dim: Optional[int] = None,
    input_dim: Optional[int] = None,
    dof: Optional[int] = None,
) -> int:
    """Helper to resolve total feature dimension from input_dim, dim, or dof."""
    if input_dim is not None:
        return input_dim
    elif dim is not None:
        return dim
    elif dof is not None:
        return 2 * dof
    else:
        raise ValueError("Must specify 'dim', 'input_dim', or 'dof'.")



class LinearModule(nn.Module):
    """
    Linear symplectic module for LA-SympNet.

    Applies a composition of linear symplectic transformations parametrized by symmetric
    matrices S_i, followed by bias shifts bp, bq.
    """

    def __init__(self, dim: int, layers: int = 2):
        super(LinearModule, self).__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even, got {dim}")
        self.dim = dim
        self.layers = layers
        d = dim // 2

        self.S = nn.ParameterList(
            [nn.Parameter(torch.randn(d, d) * 0.01) for _ in range(layers)]
        )
        self.bp = nn.Parameter(torch.zeros(d))
        self.bq = nn.Parameter(torch.zeros(d))

    def forward(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        h: Union[float, torch.Tensor] = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for i in range(self.layers):
            S_i = self.S[i]
            S_sym = S_i + S_i.t()
            if i % 2 == 0:
                p = p + (q @ S_sym) * h
            else:
                q = q + (p @ S_sym) * h
        p = p + self.bp * h
        q = q + self.bq * h
        return p, q


class ActivationModule(nn.Module):
    """
    Activation symplectic module for LA-SympNet.

    Updates either p (mode='up') or q (mode='low') using elementwise non-linear activation.
    """

    def __init__(self, dim: int, activation="sigmoid", mode: str = "up"):
        super(ActivationModule, self).__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even, got {dim}")
        self.dim = dim
        self.act = _get_activation(activation)
        self.mode = mode
        d = dim // 2

        self.a = nn.Parameter(torch.randn(d) * 0.01)

    def forward(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        h: Union[float, torch.Tensor] = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "up":
            return p + self.act(q) * self.a * h, q
        elif self.mode == "low":
            return p, q + self.act(p) * self.a * h
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Expected 'up' or 'low'.")


class GradientModule(nn.Module):
    """
    Gradient symplectic module for G-SympNet.

    Applies a symplectic update derived from the gradient of a scalar neural network potential:
    gradH = (act(x @ K + b) * a) @ K^T
    """

    def __init__(
        self,
        dim: int,
        width: int = 20,
        activation="sigmoid",
        mode: str = "up",
    ):
        super(GradientModule, self).__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even, got {dim}")
        self.dim = dim
        self.width = width
        self.act = _get_activation(activation)
        self.mode = mode
        d = dim // 2

        self.K = nn.Parameter(torch.randn(d, width) * 0.01)
        self.a = nn.Parameter(torch.randn(width) * 0.01)
        self.b = nn.Parameter(torch.zeros(width))

    def forward(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        h: Union[float, torch.Tensor] = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "up":
            gradH = (self.act(q @ self.K + self.b) * self.a) @ self.K.t()
            return p + gradH * h, q
        elif self.mode == "low":
            gradH = (self.act(p @ self.K + self.b) * self.a) @ self.K.t()
            return p, q + gradH * h
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Expected 'up' or 'low'.")


class ExtendedModule(nn.Module):
    """
    Extended symplectic module for E-SympNet (for non-canonical systems with latent/control variables).
    """

    def __init__(
        self,
        dim: int,
        latent_dim: int,
        width: int = 20,
        activation="sigmoid",
        mode: str = "up",
    ):
        super(ExtendedModule, self).__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.width = width
        self.act = _get_activation(activation)
        self.mode = mode
        d = latent_dim // 2
        dc = dim - latent_dim

        self.K1 = nn.Parameter(torch.randn(d, width) * 0.01)
        self.K2 = nn.Parameter(torch.randn(dc, width) * 0.01)
        self.a = nn.Parameter(torch.randn(width) * 0.01)
        self.b = nn.Parameter(torch.zeros(width))

    def forward(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        c: torch.Tensor,
        h: Union[float, torch.Tensor] = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == "up":
            gradH = (self.act(q @ self.K1 + c @ self.K2 + self.b) * self.a) @ self.K1.t()
            return p + gradH * h, q, c
        elif self.mode == "low":
            gradH = (self.act(p @ self.K1 + c @ self.K2 + self.b) * self.a) @ self.K1.t()
            return p, q + gradH * h, c
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Expected 'up' or 'low'.")


class SympNet(nn.Module):
    """Base class for Symplectic Neural Networks."""

    def __init__(self):
        super(SympNet, self).__init__()
        self.dim = None

    def _unpack_inputs(
        self, x: Union[torch.Tensor, Tuple, List], h: Optional[Union[float, torch.Tensor]] = None
    ):
        if isinstance(x, (tuple, list)):
            x_tensor, h_tensor = x[0], x[1]
            if h is None:
                h = h_tensor
            x = x_tensor
        if h is None:
            h = 1.0
        return x, h

    def predict(
        self,
        x: Union[torch.Tensor, Tuple, List],
        steps: int = 1,
        keep_init: bool = False,
    ) -> torch.Tensor:
        """Sequential multi-step rollouts using learned symplectic map."""
        x_curr, h = self._unpack_inputs(x)
        preds = [x_curr] if keep_init else []
        for _ in range(steps):
            x_curr = self(x_curr, h=h)
            preds.append(x_curr)
        return torch.stack(preds, dim=1)


class LASympNet(SympNet):
    """
    LA-SympNet (Linear-Activation Symplectic Network).

    Composed of alternating Linear symplectic modules and Activation symplectic modules.

    Args:
        dim: Total feature dimension (must be even). Alternatively supply input_dim or dof.
        input_dim: Alias for dim.
        dof: Degrees of freedom (dim = 2 * dof).
        layers: Number of linear layers (will create layers LinearModules and layers-1 ActivationModules).
        sublayers: Number of internal linear sub-transformations per LinearModule.
        activation: Activation function ('sigmoid', 'tanh', 'elu', 'relu', or nn.Module).
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        dof: Optional[int] = None,
        layers: int = 3,
        sublayers: int = 2,
        activation="sigmoid",
    ):
        super(LASympNet, self).__init__()
        self.dim = _parse_dim(dim, input_dim, dof)
        if self.dim % 2 != 0:
            raise ValueError(f"Dimension must be even, got {self.dim}")
        self.layers = layers
        self.sublayers = sublayers
        self.activation = activation

        self.lin_modules = nn.ModuleList(
            [LinearModule(self.dim, self.sublayers) for _ in range(layers)]
        )
        self.act_modules = nn.ModuleList(
            [
                ActivationModule(
                    self.dim, activation, mode="up" if i % 2 == 0 else "low"
                )
                for i in range(layers - 1)
            ]
        )

    def forward(
        self,
        x: Union[torch.Tensor, Tuple, List],
        h: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x, h = self._unpack_inputs(x, h)
        d = self.dim // 2
        p, q = x[..., :d], x[..., d:]

        for lin_m, act_m in zip(self.lin_modules[:-1], self.act_modules):
            p, q = lin_m(p, q, h)
            p, q = act_m(p, q, h)

        p, q = self.lin_modules[-1](p, q, h)
        return torch.cat([p, q], dim=-1)


class GSympNet(SympNet):
    """
    G-SympNet (Gradient Symplectic Network).

    Composed of stacked Gradient symplectic modules (derived from neural network scalar potentials).

    Args:
        dim: Total feature dimension (must be even). Alternatively supply input_dim or dof.
        input_dim: Alias for dim.
        dof: Degrees of freedom (dim = 2 * dof).
        layers: Number of GradientModules to stack.
        width: Hidden width of the scalar potential within each GradientModule.
        activation: Activation function ('sigmoid', 'tanh', 'elu', 'relu', or nn.Module).
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        dof: Optional[int] = None,
        layers: int = 3,
        width: int = 20,
        activation="sigmoid",
    ):
        super(GSympNet, self).__init__()
        self.dim = _parse_dim(dim, input_dim, dof)
        if self.dim % 2 != 0:
            raise ValueError(f"Dimension must be even, got {self.dim}")
        self.layers = layers
        self.width = width
        self.activation = activation

        self.grad_modules = nn.ModuleList(
            [
                GradientModule(
                    self.dim, width, activation, mode="up" if i % 2 == 0 else "low"
                )
                for i in range(layers)
            ]
        )

    def forward(
        self,
        x: Union[torch.Tensor, Tuple, List],
        h: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x, h = self._unpack_inputs(x, h)
        d = self.dim // 2
        p, q = x[..., :d], x[..., d:]

        for grad_m in self.grad_modules:
            p, q = grad_m(p, q, h)

        return torch.cat([p, q], dim=-1)


class ESympNet(SympNet):
    """
    E-SympNet (Extended Symplectic Network).

    Designed for non-canonical Hamiltonian systems or systems with control/parameter variables.

    Args:
        dim: Total feature dimension (canonical phase space + control/parameter dimensions).
        latent_dim: Latent canonical phase space dimension (must be even).
        layers: Number of ExtendedModules to stack.
        width: Hidden width of potential within each ExtendedModule.
        activation: Activation function.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        latent_dim: int = 2,
        layers: int = 3,
        width: int = 20,
        activation="sigmoid",
    ):
        super(ESympNet, self).__init__()
        self.dim = _parse_dim(dim, input_dim, None)
        self.latent_dim = latent_dim
        if self.latent_dim % 2 != 0:
            raise ValueError(f"Latent dimension must be even, got {self.latent_dim}")
        self.layers = layers
        self.width = width
        self.activation = activation

        self.ext_modules = nn.ModuleList(
            [
                ExtendedModule(
                    self.dim,
                    latent_dim,
                    width,
                    activation,
                    mode="up" if i % 2 == 0 else "low",
                )
                for i in range(layers)
            ]
        )

    def forward(
        self,
        x: Union[torch.Tensor, Tuple, List],
        h: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x, h = self._unpack_inputs(x, h)
        d = self.latent_dim // 2
        p = x[..., :d]
        q = x[..., d : 2 * d]
        c = x[..., 2 * d :]

        for ext_m in self.ext_modules:
            p, q, c = ext_m(p, q, c, h)

        return torch.cat([p, q, c], dim=-1)
