from typing import List, Optional

import hydra
from omegaconf import DictConfig
import torch
from torch import nn

from modules.solnmap import BaseSolutionMap
from modules.fixed_timestep import FixedStepSolutionMap


class PeriodicEncoding(nn.Module):
    """Periodic encoding layer defined by a set of frequencies and phases."""
    def __init__(self, periods):
        super(PeriodicEncoding, self).__init__()
        initial_frequencies = 2 * torch.pi / torch.tensor(periods)
        self.log_frequencies = nn.Parameter(torch.log(initial_frequencies))
        self.phases = nn.Parameter(torch.zeros(len(periods)))
        
    def forward(self, t):
        frequencies = torch.exp(self.log_frequencies)
        omega_t = t * frequencies + self.phases
        sin_enc = torch.sin(omega_t)
        cos_enc = torch.cos(omega_t)
        encoding = torch.cat([sin_enc, cos_enc], dim=-1)
        return encoding


class Time2VecEncoding(nn.Module):
    """Time2Vec encoding layer."""
    def __init__(self, output_dim: int, periodic_activation: str = "sin", 
                 min_period: float = 1.0, max_period: float = 1000.0):
        super(Time2VecEncoding, self).__init__()
        self.periodic_activation = {"sin": torch.sin, "cos": torch.cos}[periodic_activation]
        self.linear_term = nn.Linear(1, 1)  # linear term (omega_0 * t + b_0)
        self.log_frequencies = nn.Parameter(torch.zeros(output_dim - 1))  # omega_i in periodic terms (omega_i * t + b_i)
        self.phases = nn.Parameter(torch.zeros(output_dim - 1))  # b_i in periodic terms (omega_i * t + b_i)
        self._initialize_frequencies(min_period, max_period)

    def _initialize_frequencies(self, min_period, max_period):
        # Generate logarithmically spaced frequencies
        min_freq, max_freq = 2 * torch.pi / max_period, 2 * torch.pi / min_period
        initial_frequencies = torch.logspace(
            torch.log10(torch.tensor(min_freq)),
            torch.log10(torch.tensor(max_freq)),
            steps=self.log_frequencies.shape[0],
        )
        self.log_frequencies.data = torch.log(initial_frequencies)

    def forward(self, t):
        linear = self.linear_term(t)  # shape: (bs, 1)
        periodic = t * torch.exp(self.log_frequencies) + self.phases  # shape: (bs, output_dim - 1)
        periodic = self.periodic_activation(periodic)
        encoding = torch.cat([linear, periodic], dim=-1)  # shape: (bs, output_dim)
        return encoding


class IdentityEnforcedSolutionMap(BaseSolutionMap):
    """
    Solution map enforcing the identity function at t = 0.

    Definition:
        Phi(u0, t) = u0 + scale * mult_act(multi_w * t) * net([u0, t])
    """

    def __init__(
        self, 
        network: DictConfig, 
        multiplier_activation: Optional[DictConfig] = None,
        temporal_encoding: Optional[DictConfig] = None,
        **kwargs
    ) -> None:
        super(IdentityEnforcedSolutionMap, self).__init__(**kwargs)
        
        self.save_hyperparameters(logger=False)
        
        if multiplier_activation is None:
            multiplier_activation = {"_target_": "torch.nn.Identity"}
        
        self.net = hydra.utils.instantiate(network)
        self.mult_act = hydra.utils.instantiate(multiplier_activation)
        self.mult_w = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.scale = nn.Parameter(torch.tensor(1e-2), requires_grad=True)
        self.temp_enc = hydra.utils.instantiate(temporal_encoding) if temporal_encoding else None

        if self.weight_init is not None:
            self._init_weights()

    def forward(self, u0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # u0 shape: (bs, 2*dof)
        # t shape: (bs, 1)
        # out shape: (bs, 2*dof)

        u0 = self._apply_nondim(u0)

        if self.temp_enc is not None:
            t_in = self.temp_enc(t)
        else:
            t_in = t

        out = self.net(torch.cat([u0, t_in], dim=-1))
        mult = self.mult_w * t
        mult = self.mult_act(mult)
        out = u0 + self.scale * mult * out

        out = self._apply_dim(out)

        return out


class T0CenteredSolutionMap(BaseSolutionMap):
    """
    Solution map with the timestep `t` centered around a reference time `T0`.

    Definition:
        Phi(u0, t) = net_T0(u0) + scale * mult_act(multi_w * (t-T0)) * net_res([u0, t-T0])  
    """

    def __init__(
        self,
        T0: float,
        net_T0: DictConfig,
        net_residual: DictConfig,
        multiplier_activation: Optional[DictConfig] = None,
        temporal_encoding: Optional[DictConfig] = None,
        **kwargs
    ) -> None:
        super(T0CenteredSolutionMap, self).__init__(**kwargs)

        self.save_hyperparameters(logger=False)

        if multiplier_activation is None:
            multiplier_activation = {"_target_": "torch.nn.Identity"}

        self.register_buffer("T0", torch.tensor(T0))

        if net_T0["_target_"] in [
            "modules.fixed_timestep.FixedStepSolutionMap",
            "modules.variable_timestep.IdentityEnforcedSolutionMap",
            "modules.variable_timestep.T0CenteredSolutionMap"]:
            self.net_T0 = hydra.utils.instantiate(net_T0, _recursive_=False)
        else:
            self.net_T0 = hydra.utils.instantiate(net_T0)
        self.net_res = hydra.utils.instantiate(net_residual)
        self.mult_act = hydra.utils.instantiate(multiplier_activation)
        self.mult_w = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.scale = nn.Parameter(torch.tensor(1e-2), requires_grad=True)

        self.temp_enc = hydra.utils.instantiate(temporal_encoding) if temporal_encoding else None

        if self.weight_init is not None:
            self._init_weights()

    def forward(self, u0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # u0 shape: (bs, 2*dof)
        # t shape: (bs, 1)
        # out shape: (bs, 2*dof)

        u0 = self._apply_nondim(u0)

        if self.temp_enc is not None:
            t_in = self.temp_enc(t - self.T0)
        else:
            t_in = t - self.T0
        res_in = torch.cat([u0, t_in], dim=-1)
        res_out = self.net_res(res_in)

        mult = self.mult_w * (t - self.T0)
        mult = self.mult_act(mult)

        net_T0_out = self.evaluate_net_T0(u0)
        out = net_T0_out + self.scale * mult * res_out

        out = self._apply_dim(out)

        return out

    def evaluate_net_T0(self, u0_nd: torch.Tensor) -> torch.Tensor:
        if isinstance(self.net_T0, T0CenteredSolutionMap) or isinstance(self.net_T0, IdentityEnforcedSolutionMap):
            u0 = self._apply_dim(u0_nd)
            t = torch.ones(u0.shape[0], 1).to(u0) * self.T0
            out = self.net_T0(u0, t)
            out_nd = self._apply_nondim(out)
            return out_nd
        elif isinstance(self.net_T0, FixedStepSolutionMap):
            u0 = self._apply_dim(u0_nd)
            out = self.net_T0(u0, None, 2)[1]
            out_nd = self._apply_nondim(out)
            return out_nd
        else:
            return self.net_T0(u0_nd)

    def extra_repr(self):
        return super(T0CenteredSolutionMap, self).extra_repr() + f"\nT0: {self.T0.item()}"
    
    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: str,
        pretrained_class: str,
        T0: float,
        net_residual: DictConfig,
        multiplier_activation: Optional[DictConfig] = None,
        temporal_encoding: Optional[DictConfig] = None,
        pretrained_frozen: bool = True,
        **kwargs
    ) -> "T0CenteredSolutionMap":

        ckpt = torch.load(ckpt_path)
        config = ckpt["hyper_parameters"]
        state_dict = ckpt["state_dict"]
        config["_target_"] = pretrained_class

        solnmap = cls(
            T0=T0,
            net_T0=config,
            net_residual=net_residual,
            multiplier_activation=multiplier_activation,
            temporal_encoding=temporal_encoding,
            **kwargs
        )
        solnmap.net_T0.load_state_dict(state_dict, strict=False)

        if isinstance(solnmap.net_T0, FixedStepSolutionMap):
            if solnmap.net_T0.T0 != T0:
                raise ValueError("T0 must match the fixed timestep of the pretrained model.")

        if pretrained_frozen:
            for param in solnmap.net_T0.parameters():
                param.requires_grad = False

        return solnmap
    
