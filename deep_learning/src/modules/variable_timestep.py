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


class FeatureNormalization(nn.Module):
    """Feature normalization layer."""
    def __init__(self, means: List[float], stds: List[float]):
        super(FeatureNormalization, self).__init__()
        if len(means) != len(stds):
            raise ValueError("Number of means and standard deviations must match.")
        if any(std <= 0 for std in stds):
            raise ValueError("Standard deviations must be positive.")
        self.register_buffer("means", torch.tensor(means))
        self.register_buffer("stds", torch.tensor(stds))

    def forward(self, x):
        return (x - self.means) / self.stds
    
    def inverse(self, x):
        return x * self.stds + self.means


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
        feature_normalization: Optional[DictConfig] = None,
        **kwargs
    ) -> None:
        super(IdentityEnforcedSolutionMap, self).__init__(**kwargs)
        
        self.save_hyperparameters(logger=False)
        
        if multiplier_activation is None:
            multiplier_activation = {"_target_": "torch.nn.Identity"}
        
        self.net = hydra.utils.instantiate(network)
        self.mult_act = hydra.utils.instantiate(multiplier_activation)
        self.mult_w = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.scale = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.temp_enc = hydra.utils.instantiate(temporal_encoding) if temporal_encoding else None
        self.feat_norm = hydra.utils.instantiate(feature_normalization) if feature_normalization else None
    
        if self.weight_init is not None:
            self._init_weights()

    def forward(self, u0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # u0 shape: (bs, 2*dof)
        # t shape: (bs, 1)
        # out shape: (bs, 2*dof)

        u0 = self._apply_nondim(u0)

        t_in = self.temp_enc(t) if self.temp_enc is not None else t
        u0_in = self.feat_norm(u0) if self.feat_norm is not None else u0
        out = self.net(torch.cat([u0_in, t_in], dim=-1))
        if self.feat_norm is not None:
            out = self.feat_norm.inverse(out)

        mult = self.mult_w * t
        mult = self.mult_act(mult)
        out = u0 + self.scale * mult * out

        out = self._apply_dim(out)

        return out


class T0CenteredSolutionMap(BaseSolutionMap):
    """
    Solution map with the timestep `t` centered around a reference time `T0`.

    Definition:
        Phi(u0, t) = net_T0(u0) + scale * mult_act(multi_w * (t-T0)) * net_res([net_T0(u0), t-T0])  
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
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.temp_enc = hydra.utils.instantiate(temporal_encoding) if temporal_encoding else None

        if self.weight_init is not None:
            self._init_weights()

    def forward(self, u0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # u0 shape: (bs, 2*dof)
        # t shape: (bs, 1)
        # out shape: (bs, 2*dof)

        u0 = self._apply_nondim(u0)

        net_T0_out = self.evaluate_net_T0(u0)

        if self.temp_enc is not None:
            t_in = self.temp_enc(t - self.T0)
        else:
            t_in = t - self.T0
        res_in = torch.cat([net_T0_out, t_in], dim=-1)
        res_out = self.net_res(res_in)

        mult = self.mult_w * (t - self.T0)
        mult = self.mult_act(mult)
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
    

class StackedSolutionMap(BaseSolutionMap):
    """
    Stacked solution map.

    Definition:
        Let Phi_k(u0, t) be the solution map for t_k <= t <= t_{k+1}. Then Phi(u0, t) is defined recursively as
        1. Phi_0(u0, t) = u0 + F(u0, t)
        2. Phi_k(u0, t) = Phi_{k-1}(u0, t_k) + F(Phi_{k-1}(u0, t_k), t-t_k) for k = 1, 2, ...
    """

    def __init__(
        self,
        time_points: List[float],
        network: DictConfig,
        multiplier_activation: Optional[DictConfig] = None,
        temporal_encoding: Optional[DictConfig] = None,
        feature_normalization: Optional[DictConfig] = None,
        **kwargs
    ) -> None:
        super(StackedSolutionMap, self).__init__(**kwargs)
        
        self.save_hyperparameters(logger=False)
        
        if time_points[0] != 0.:
            time_points = [0.] + time_points
        if not all(t1 < t2 for t1, t2 in zip(time_points, time_points[1:])):
            raise ValueError("Time points must be strictly increasing.")
        self.register_buffer("time_points", torch.tensor(time_points))

        if multiplier_activation is None:
            multiplier_activation = {"_target_": "torch.nn.Identity"}
        
        self.net = hydra.utils.instantiate(network)
        self.mult_act = hydra.utils.instantiate(multiplier_activation)
        self.mult_w = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.scale = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.temp_enc = hydra.utils.instantiate(temporal_encoding) if temporal_encoding else None
        self.feat_norm = hydra.utils.instantiate(feature_normalization) if feature_normalization else None

        if self.weight_init is not None:
            self._init_weights()
    
    def extra_repr(self):
        return super(StackedSolutionMap, self).extra_repr() + f"\ntime_points: {self.time_points.tolist()}"
    
    def forward(self, u0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # u0 shape: (bs, 2*dof)
        # t shape: (bs, 1)
        # out shape: (bs, 2*dof)

        u0 = self._apply_nondim(u0)
        u_out = u0.clone()

        for k in range(len(self.time_points)):
            t_k = self.time_points[k]
            t_k_plus_1 = self.time_points[k+1] if k+1 < len(self.time_points) else float("inf")

            # case 0: t <= t_k, no need to apply Phi_k
            mask0 = (t <= t_k).squeeze(-1)
            if mask0.all():
                break

            # case 1: t_k < t <= t_{k+1}
            mask1 = ((t > t_k) & (t <= t_k_plus_1)).squeeze(-1)
            if mask1.any():
                u_out[mask1] = self.apply_Phi_k(u_out[mask1], t[mask1], t_k)

            if k + 1 == len(self.time_points):
                break

            # case 2: t > t_{k+1}
            mask2 = (t > t_k_plus_1).squeeze(-1)
            if mask2.any():
                t_ = torch.ones_like(t[mask2]) * t_k_plus_1
                u_out[mask2] = self.apply_Phi_k(u_out[mask2], t_, t_k)

        u_out = self._apply_dim(u_out)
        return u_out

    def apply_Phi_k(self, u_k: torch.Tensor, t: torch.Tensor, t_k: torch.Tensor) -> torch.Tensor:
        # u_k shape: (masked bs, 2*dof)
        # t shape: (masked bs, 1)
        # t_k shape: () 
        # out shape: (masked bs, 2*dof)

        t_in = self.temp_enc(t - t_k) if self.temp_enc is not None else t - t_k
        u_k_in = self.feat_norm(u_k) if self.feat_norm is not None else u_k
        
        F_out = self.net(torch.cat([u_k_in, t_in], dim=-1))
        if self.feat_norm is not None:
            F_out = self.feat_norm.inverse(F_out)

        mult = self.mult_w * (t - t_k)
        mult = self.mult_act(mult)
        F_out = self.scale * mult * F_out

        return u_k + F_out

    @classmethod
    def from_IdentityEnforcedSolutionMap(
        cls,
        ckpt_path: str,
        time_points: List[float],
        **kwargs
    ) -> "StackedSolutionMap":

        ckpt = torch.load(ckpt_path)
        config = ckpt["hyper_parameters"]
        state_dict = ckpt["state_dict"]
        for key in ["network", "multiplier_activation", "temporal_encoding", "feature_normalization"]:
            if key in kwargs:
                kwargs.pop(key)  # ignore the key if it is provided in kwargs

        solnmap = cls(
            time_points=time_points,
            network=config["network"],
            multiplier_activation=config.get("multiplier_activation", None),
            temporal_encoding=config.get("temporal_encoding", None),
            feature_normalization=config.get("feature_normalization", None),
            **kwargs
        )
        solnmap.load_state_dict(state_dict, strict=False)

        return solnmap
    

class T0CenteredSolutionMap_old(BaseSolutionMap):
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
        super(T0CenteredSolutionMap_old, self).__init__(**kwargs)

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
        self.scale = nn.Parameter(torch.tensor(1e0), requires_grad=True)

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
        return super(T0CenteredSolutionMap_old, self).extra_repr() + f"\nT0: {self.T0.item()}"
    
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


class StackedSolutionMap_old(BaseSolutionMap):
    """
    Stacked solution map.

    Definition:
        Let Phi_k(u0, t) be the solution map for t_k <= t <= t_{k+1}. Then Phi(u0, t) is defined recursively as
        1. Phi_0(u0, t) = u0 + F(u0, t)
        2. Phi_k(u0, t) = Phi_{k-1}(u0, t_k) + F(Phi_{k-1}(u0, t_k), t-t_k) for k = 1, 2, ...
    """

    def __init__(
        self,
        time_points: List[float],
        net_F: DictConfig,
        multiplier_activation: Optional[DictConfig] = None,
        temporal_encoding: Optional[DictConfig] = None,
        **kwargs
    ) -> None:
        super(StackedSolutionMap_old, self).__init__(**kwargs)
        
        self.save_hyperparameters(logger=False)
        
        if time_points[0] != 0.:
            time_points = [0.] + time_points
        if not all(t1 < t2 for t1, t2 in zip(time_points, time_points[1:])):
            raise ValueError("Time points must be strictly increasing.")
        self.register_buffer("time_points", torch.tensor(time_points))

        if multiplier_activation is None:
            multiplier_activation = {"_target_": "torch.nn.Identity"}
        
        self.net_F = hydra.utils.instantiate(net_F)
        self.mult_act = hydra.utils.instantiate(multiplier_activation)
        self.mult_w = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.scale = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.temp_enc = hydra.utils.instantiate(temporal_encoding) if temporal_encoding else None

        if self.weight_init is not None:
            self._init_weights()
    
    def extra_repr(self):
        return super(StackedSolutionMap, self).extra_repr() + f"\ntime_points: {self.time_points.tolist()}"
    
    def forward(self, u0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # u0 shape: (bs, 2*dof)
        # t shape: (bs, 1)
        # out shape: (bs, 2*dof)

        u0 = self._apply_nondim(u0)
        u_out = u0.clone()

        for k in range(len(self.time_points)):
            t_k = self.time_points[k]
            t_k_plus_1 = self.time_points[k+1] if k+1 < len(self.time_points) else float("inf")

            # case 0: t <= t_k, no need to apply Phi_k
            mask0 = (t <= t_k).squeeze(-1)
            if mask0.all():
                break

            # case 1: t_k < t <= t_{k+1}
            mask1 = ((t > t_k) & (t <= t_k_plus_1)).squeeze(-1)
            if mask1.any():
                u_out[mask1] = self.apply_Phi_k(u_out[mask1], t[mask1], t_k)

            if k + 1 == len(self.time_points):
                break

            # case 2: t > t_{k+1}
            mask2 = (t > t_k_plus_1).squeeze(-1)
            if mask2.any():
                t_ = torch.ones_like(t[mask2]) * t_k_plus_1
                u_out[mask2] = self.apply_Phi_k(u_out[mask2], t_, t_k)

        u_out = self._apply_dim(u_out)
        return u_out

    def apply_Phi_k(self, u_k: torch.Tensor, t: torch.Tensor, t_k: torch.Tensor) -> torch.Tensor:
        # u_k shape: (masked bs, 2*dof)
        # t shape: (masked bs, 1)
        # t_k shape: () 
        # out shape: (masked bs, 2*dof)

        if self.temp_enc is not None:
            t_in = self.temp_enc(t - t_k)
        else:
            t_in = t - t_k
        F_in = torch.cat([u_k, t_in], dim=-1)
        F_out = self.net_F(F_in)

        mult = self.mult_w * (t - t_k)
        mult = self.mult_act(mult)
        F_out = mult * F_out

        return u_k + F_out
