from typing import Dict, List, Optional

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


class PreserveVelocityNorm(nn.Module):
    """
    Normalize the first two components (vx, vy) of an input tensor so that
    their norm matches that of a reference tensor.
    
    For example, if:
      original = [vx, vy, ...] with target norm r = sqrt(vx^2 + vy^2),
      updated  = [vx_new, vy_new, ...],
    then the module rescales (vx_new, vy_new) so that their norm equals r.
    """
    def __init__(self, eps: float = 1e-8):
        super(PreserveVelocityNorm, self).__init__()
        self.eps = eps

    def forward(self, original: torch.Tensor, updated: torch.Tensor) -> torch.Tensor:
        target_norm = torch.norm(original[..., :2], p=2, dim=-1, keepdim=True)        
        updated_norm = torch.norm(updated[..., :2], p=2, dim=-1, keepdim=True) + self.eps
        normalized_velocity = updated[..., :2] / updated_norm * target_norm        
        out = torch.cat([normalized_velocity, updated[..., 2:]], dim=-1)
        return out


class SolutionMapWithF(BaseSolutionMap):
    """
    Solution map with an F function.
    
    Definition:
        F(u, t, p) = scale * mult_act(multi_w * t) * net([u, t, p])
        Optionally, augment the network input with f(u): net([f(u), u, t, p])
    """

    def __init__(
        self, 
        network: DictConfig, 
        multiplier_activation: Optional[DictConfig] = None,
        temporal_encoding: Optional[DictConfig] = None,
        feature_normalization: Optional[DictConfig] = None,
        use_dudt: bool = False,
        **kwargs
    ) -> None:
        super(SolutionMapWithF, self).__init__(**kwargs)

        if multiplier_activation is None:
            multiplier_activation = {"_target_": "torch.nn.Identity"}
        
        self.net = hydra.utils.instantiate(network)
        self.mult_act = hydra.utils.instantiate(multiplier_activation)
        self.mult_w = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.scale = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.temp_enc = hydra.utils.instantiate(temporal_encoding) if temporal_encoding else None
        self.feat_norm = hydra.utils.instantiate(feature_normalization) if feature_normalization else None
        self.use_dudt = use_dudt

        if self.weight_init is not None:
            self._init_weights()
    
    def _apply_F(self, u: torch.Tensor, t: torch.Tensor, p: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # u shape: (bs, 2*dof)
        # t shape: (bs, 1)
        # p[key] shape: (bs, 1) (only keys in self.problem_param_keys are used)
        # out shape: (bs, 2*dof)
        
        if self.use_dudt:
            # switch to dimensional space to compute f(u, p) then switch back to nondimensional space
            du = self.problem.compute_du(self._apply_dim(u, p), t=None, p=p)
            du = self._apply_nondim(du, p, deriv_mode=True)

        u_in = self.feat_norm(u) if self.feat_norm is not None else u
        t_in = self.temp_enc(t) if self.temp_enc is not None else t
        
        if self.use_dudt:
            out = self.net(torch.cat([du, u_in, t_in] + self.prepare_params_input(p), dim=-1))
        else:
            out = self.net(torch.cat([u_in, t_in] + self.prepare_params_input(p), dim=-1))
        if self.feat_norm is not None:
            out = self.feat_norm.inverse(out)

        mult = self.mult_act(self.mult_w * t)
        out = self.scale * mult * out

        return out


class IdentityEnforcedSolutionMap(SolutionMapWithF):
    """
    Solution map enforcing the identity function at t = 0.

    Definition:
        Phi(u0, t, p) = u0 + F(u0, t, p) 
                      = u0 + scale * mult_act(multi_w * t) * net([u0, t, p])
    """
    
    def __init__(
        self, 
        network: DictConfig, 
        multiplier_activation: Optional[DictConfig] = None,
        temporal_encoding: Optional[DictConfig] = None,
        feature_normalization: Optional[DictConfig] = None,
        use_dudt: bool = False,
        preserve_velocity_norm: bool = False,
        **kwargs
    ) -> None:
        super(IdentityEnforcedSolutionMap, self).__init__(
            network=network,
            multiplier_activation=multiplier_activation,
            temporal_encoding=temporal_encoding,
            feature_normalization=feature_normalization,
            use_dudt=use_dudt,
            **kwargs
        )
        self.save_hyperparameters(logger=False)
        self.preserve_velocity_norm = PreserveVelocityNorm() if preserve_velocity_norm else None

    def forward(self, u0: torch.Tensor, t: torch.Tensor, p: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # u0 shape: (bs, 2*dof)
        # t shape: (bs, 1)
        # p[key] shape: (bs, 1) (only keys in self.problem_param_keys are used)
        # out shape: (bs, 2*dof)

        u0 = self._apply_nondim(u0, p)
        out = u0 + self._apply_F(u0, t, p)
        if self.preserve_velocity_norm is not None:
            out = self.preserve_velocity_norm(u0, out)
        out = self._apply_dim(out, p)

        return out

    def _calc_dPhidt_at_zero_manually(self, u0: torch.Tensor, p: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # todo: fix this function when self.preserve_velocity_norm is not None
        u0 = self._apply_nondim(u0, p)

        t = torch.zeros_like(u0[:, :1])
        u0_in = self.feat_norm(u0) if self.feat_norm is not None else u0
        t_in = self.temp_enc(t) if self.temp_enc is not None else t

        out = self.net(torch.cat([u0_in, t_in] + self.prepare_params_input(p), dim=-1))
        
        out = self.scale * self.mult_w * out
        out = self._apply_dim(out, p)
        return out 


class StackedSolutionMap(SolutionMapWithF):
    """
    Stacked solution map.

    Definition:
        Let Phi_k(u0, t, p) be the solution map for t_k <= t <= t_{k+1}. Then Phi(u0, t, p) is defined recursively as
        1. Phi_0(u0, t, p) = u0 + F(u0, t, p)
        2. Phi_k(u0, t, p) = Phi_{k-1}(u0, t_k, p) + F(Phi_{k-1}(u0, t_k, p), t-t_k, p) for k = 1, 2, ...
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
        super(StackedSolutionMap, self).__init__(
            network=network,
            multiplier_activation=multiplier_activation,
            temporal_encoding=temporal_encoding,
            feature_normalization=feature_normalization,
            **kwargs
        )
        self.save_hyperparameters(logger=False)

        if time_points[0] != 0.:
            time_points = [0.] + time_points
        if not all(t1 < t2 for t1, t2 in zip(time_points, time_points[1:])):
            raise ValueError("Time points must be strictly increasing.")
        self.register_buffer("time_points", torch.tensor(time_points))

    def extra_repr(self):
        return super(StackedSolutionMap, self).extra_repr() + f"\ntime_points: {self.time_points.tolist()}"
    
    def forward(self, u0: torch.Tensor, t: torch.Tensor, p: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # u0 shape: (bs, 2*dof)
        # t shape: (bs, 1)
        # p[key] shape: (bs, 1) (only keys in self.problem_param_keys are used)
        # out shape: (bs, 2*dof)

        u0 = self._apply_nondim(u0, p)
        out = u0.clone()

        for k in range(len(self.time_points)):
            t_k = self.time_points[k]
            mask = (t > t_k).squeeze(-1)

            # break loop if all t values are <= t_k (no operation needed)
            if not mask.any():
                break
            
            # calculate delta_t based on whether this is the last time point
            if k + 1 == len(self.time_points):
                delta_t = t[mask] - t_k
            else:
                t_k_plus_1 = self.time_points[k+1]
                delta_t = torch.minimum(t[mask] - t_k, torch.full_like(t[mask], t_k_plus_1 - t_k))

            # apply F with the appropriate delta_t
            p_mask = {key: val[mask] for key, val in p.items()} if p is not None else None
            out[mask] = out[mask] + self._apply_F(out[mask], delta_t, p_mask)

        out = self._apply_dim(out, p)

        return out
    
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
        solnmap.load_state_dict(state_dict, strict=True)

        return solnmap


class T0CenteredSolutionMap(SolutionMapWithF):
    """
    Solution map with the timestep `t` centered around a reference time `T0`.

    Definition:
        Phi(u0, t, p) = net_T0(u0, p) + F(net_T0(u0, p), t-T0, p)
                      = net_T0(u0, p) + scale * mult_act(multi_w * (t-T0)) * net([net_T0(u0, p), t-T0, p])  
    """

    def __init__(
        self,
        T0: float,
        net_T0: DictConfig,
        net_residual: DictConfig,
        multiplier_activation: Optional[DictConfig] = None,
        temporal_encoding: Optional[DictConfig] = None,
        feature_normalization: Optional[DictConfig] = None,
        use_dudt: bool = False,
        preserve_velocity_norm: bool = False,
        **kwargs
    ) -> None:
        super(T0CenteredSolutionMap, self).__init__(
            network=net_residual,
            multiplier_activation=multiplier_activation,
            temporal_encoding=temporal_encoding,
            feature_normalization=feature_normalization,
            use_dudt=use_dudt,
            **kwargs,
        )
        self.save_hyperparameters(logger=False)

        self.register_buffer("T0", torch.tensor(T0))
        if net_T0["_target_"] in [
            "modules.fixed_timestep.FixedStepSolutionMap",
            "modules.variable_timestep.IdentityEnforcedSolutionMap",
            "modules.variable_timestep.T0CenteredSolutionMap"]:
            self.net_T0 = hydra.utils.instantiate(net_T0, _recursive_=False)
        else:
            self.net_T0 = hydra.utils.instantiate(net_T0)
        self.preserve_velocity_norm = PreserveVelocityNorm() if preserve_velocity_norm else None
        self.use_dudt = use_dudt

        if self.weight_init is not None:
            self._init_weights()

    def forward(self, u0: torch.Tensor, t: torch.Tensor, p: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # u0 shape: (bs, 2*dof)
        # t shape: (bs, 1)
        # p[key] shape: (bs, 1) (only keys in self.problem_param_keys are used)
        # out shape: (bs, 2*dof)

        u0 = self._apply_nondim(u0, p)
        net_T0_out = self._apply_net_T0(u0, p)
        out = net_T0_out + self._apply_F(net_T0_out, t-self.T0, p)
        if self.preserve_velocity_norm is not None:
            out = self.preserve_velocity_norm(u0, out)
        out = self._apply_dim(out, p)

        return out

    def _apply_net_T0(self, u0_nd: torch.Tensor, p: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        if isinstance(self.net_T0, BaseSolutionMap):
            # temporarily switch back to dimensional space 
            # because solution maps typically expect dimensional inputs
            u0 = self._apply_dim(u0_nd, p)
            if isinstance(self.net_T0, FixedStepSolutionMap):
                out = self.net_T0(u0, p, None, 2)[1]
            elif isinstance(self.net_T0, T0CenteredSolutionMap) or isinstance(self.net_T0, IdentityEnforcedSolutionMap):
                t = torch.full_like(u0[:, :1], self.T0)
                out = self.net_T0(u0, t, p)
            else:
                raise ValueError("Unsupported SolutionMap type.")
            return self._apply_nondim(out, p)
        else:
            # a standard neural network expects nondimensional inputs
            if self.use_dudt:
                # switch to dimensional space to compute f(u, p) then switch back to nondimensional space
                du0 = self.problem.compute_du(self._apply_dim(u0_nd, p), t=None, p=p)
                du0_nd = self._apply_nondim(du0, p, deriv_mode=True)
                return self.net_T0(torch.cat([du0_nd, u0_nd] + self.prepare_params_input(p), dim=-1))
            else:
                return self.net_T0(torch.cat([u0_nd] + self.prepare_params_input(p), dim=-1))

    def extra_repr(self):
        return super(T0CenteredSolutionMap, self).extra_repr() + f"\nT0: {self.T0.item()}"
    
    def load_state_dict(self, state_dict, strict=True):
        # rename keys in state_dict to match the current model
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("net_res."):
                new_key = "net" + key[len("net_res"):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        super(T0CenteredSolutionMap, self).load_state_dict(new_state_dict, strict=strict)

    @classmethod
    def from_pretrained_net_T0(
        cls,
        ckpt_path: str,
        pretrained_class: str,
        T0: float,
        net_residual: DictConfig,
        multiplier_activation: Optional[DictConfig] = None,
        temporal_encoding: Optional[DictConfig] = None,
        feature_normalization: Optional[DictConfig] = None,
        preserve_velocity_norm: bool = False,
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
            feature_normalization=feature_normalization,
            preserve_velocity_norm=preserve_velocity_norm,
            **kwargs
        )
        solnmap.net_T0.load_state_dict(state_dict, strict=True)

        if isinstance(solnmap.net_T0, FixedStepSolutionMap):
            if solnmap.net_T0.T0 != T0:
                raise ValueError("T0 must match the fixed timestep of the pretrained model.")

        if pretrained_frozen:
            for param in solnmap.net_T0.parameters():
                param.requires_grad = False

        return solnmap
    
    # @classmethod
    # def from_pretrained_net_residual(cls,):
