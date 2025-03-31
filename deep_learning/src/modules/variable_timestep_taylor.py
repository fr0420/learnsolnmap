from typing import Dict, List, Optional

import hydra
from omegaconf import DictConfig
import torch
from torch import nn

from modules.solnmap import BaseSolutionMap
from modules.fixed_timestep import FixedStepSolutionMap
from modules.variable_timestep import PreserveVelocityNorm


class SolutionMapWithFp(BaseSolutionMap):
    """
    Solution map with the F_p operation. The time derivative of F_p evaluated at t=0 is exact up to order p. 
    
    Definition: 
        F_0(u, t, param; f) = mult_act(w1 * t) * net([f(u, param), u, t, param])
        F_1(u, t, param; f) = mult_act(w1 * t) / w1 * (
            f(u, param) + mult_act(w2 * t) * net([f(u, param), u, t, param])
        )
        F_2(u, t, param; f) = mult_act(w1 * t) / w1 * (
            f(u, param) + mult_act(w2 * t) / w2 * (
                1/2 * f'(u, param) f(u, param) + mult_act(w3 * t) * net([f(u, param), u, t, param])
            )
        )
    """

    def __init__(
        self, 
        order: int,
        network: DictConfig, 
        multiplier_activation: Optional[DictConfig] = None,
        temporal_encoding: Optional[DictConfig] = None,
        feature_normalization: Optional[DictConfig] = None,
        **kwargs
    ) -> None:
        super(SolutionMapWithFp, self).__init__(**kwargs)

        if multiplier_activation is None:
            multiplier_activation = {"_target_": "torch.nn.Identity"}
        
        if order not in [0, 1, 2]:
            raise ValueError("Order must be one of [0, 1, 2].")  
        self.order = order

        self.net = hydra.utils.instantiate(network)
        self.mult_act = hydra.utils.instantiate(multiplier_activation)
        self.w1 = nn.Parameter(torch.tensor(1.), requires_grad=True)
        if self.order >= 1:
            self.w2 = nn.Parameter(torch.tensor(1.), requires_grad=True)
        if self.order == 2:
            self.w3 = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.temp_enc = hydra.utils.instantiate(temporal_encoding) if temporal_encoding else None
        self.feat_norm = hydra.utils.instantiate(feature_normalization) if feature_normalization else None
    
        if self.weight_init is not None:
            self._init_weights()

    def _apply_F(self, u: torch.Tensor, t: torch.Tensor, p: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # u shape: (bs, 2*dof)
        # t shape: (bs, 1)
        # p[key] shape: (bs, 1) (only keys in self.problem_param_keys are used)
        # out shape: (bs, 2*dof)
        
        # switch to dimensional space to compute du(u, p) and ddu(u, p) then switch back to nondimensional space
        dim_u = self._apply_dim(u, p)
        du = self.problem.compute_du(dim_u, None, p)
        du = self._apply_nondim(du, p, deriv_mode=True)
        if self.order == 2:
            ddu = self.problem.compute_ddu(dim_u, None, p)
            ddu = self._apply_nondim(ddu, p, deriv_mode=True)  # fix this 

        # pass through the network
        u_in = self.feat_norm(u) if self.feat_norm is not None else u
        t_in = self.temp_enc(t) if self.temp_enc is not None else t
        out = self.net(torch.cat([du, u_in, t_in] + self.prepare_params_input(p), dim=-1))
        if self.feat_norm is not None:
            out = self.feat_norm.inverse(out)

        # construct F_p(u, t, p)
        if self.order == 0:
            mult1 = self.mult_act(self.w1 * t)
            out = mult1 * out
            return out 
        
        elif self.order == 1:
            mult2 = self.mult_act(self.w2 * t)
            out = du + mult2 * out
            mult1 = self.mult_act(self.w1 * t) / self.w1
            out = mult1 * out
            return out
        
        elif self.order == 2:
            mult3 = self.mult_act(self.w3 * t)
            out = 0.5 * ddu + mult3 * out
            mult2 = self.mult_act(self.w2 * t) / self.w2
            out = du + mult2 * out
            mult1 = self.mult_act(self.w1 * t) / self.w1
            out = mult1 * out
            return out 
    
    def _apply_F_ignore_net(self, u: torch.Tensor, t: torch.Tensor, p: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # u shape: (bs, 2*dof)
        # t shape: (bs, 1)
        # p[key] shape: (bs, 1) (only keys in self.problem_param_keys are used)
        # out shape: (bs, 2*dof)
        
        # switch to dimensional space to compute du(u, p) and ddu(u, p) then switch back to nondimensional space
        dim_u = self._apply_dim(u, p)
        du = self.problem.compute_du(dim_u, None, p)
        du = self._apply_nondim(du, p, deriv_mode=True)
        if self.order == 2:
            ddu = self.problem.compute_ddu(dim_u, None, p)
            ddu = self._apply_nondim(ddu, p, deriv_mode=True)  # fix this 

        # construct F_p(u, t, p) without the network component
        if self.order == 0:
            return torch.zeros_like(u)
        
        elif self.order == 1:
            out = du
            mult1 = self.mult_act(self.w1 * t) / self.w1
            out = mult1 * out
            return out
        
        elif self.order == 2:
            out = 0.5 * ddu
            mult2 = self.mult_act(self.w2 * t) / self.w2
            out = du + mult2 * out
            mult1 = self.mult_act(self.w1 * t) / self.w1
            out = mult1 * out
            return out 
    
    def _display_net_input_and_output(self, u: torch.Tensor, t: torch.Tensor, p: Optional[Dict[str, torch.Tensor]] = None) -> None:
        u = self._apply_nondim(u, p)

        # switch to dimensional space to compute du(u, p) and ddu(u, p) then switch back to nondimensional space
        dim_u = self._apply_dim(u, p)
        du = self.problem.compute_du(dim_u, None, p)
        du = self._apply_nondim(du, p, deriv_mode=True)
        ddu = self.problem.compute_ddu(dim_u, None, p)
        ddu = self._apply_nondim(ddu, p, deriv_mode=True)  # fix this 

        # pass through the network
        u_in = self.feat_norm(u) if self.feat_norm is not None else u
        t_in = self.temp_enc(t) if self.temp_enc is not None else t
        out = self.net(torch.cat([du, u_in, t_in] + self.prepare_params_input(p), dim=-1))
        if self.feat_norm is not None:
            out = self.feat_norm.inverse(out)

        p_in = self.prepare_params_input(p)
        print(f"u_in: {u_in}")
        print(f"t_in: {t_in}")
        print(f"p_in: {p_in}")
        print(f"du_in: {du}")
        print(f"ddu_in: {ddu}")
        print(f"out: {out}")
    

class TaylorBasedIdentityEnforcedSolutionMap(SolutionMapWithFp):
    """
    Solution map enforcing the identity function and the correct derivatives up to order p at t = 0.

    Definition:
        Phi(u0, t, param; f) = u0 + F_p(u0, t, param; f) 
    """
    
    def __init__(
        self, 
        network: DictConfig, 
        multiplier_activation: Optional[DictConfig] = None,
        temporal_encoding: Optional[DictConfig] = None,
        feature_normalization: Optional[DictConfig] = None,
        preserve_velocity_norm: bool = False,
        order: int = 1,
        **kwargs
    ) -> None:
        super(TaylorBasedIdentityEnforcedSolutionMap, self).__init__(
            order=order,
            network=network,
            multiplier_activation=multiplier_activation,
            temporal_encoding=temporal_encoding,
            feature_normalization=feature_normalization,
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
    
    def forward_ignore_net(self, u0: torch.Tensor, t: torch.Tensor, p: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # u0 shape: (bs, 2*dof)
        # t shape: (bs, 1)
        # p[key] shape: (bs, 1) (only keys in self.problem_param_keys are used)
        # out shape: (bs, 2*dof)

        u0 = self._apply_nondim(u0, p)
        out = u0 + self._apply_F_ignore_net(u0, t, p)
        # if self.preserve_velocity_norm is not None:
        #     out = self.preserve_velocity_norm(u0, out)
        out = self._apply_dim(out, p)

        return out
    
    def extra_repr(self):
        return super(TaylorBasedIdentityEnforcedSolutionMap, self).extra_repr() + f"\norder: {self.order}"


class TaylorBasedT0CenteredSolutionMap(SolutionMapWithFp):
    """
    Solution map with the timestep `t` centered around a reference time `T0`.

    Definition:
        Phi(u0, t, param; f) = u_T0 + F_p(u_T0, t-T0, param; f)
                        u_T0 = net_T0([f(u0), u0, param]) 
    """

    def __init__(
        self,
        T0: float,
        net_T0: DictConfig,
        net_residual: DictConfig,
        multiplier_activation: Optional[DictConfig] = None,
        temporal_encoding: Optional[DictConfig] = None,
        feature_normalization: Optional[DictConfig] = None,
        preserve_velocity_norm: bool = False,
        order: int = 1,
        **kwargs
    ) -> None:
        super(TaylorBasedT0CenteredSolutionMap, self).__init__(
            order=order,
            network=net_residual,
            multiplier_activation=multiplier_activation,
            temporal_encoding=temporal_encoding,
            feature_normalization=feature_normalization,
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

    def _apply_net_T0(self, u0: torch.Tensor, p: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        if isinstance(self.net_T0, BaseSolutionMap):
            # temporarily switch to dimensional space 
            # because solution maps typically expect dimensional inputs
            u0 = self._apply_dim(u0, p)
            if isinstance(self.net_T0, FixedStepSolutionMap):
                out = self.net_T0(u0, p, None, 2)[1]
            else:  # Variable-time solution maps
                t = torch.full_like(u0[:, :1], self.T0)
                out = self.net_T0(u0, t, p)
            return self._apply_nondim(out, p)  # switch back to nondimensional space
        else:
            # a standard neural network expects nondimensional inputs
            # switch to dimensional space to compute f(u, p) then switch back to nondimensional space
            du0 = self.problem.compute_du(self._apply_dim(u0, p), None, p)
            du0 = self._apply_nondim(du0, p, deriv_mode=True)
            out = self.net_T0(torch.cat([du0, u0] + self.prepare_params_input(p), dim=-1))
            if self.preserve_velocity_norm is not None:
                out = self.preserve_velocity_norm(u0, out)
            return out

    def extra_repr(self):
        return super(TaylorBasedT0CenteredSolutionMap, self).extra_repr() + f"\nT0: {self.T0.item()}\norder: {self.order}"
    
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
    ) -> "TaylorBasedT0CenteredSolutionMap":

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
