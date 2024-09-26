from typing import Dict 

import hydra
import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf

from modules.default import BaseLitModel
from integrators.integrators import instantiate_symplectic_integrator, SymplecticIntegrator
from problems.default import SeparableHamiltonianSystem



DEFAULT_LOSS_HPARAMS = {}
DEFAULT_METRIC_HPARAMS = {}


def velocity_verlet_step(u0, h, A):
    """
    Perform a single step of the velocity Verlet algorithm.
    
    :param u0: current state
    :param h: time step size
    :param A: right-hand side of the ODE x'' = A(x)
    """
    v0, x0 = u0.chunk(2, dim=-1)
    v_mid = v0 + 0.5 * h * A(x0, None)
    x = x0 + v_mid * h
    v = v_mid + 0.5 * h * A(x, None)
    return torch.cat((v, x), dim=-1)


class BaseVariableDtSolutionMap(BaseLitModel):
    """Base variable Dt solution map model."""

    def __init__(
            self,
            problem: DictConfig,
            loss: DictConfig,
            loss_hparams: DictConfig = None,
            metric_hparams: DictConfig = None,
            use_dimensionless: bool = True,
            use_dimensionless_for_loss: bool = True,
            **kwargs
        ) -> None:
        super(BaseVariableDtSolutionMap, self).__init__(**kwargs)

        self.problem: SeparableHamiltonianSystem = hydra.utils.instantiate(problem)
        self.loss_fn: nn.Module = hydra.utils.instantiate(loss)
        self.loss_hparams = self._initialize_hparams(loss_hparams, DEFAULT_LOSS_HPARAMS)
        self.metric_hparams = self._initialize_hparams(metric_hparams, DEFAULT_METRIC_HPARAMS)
        self.use_dimensionless = use_dimensionless
        self.use_dimensionless_for_loss = use_dimensionless_for_loss
        if self.use_dimensionless or self.use_dimensionless_for_loss:
            self._validate_nondim_methods()
        self.loss_integrators = None
        self.metric_integrators = None

    def _validate_nondim_methods(self):
        methods = ["nondim_u", "dim_u", "nondim_du", "dim_du"]
        for method in methods:
            assert callable(getattr(self.problem, method, None)), f"Method {method} is not implemented."
    
    def _initialize_hparams(self, hparams: DictConfig, default_hparams: dict):
        initialized_hparams = OmegaConf.create(default_hparams)
        return OmegaConf.merge(initialized_hparams, hparams) if hparams else initialized_hparams
    
    def update_loss_hparams(self, hparams: DictConfig):
        if hparams is not None:
            self.loss_hparams = OmegaConf.merge(self.loss_hparams, hparams)
 
    def update_metric_hparams(self, hparams: DictConfig):
        if hparams is not None:
            self.metric_hparams = OmegaConf.merge(self.metric_hparams, hparams) 
       
    def forward(self, u0: torch.Tensor, Dt: torch.Tensor) -> torch.Tensor:
        pass 
    
    def predict_sequence(self, u0: torch.Tensor, Dt: torch.Tensor, sequence_len: int) -> torch.Tensor:
        pred_seq = [u0]
        u = u0
        for _ in range(sequence_len-1):
            u = self(u, Dt)
            pred_seq.append(u)
        return pred_seq
    
    def training_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx, sequence_len=2):
        u0, _, Dt = self._unpack_batch(batch)
        pred_seq = self.predict_sequence(u0, Dt, sequence_len)
        batch["pred_seq"] = pred_seq
        return batch
    
    def model_step(self, batch: dict, batch_idx: int, stage: str = None):
        # unpack batch
        u0, u1, Dt = self._unpack_batch(batch)

        # predict
        u1_pred = self(u0, Dt)

        # initialize integrators if needed
        self._ensure_integrators_initialized()

        # compute losses
        sup_losses = self._compute_supervised_losses(u1_pred, u1, self.loss_hparams)
        unsup_losses = self._compute_unsupervised_losses(None, self.loss_hparams, self.loss_integrators)
        loss = self._compute_total_loss(sup_losses, unsup_losses)

        # compute metrics
        fitting_metrics = self.problem.compute_errors(u1_pred, u1, reduction="mean")
        structure_metrics = self._compute_unsupervised_losses(u0, self.metric_hparams, self.metric_integrators)

        # log losses and metrics
        metrics = {"loss": loss.detach()}
        metrics.update(self._prepare_metrics(sup_losses, suffix="_loss"))
        metrics.update(self._prepare_metrics(unsup_losses, suffix="_loss"))
        metrics.update(self._prepare_metrics(fitting_metrics))
        metrics.update(self._prepare_metrics(structure_metrics, suffix="_err"))
        batch_size = len(u0)
        self._log_step(metrics, stage, loss, batch_size)

        return {"loss": loss, "batch_size": batch_size, "metrics": metrics}
    
    def _unpack_batch(self, batch: dict) -> tuple:
        u0 = batch["input"]
        u1 = batch["target_seq"][1]
        Dt = batch["Dt"]
        return u0, u1, Dt
    
    def _ensure_integrators_initialized(self):
        if self.loss_integrators is None:
            self.loss_integrators = self._initialize_integrators(self.loss_hparams)
        if self.metric_integrators is None:
            self.metric_integrators = self._initialize_integrators(self.metric_hparams)
    
    def _initialize_integrators(self, hparams: DictConfig) -> dict:
        integrators = {}
        for key in ["additive_vv", "commutative_vv"]:
            if key in hparams:
                params = hparams[key]
                assert params.get("integrator"), f"{key}.integrator is required."
                integrators[key] = self._instantiate_integrator(params["integrator"])
        return integrators

    def _instantiate_integrator(self, config: DictConfig):
        return instantiate_symplectic_integrator(config["method"], config["stepsize"], config["nsteps"], self.problem.compute_ddx)
    
    def _compute_supervised_losses(self, u_pred: torch.Tensor, u_true: torch.Tensor, hparams: DictConfig) -> Dict[str, torch.Tensor]:
        losses = {}
        if "misfit" in hparams:
            losses["misfit"] = self._calc_misfit_error(u_pred, u_true)
        return losses

    def _compute_unsupervised_losses(self, u0: torch.Tensor, hparams: DictConfig, integrators: dict) -> Dict[str, torch.Tensor]:
        LOSS_FN_DICT = {
            "additive": self._calc_additive_error,
            "commutative": self._calc_commutative_error,
            "additive_vv": self._calc_additive_vv_error,
            "commutative_vv": self._calc_commutative_vv_error,
            "dyadic": self._calc_dyadic_error,
            "reversibility": self._calc_reversibiliy_error,
            "dynamics": self._calc_dynamics_error,
            "combined": self._calc_combined_error,
        }

        def prepare_u0(u0, batch_size):
            if u0 is None:
                assert batch_size is not None, "batch_size should be provided."
                return self._prepare_random_states(batch_size)
            return u0 
        
        losses = {}
        for loss_name, loss_fn in LOSS_FN_DICT.items():
            if loss_name in hparams:
                params = hparams[loss_name]
                if loss_name in {"additive", "commutative"}:
                # if loss_name in {"additive", "commutative", "additive_vv", "commutative_vv"}:
                    u0_ = prepare_u0(u0, params.get("batch_size", None))
                    s = self._prepare_random_timesteps(u0_.shape[0], params["s_min"], params["s_max"])
                    t = self._prepare_random_timesteps(u0_.shape[0], params["t_min"], params["t_max"])
                    losses[loss_name] = loss_fn(u0_, s, t)
                elif loss_name in {"additive_vv", "commutative_vv", "dyadic", "reversibility", "dynamics"}:
                    u0_ = prepare_u0(u0, params.get("batch_size", None))
                    h = self._prepare_random_timesteps(u0_.shape[0], params["h_min"], params["h_max"])
                    if loss_name in {"additive_vv", "commutative_vv"}:
                        losses[loss_name] = loss_fn(u0_, h, integrators[loss_name])
                    else:
                        losses[loss_name] = loss_fn(u0_, h)
                # elif loss_name == "dynamics":
                #     u0_ = prepare_u0(u0, params.get("batch_size", None))
                #     losses[loss_name] = loss_fn(u0_)
                elif loss_name == "combined":
                    u0_ = prepare_u0(u0, params.get("batch_size", None))
                    s = self._prepare_random_timesteps(u0_.shape[0], params["s_min"], params["s_max"])
                    h = self._prepare_random_timesteps(u0_.shape[0], params["h_min"], params["h_max"])
                    losses[loss_name] = loss_fn(u0_, s, h, params["K"])

        return losses
    
    def _compute_total_loss(self, sup_losses: Dict[str, torch.Tensor], unsup_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = sum(self.loss_hparams[loss_name]["strength"] * loss_value
               for loss_name, loss_value in {**sup_losses, **unsup_losses}.items()
               if loss_name in self.loss_hparams)
        return loss
    
    def _prepare_metrics(self, metric_dict: dict, suffix: str = "") -> dict:
        return {f"{name}{suffix}": value.detach() for name, value in metric_dict.items()}

    def _log_step(self, metrics, stage, loss, batch_size):
        if stage == "train":
            self.training_step_outputs.append({"batch_size": batch_size, "metrics": metrics})
            self.log("step_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        elif stage == "val":
            self.validation_step_outputs.append({"batch_size": batch_size, "metrics": metrics})
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        elif stage == "test":
            self.test_step_outputs.append({"batch_size": batch_size, "metrics": metrics})

    def _calc_u_misfit(self, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.use_dimensionless_for_loss:
            return self.loss_fn(self.problem.nondim_u(u), self.problem.nondim_u(w))
        else:
            return self.loss_fn(u, w)
    
    def _calc_du_misfit(self, du: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        if self.use_dimensionless_for_loss:
            return nn.functional.mse_loss(self.problem.nondim_du(du), self.problem.nondim_du(dw))
        else:
            return nn.functional.mse_loss(du, dw)

    def _calc_misfit_error(self, u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
        return self._calc_u_misfit(u_pred, u_true)
    
    def _prepare_random_states(self, batch_size: int) -> torch.Tensor:
        return self.problem.random_states(batch_size).to(self.device).to(self.dtype)
    
    def _prepare_random_timesteps(self, batch_size: int, min_h: float, max_h: float) -> torch.Tensor:
        assert min_h <= max_h, "min_h should be less than or equal to max_h."
        return torch.rand(batch_size, 1).to(self.device).to(self.dtype) * (max_h - min_h) + min_h

    def _calc_commutative_error(self, u0: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._calc_u_misfit(self(self(u0, t), s), self(self(u0, s), t))
    
    def _calc_additive_error(self, u0: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._calc_u_misfit(self(self(u0, t), s), self(u0, s+t))
    
    def _calc_commutative_vv_error(self, u0: torch.Tensor, h: torch.Tensor, integrator: SymplecticIntegrator) -> torch.Tensor:
        return self._calc_u_misfit(integrator(self(u0, h)), self(integrator(u0), h))
    
    def _calc_additive_vv_error(self, u0: torch.Tensor, h: torch.Tensor, integrator: SymplecticIntegrator) -> torch.Tensor:
        return self._calc_u_misfit(integrator(self(u0, h)), self(u0, h+integrator.dt))
    
    # def _calc_commutative_vv_error_old(self, u0: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    #     return self._calc_u_misfit(self._vv_step(self(u0, s), t), self(self._vv_step(u0, t), s))
    
    # def _calc_additive_vv_error_old(self, u0: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    #     return self._calc_u_misfit(self._vv_step(self(u0, s), t), self(u0, s+t))
    
    def _calc_dyadic_error(self, u0: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self._calc_u_misfit(self(self(u0, h), h), self(u0, 2*h))

    def _calc_reversibiliy_error(self, u0: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self._calc_u_misfit(self(self(u0, h), -h), u0)
    
    def _calc_combined_error(self, u0: torch.Tensor, s: torch.Tensor, h: torch.Tensor, K: int) -> torch.Tensor:
        part1 = 0.5 * self._calc_u_misfit(self(u0, h), self._vv_step(u0, h)) + \
                0.5 * self._calc_u_misfit(self(u0, -h), self._vv_step(u0, -h))
        part2 = 0.
        for k in range(K+1):
            part2 += 0.5 * self._calc_dyadic_error(u0, 2**k * h) + 0.5 * self._calc_dyadic_error(u0, - 2**k * h)
        part3 = 0.
        for k in range(K+1):
            part3 += 0.5 * self._calc_commutative_error(u0, s, 2**k * h) + 0.5 * self._calc_commutative_error(u0, s, -2**k * h)
        return (part1 + part2 + part3) / (2*K + 3)
    
    def _vv_step(self, u0: torch.Tensor, Dt: torch.Tensor) -> torch.Tensor:
        return velocity_verlet_step(u0, Dt, self.problem.compute_ddx)
    
    def _calc_dynamics_error(self, u0: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        du = self.problem.compute_du(self(u0, h))
        du_pred = torch.func.vmap(torch.func.jacrev(self, argnums=1))(u0, h)  # (bs, 2*dof, 1)
        du_pred = du_pred.squeeze(-1)  # (bs, 2*dof)
        return self._calc_du_misfit(du_pred, du)

    # def _calc_dynamics_error_old(self, u0: torch.Tensor) -> torch.Tensor:
    #     du = self.problem.compute_du(u0)
    #     Dt = torch.zeros(u0.shape[0], 1).to(u0)
    #     du_pred = torch.func.vmap(torch.func.jacrev(self, argnums=1))(u0, Dt)  # (bs, 2*dof, 1)
    #     du_pred = du_pred.squeeze(-1)  # (bs, 2*dof)
    #     return self._calc_du_misfit(du_pred, du)

class VariableDtSolutionMap(BaseVariableDtSolutionMap):
    """Variable Dt solution map."""

    def __init__(self, network: DictConfig, **kwargs) -> None:
        super(VariableDtSolutionMap, self).__init__(**kwargs)
        
        self.save_hyperparameters(logger=False)
        
        self.net = hydra.utils.instantiate(network)
        self.scaling_factor = nn.Parameter(torch.tensor(1e-2), requires_grad=True)

        if self.weight_init is not None:
            self._init_weights()

    def forward(self, u0: torch.Tensor, Dt: torch.Tensor) -> torch.Tensor:

        if self.use_dimensionless:
            u0 = self.problem.nondim_u(u0)

        out = self.net(torch.cat([u0, Dt], dim=-1))
        out = u0 + self.scaling_factor * Dt * out

        if self.use_dimensionless:
            out = self.problem.dim_u(out)

        return out