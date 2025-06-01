from typing import Dict, List, Optional 

import logging
import hydra
import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf

from modules.default import BaseLitModel
from modules.collocation_handler import CollocationPoints, CollocationPointsGenerator
from integrators.integrator import Integrator
from utils.integrator_utils import instantiate_first_order_ode_integrator, instantiate_dynamical_ode_integrator
from problems.default import SeparableHamiltonianSystem


logger = logging.getLogger(__name__)

DEFAULT_LOSS_HPARAMS = {}
DEFAULT_METRIC_HPARAMS = {}
DEFAULT_SEQ_WEIGHTS = [0.0, 1.0]


class BaseSolutionMap(BaseLitModel):
    """Base solution map model."""

    def __init__(
            self,
            problem: DictConfig,
            loss: DictConfig,
            loss_hparams: DictConfig = None,
            metric_hparams: DictConfig = None,
            use_dimensionless: bool = True,
            problem_param_keys: Optional[List[str]] = None,
            **kwargs
        ) -> None:
        super(BaseSolutionMap, self).__init__(**kwargs)

        self.problem: SeparableHamiltonianSystem = hydra.utils.instantiate(problem)
        self.loss_fn: nn.Module = hydra.utils.instantiate(loss, reduction="none")
        self.loss_hparams = self._initialize_hparams(loss_hparams, DEFAULT_LOSS_HPARAMS)
        self.metric_hparams = self._initialize_hparams(metric_hparams, DEFAULT_METRIC_HPARAMS)
        logger.info(f"Initialized loss hparams: {self.loss_hparams}")
        logger.info(f"Initialized metric hparams: {self.metric_hparams}")
        self.loss_integrators = None
        self.metric_integrators = None
        self.set_seq_weights(DEFAULT_SEQ_WEIGHTS)

        self.use_dimensionless = use_dimensionless
        if self.use_dimensionless:
            self._validate_nondim_methods()

        self.problem_param_keys = problem_param_keys if problem_param_keys is not None else []

    def extra_repr(self):
        return f"problem: {self.problem}\nuse_dimensionless: {self.use_dimensionless}\nproblem_param_keys: {self.problem_param_keys}"
    
    def _validate_nondim_methods(self):
        methods = ["nondim_u", "dim_u", "nondim_du", "dim_du"]
        for method in methods:
            if not callable(getattr(self.problem, method, None)):
                raise NotImplementedError(f"Method {method} is required for dimensionless calculations.")
    
    def _apply_nondim(self, u: torch.Tensor, p: Dict[str, torch.Tensor], deriv_mode: bool = False) -> torch.Tensor:
        if self.use_dimensionless:
            return self.problem.nondim_u(u, p) if not deriv_mode else self.problem.nondim_du(u, p)
        return u
    
    def _apply_dim(self, u: torch.Tensor, p: Dict[str, torch.Tensor], deriv_mode: bool = False) -> torch.Tensor:
        if self.use_dimensionless:
            return self.problem.dim_u(u, p) if not deriv_mode else self.problem.dim_du(u, p)
        return u

    def _initialize_hparams(self, hparams: DictConfig, default_hparams: dict):
        initialized_hparams = OmegaConf.create(default_hparams)
        return OmegaConf.merge(initialized_hparams, hparams) if hparams else initialized_hparams
    
    def get_loss_hparams(self) -> DictConfig:
        return self.loss_hparams

    def update_loss_hparams(self, hparams: DictConfig):
        if hparams is not None:
            self.loss_hparams = OmegaConf.merge(self.loss_hparams, hparams)
            logger.info(f"Updated loss hparams: {self.loss_hparams}")
 
    def update_metric_hparams(self, hparams: DictConfig):
        if hparams is not None:
            self.metric_hparams = OmegaConf.merge(self.metric_hparams, hparams)
            logger.info(f"Updated metric hparams: {self.metric_hparams}")
    
    def set_seq_weights(self, seq_weights: List[float]):
        self.seq_weights = seq_weights

    def prepare_params_input(self, p: Dict[str, torch.Tensor]) -> torch.Tensor:
        p = {} if p is None else p
        params_to_use = []
        if self.problem_param_keys:
            for key in self.problem_param_keys:
                if key in p:
                    params_to_use.append(p[key])
                else:
                    raise KeyError(f"Expected parameter '{key}' not found in input dictionary p.")
        return params_to_use
        
    def forward(self, u0: torch.Tensor, t: torch.Tensor, p: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass 
    
    def predict_sequence(self, u0: torch.Tensor, t: torch.Tensor, p: Dict[str, torch.Tensor], sequence_len: int) -> List[torch.Tensor]:
        pred_seq = [u0]
        u = u0
        for _ in range(sequence_len-1):
            u = self(u, t, p)
            pred_seq.append(u)
        return pred_seq
    
    def training_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx, sequence_len=2):
        u0, t, _, _, _, _, _ = self._unpack_batch(batch)
        pred_seq = self.predict_sequence(u0, t, sequence_len)
        batch["pred_seq"] = pred_seq
        return batch
    
    def model_step(self, batch: dict, batch_idx: int, stage: str = None):
        # unpack batch
        u0, t, p, true_seq, u0_unsup, t_unsup, p_unsup = self._unpack_batch(batch)

        # initialize integrators if needed
        self._ensure_integrators_initialized()

        # generate collocation points for unsupervised losses
        generator = CollocationPointsGenerator(
            self.device, self.dtype, self.problem.random_states, 
            lambda n_samples: self.problem.random_params(n_samples, self.problem_param_keys))
        collocation_points = generator.generate_collocation_points(self.loss_hparams, u0_unsup, t_unsup, p_unsup)

        # initialize metrics and prediction container
        metrics = {}
        pred_seq = None
        
        if self.automatic_optimization:  # lightning handles optimization
            
            # forward pass
            pred_seq = self.predict_sequence(u0, t, p, len(self.seq_weights))

            # compute losses
            sup_losses = self._compute_supervised_losses(pred_seq, true_seq, p, self.loss_hparams, self.loss_integrators)
            unsup_losses = self._compute_unsupervised_losses(collocation_points, self.loss_hparams, self.loss_integrators)
            loss = self._compute_total_loss(sup_losses, unsup_losses)
                
            # log losses 
            metrics.update({"loss": loss.detach()})
            metrics.update(self._prepare_metrics(sup_losses, suffix="_loss"))
            metrics.update(self._prepare_metrics(unsup_losses, suffix="_loss"))

        else:  # manual optimization (required for L-BFGS)
            
            # get optimizer
            opt = self.optimizers()
            
            # define loss computation closure
            def closure(backward=True):
                nonlocal pred_seq  # allow modification of outer scope variable

                # zero grad
                opt.zero_grad()

                # forward pass
                pred_seq = self.predict_sequence(u0, t, p, len(self.seq_weights))

                # compute losses
                sup_losses = self._compute_supervised_losses(pred_seq, true_seq, p, self.loss_hparams, self.loss_integrators)
                unsup_losses = self._compute_unsupervised_losses(collocation_points, self.loss_hparams, self.loss_integrators)
                loss = self._compute_total_loss(sup_losses, unsup_losses)
                
                # log losses 
                metrics.update({"loss": loss.detach()})
                metrics.update(self._prepare_metrics(sup_losses, suffix="_loss"))
                metrics.update(self._prepare_metrics(unsup_losses, suffix="_loss"))

                # backward pass if needed
                if backward:
                    self.manual_backward(loss)

                return loss
            
            # execute optimization step or forward pass
            if stage == "train":
                loss = opt.step(closure)
            else:
                loss = closure(backward=False)
            
        # compute metrics
        fitting_metrics = self._compute_fitting_metrics(pred_seq, true_seq, p)
        collocation_points = generator.generate_collocation_points(self.metric_hparams, u0_unsup, t_unsup, p_unsup)
        unsup_metrics = self._compute_unsupervised_losses(collocation_points, self.metric_hparams, self.metric_integrators)
        
        # log metrics
        metrics.update(self._prepare_metrics(fitting_metrics))
        metrics.update(self._prepare_metrics(unsup_metrics, suffix="_err"))
        batch_size = len(u0)
        self._log_step(metrics, stage, loss, batch_size)

        if not self.automatic_optimization:  # manually perform scheduler step during training
            if stage == "train":
                if self.trainer.lr_scheduler_configs:
                    config = self.trainer.lr_scheduler_configs[0]
                    sch = config.scheduler
                    if config.interval == "step" or (config.interval == "epoch" and self.trainer.is_last_batch):
                        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            sch.step(self.trainer.callback_metrics["step_loss"])
                        else:
                            sch.step()

        return {"loss": loss, "batch_size": batch_size, "metrics": metrics}
        
    def _unpack_batch(self, batch: dict) -> tuple:
        if "supervised" in batch.keys():
            u0 = batch["supervised"]["input"]
            t = batch["supervised"]["Dt"]
            p = batch["supervised"].get("params", None)
            true_seq = batch["supervised"]["target_seq"]
            u0_unsup = batch["unsupervised"]["input"]
            t_unsup = batch["unsupervised"]["Dt"]
            p_unsup = batch["unsupervised"].get("params", None)
        else:
            u0 = batch["input"]
            t = batch["Dt"]
            p = batch.get("params", None)
            true_seq = batch["target_seq"]
            u0_unsup = batch["input"]
            t_unsup = batch["Dt"]
            p_unsup = batch.get("params", None)
        return u0, t, p, true_seq, u0_unsup, t_unsup, p_unsup
    
    def _ensure_integrators_initialized(self):
        if self.loss_integrators is None:
            self.loss_integrators = self._initialize_integrators(self.loss_hparams, suffix="_loss")
        if self.metric_integrators is None:
            self.metric_integrators = self._initialize_integrators(self.metric_hparams, suffix="_err")
        
    def reinitialize_loss_integrators(self):
        self.loss_integrators = self._initialize_integrators(self.loss_hparams, suffix="_loss")

    def _initialize_integrators(self, hparams: DictConfig, suffix: str = "") -> dict:
        integrators = {}
        for key in ["numerical_residual", "additive_vv", "commutative_vv"]:
            if key in hparams:
                params = hparams[key]
                if "integrator" not in params:
                    raise ValueError(f"{key}.integrator is required.")
                integrators[key] = self._instantiate_integrator(params["integrator"])
                logger.info(f"Initialized {key}{suffix} integrator: {integrators[key]}")
        return integrators

    def _instantiate_integrator(self, config: DictConfig):
        for key in ["method", "stepsize", "nsteps"]:
            if key not in config:
                raise ValueError(f"{key} is required in integrator configuration.")
        if callable(getattr(self.problem, "compute_ddx", None)):
            return instantiate_dynamical_ode_integrator(config["method"], config["stepsize"], config["nsteps"], self.problem.compute_ddx)
        else:
            return instantiate_first_order_ode_integrator(config["method"], config["stepsize"], config["nsteps"], self.problem.compute_du)

    def _compute_supervised_losses(self, pred_seq: List[torch.Tensor], true_seq: List[torch.Tensor], p: Dict[str, torch.Tensor], 
                                   hparams: DictConfig, integrators: dict, reduction: str = "mean") -> Dict[str, torch.Tensor]:
        losses = {}
        if "data_misfit" in hparams:
            loss, loss_per_steps = self._calc_data_misfit_error(pred_seq, true_seq, p, reduction)
            losses["data_misfit"] = loss
            for k, loss_per_step in enumerate(loss_per_steps):
                if loss_per_step is not None:
                    losses[f"data_misfit_step{k}"] = loss_per_step
        return losses

    def _compute_unsupervised_losses(self, collocation_points: Dict[str, CollocationPoints], hparams: DictConfig, 
                                     integrators: dict, reduction: str = "mean") -> Dict[str, torch.Tensor]:
        LOSS_FN_DICT = {
            "numerical_residual": self._calc_numerical_residual_error,
            "residual": self._calc_residual_error,
            "additive": self._calc_additive_error,
            "commutative": self._calc_commutative_error,
            "additive_vv": self._calc_additive_vv_error,
            "commutative_vv": self._calc_commutative_vv_error,
            "dyadic": self._calc_dyadic_error,
            "reversibility": self._calc_reversibiliy_error,
        }

        losses = {}
        for loss_name, loss_fn in LOSS_FN_DICT.items():
            if loss_name in hparams:
                points = collocation_points[loss_name]
                params = hparams[loss_name]
                use_mse = params.get("use_mse", True)
                if loss_name in {"additive", "commutative"}:
                    losses[loss_name] = loss_fn(points.u0, points.s, points.t, reduction, use_mse)
                elif loss_name in {"numerical_residual", "additive_vv", "commutative_vv"}:
                    if params.get("adaptive", False):
                        losses[loss_name] = loss_fn(points.u0, points.t, points.p, integrators[loss_name], reduction, use_mse, points.p["h_max"])
                    else:
                        losses[loss_name] = loss_fn(points.u0, points.t, points.p, integrators[loss_name], reduction, use_mse)
                else:  # residual, dyadic, reversibility
                    losses[loss_name] = loss_fn(points.u0, points.t, points.p, reduction, use_mse)  
        return losses
    
    def _compute_total_loss(self, sup_losses: Dict[str, torch.Tensor], unsup_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = sum(self.loss_hparams[loss_name]["strength"] * loss_value
               for loss_name, loss_value in {**sup_losses, **unsup_losses}.items()
               if loss_name in self.loss_hparams)
        return loss
    
    def _compute_fitting_metrics(self, pred_seq: List[torch.Tensor], true_seq: List[torch.Tensor], 
                                 p: Dict[str, torch.Tensor], reduction: str = "mean") -> Dict[str, torch.Tensor]:
        metrics = {}
        for k, (u_pred, u_true) in enumerate(zip(pred_seq, true_seq)):
            errors = self.problem.compute_errors(u_pred, u_true, p, reduction)
            metrics.update({f"{name}_step{k}": value for name, value in errors.items()})
        return metrics
    
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

    def _calc_data_misfit_error(self, pred_seq: List[torch.Tensor], true_seq: List[torch.Tensor], 
                                p: Dict[str, torch.Tensor], reduction: str) -> tuple:
        errors = [None for _ in range(len(self.seq_weights))]
        for k, weight in enumerate(self.seq_weights):
            u_pred = pred_seq[k]
            u_true = true_seq[k]
            if weight != 0:
                errors[k] = self._calc_error_uw(u_pred, u_true, p, reduction, use_mse=False)
        stacked_errors = torch.stack(
            [errors[k] * weight for k, weight in enumerate(self.seq_weights) if errors[k] is not None]
        )
        weighted_sum = stacked_errors.sum(dim=0) / sum(self.seq_weights)
        return weighted_sum, errors

    def _calc_residual_error(self, u0: torch.Tensor, t: torch.Tensor, p: Dict[str, torch.Tensor], reduction: str, use_mse: bool) -> torch.Tensor:
        du = self.problem.compute_du(self(u0, t, p), t, p)
        du_pred = self._calc_dPhidt(u0, t, p)
        return self._calc_error_dudw(du_pred, du, p, reduction, use_mse)

    def _calc_numerical_residual_error(
            self, u0: torch.Tensor, t: torch.Tensor, p: Dict[str, torch.Tensor], integrator: Integrator, reduction: str, use_mse: bool, 
            h_max: torch.Tensor = None) -> torch.Tensor:
        if h_max is not None:
            h = torch.minimum(h_max/4, torch.ones_like(h_max) * integrator.h)
            u_n_plus_1, pred_u_n_plus_1 = integrator.compute_residual(self(u0, t, p), self(u0, t+h, p), t, h, p)
            residual = (u_n_plus_1 - pred_u_n_plus_1) / h 
            return self._calc_error_uw(residual, torch.zeros_like(residual), p, reduction, use_mse)
        else:
            u_n_plus_1, pred_u_n_plus_1 = integrator.compute_residual(self(u0, t, p), self(u0, t+integrator.h, p), t, integrator.h, p)
            return self._calc_error_uw(u_n_plus_1, pred_u_n_plus_1, p, reduction, use_mse) / (integrator.h ** 2)
            # return self._calc_error_dudw(u_n_plus_1/integrator.h, pred_u_n_plus_1/integrator.h, reduction, use_mse)
        
    def _calc_commutative_error(
            self, u0: torch.Tensor, s: torch.Tensor, t: torch.Tensor, reduction: str, use_mse: bool) -> torch.Tensor:
        return self._calc_error_uw(self(self(u0, t), s), self(self(u0, s), t), reduction, use_mse)
    
    def _calc_additive_error(
            self, u0: torch.Tensor, s: torch.Tensor, t: torch.Tensor, reduction: str, use_mse: bool) -> torch.Tensor:
        return self._calc_error_uw(self(self(u0, t), s), self(u0, s+t), reduction, use_mse)
    
    def _calc_commutative_vv_error(
            self, u0: torch.Tensor, t: torch.Tensor, integrator: Integrator, reduction: str, use_mse: bool) -> torch.Tensor:
        return self._calc_error_uw(integrator(self(u0, t)), self(integrator(u0), t), reduction, use_mse)
    
    def _calc_additive_vv_error(
            self, u0: torch.Tensor, t: torch.Tensor, integrator: Integrator, reduction: str, use_mse: bool) -> torch.Tensor:
        return self._calc_error_uw(integrator(self(u0, t)), self(u0, t+integrator.T), reduction, use_mse)

    def _calc_dyadic_error(self, u0: torch.Tensor, t: torch.Tensor, reduction: str, use_mse: bool) -> torch.Tensor:
        return self._calc_error_uw(self(self(u0, t), t), self(u0, 2*t), reduction, use_mse)

    def _calc_reversibiliy_error(self, u0: torch.Tensor, t: torch.Tensor, reduction: str, use_mse: bool) -> torch.Tensor:
        return self._calc_error_uw(self(self(u0, t), -t), u0, reduction, use_mse)

    def _calc_error_uw(self, u: torch.Tensor, w: torch.Tensor, p: Dict[str, torch.Tensor], reduction: str, use_mse: bool) -> torch.Tensor:
        u, w = self._apply_nondim(u, p), self._apply_nondim(w, p)
        if use_mse:
            return nn.functional.mse_loss(u, w, reduction=reduction)
        else:
            if isinstance(self.loss_fn, nn.MSELoss):
                errors = self.loss_fn(u, w)
            else:  # for custom loss functions
                errors = self.loss_fn(u, w, p)  
            if reduction == "mean":
                return errors.mean()  # scalar
            elif reduction == "sum":
                return errors.sum()  # scalar
            elif reduction == "none":
                return errors  # (bs, dim)
            else:
                raise ValueError(f"Unsupported reduction: {reduction}")
    
    def _calc_error_dudw(self, du: torch.Tensor, dw: torch.Tensor, p: Dict[str, torch.Tensor], reduction: str, use_mse: bool) -> torch.Tensor:
        du, dw = self._apply_nondim(du, p, deriv_mode=True), self._apply_nondim(dw, p, deriv_mode=True)
        if use_mse:
            return nn.functional.mse_loss(du, dw, reduction=reduction)
        else:
            raise NotImplementedError("Non-MSE loss is not supported for derivative errors.")

    def _calc_dPhidt(self, u0: torch.Tensor, t: torch.Tensor, p: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.func.vmap(torch.func.jacrev(self, argnums=1))(u0, t, p).squeeze(-1)  # (bs, 2*dof)

    def _calc_dPhidt_at_t(self, u0: torch.Tensor, p: Dict[str, torch.Tensor], t: float) -> torch.Tensor:
        t = torch.full_like(u0[..., :1], t)
        return torch.func.vmap(torch.func.jacrev(self, argnums=1))(u0, t, p).squeeze(-1)  # (bs, 2*dof)
    
    def _calc_d2Phidt2(self, u0: torch.Tensor, t: torch.Tensor, p: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.func.vmap(torch.func.hessian(self, argnums=1))(u0, t, p).squeeze(-1).squeeze(-1)  # (bs, 2*dof)
    
    def _calc_d2Phidt2_at_t(self, u0: torch.Tensor, p: Dict[str, torch.Tensor], t: float) -> torch.Tensor:
        t = torch.full_like(u0[..., :1], t)
        return torch.func.vmap(torch.func.hessian(self, argnums=1))(u0, t, p).squeeze(-1).squeeze(-1)  # (bs, 2*dof)
    
    def _calc_dPhidu(self, u0: torch.Tensor, t: torch.Tensor, p: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.func.vmap(torch.func.jacrev(self, argnums=0))(u0, t, p)  # (bs, 2*dof, 2*dof)

    def _calc_det_dPhidu(self, u0: torch.Tensor, t: torch.Tensor, p: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.det(self._calc_dPhidu(u0, t, p))  # (bs,)
