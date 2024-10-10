from typing import Dict, List, Optional 

import logging
import hydra
import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf

from modules.default import BaseLitModel
from integrators.integrator import Integrator
from utils.integrator_utils import instantiate_dynamical_ode_integrator
from problems.default import SeparableHamiltonianSystem


logger = logging.getLogger(__name__)

DEFAULT_LOSS_HPARAMS = {}
DEFAULT_METRIC_HPARAMS = {}
DEFAULT_SEQ_WEIGHTS = [0.0, 1.0]


def prepare_random_timesteps(batch_size: int, dist_config: DictConfig, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Prepares random timesteps based on a given distribution configuration.

    Args:
        batch_size (int): Number of random timesteps to generate.
        dist_config (DictConfig): Configuration for the distribution. Must contain 'type' and relevant parameters.
        device (str): Device to allocate the tensor to ('cpu' or 'cuda').
        dtype (torch.dtype): Data type of the resulting tensor.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 1) with generated timesteps.
    """
    if "type" not in dist_config:
        raise ValueError("dist_config should have a 'type' key specifying the distribution type.")
    
    dist_type = dist_config["type"]
    
    distribution_map = {
        "uniform": {
            "keys": ["min", "max"],
            "generator": lambda batch_size, config: torch.rand(batch_size, 1, device=device, dtype=dtype) * (config["max"] - config["min"]) + config["min"]
        },
        "normal": {
            "keys": ["mean", "std"],
            "generator": lambda batch_size, config: torch.randn(batch_size, 1, device=device, dtype=dtype) * config["std"] + config["mean"]
        },
    }

    if dist_type not in distribution_map:
        raise ValueError(f"Unsupported distribution type: {dist_type}. Supported types: {list(distribution_map.keys())}.")

    def validate_config(config, required_keys):
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing keys in dist_config for {dist_type} distribution: {missing_keys}")

    required_keys = distribution_map[dist_type]["keys"]
    validate_config(dist_config, required_keys)
    generator_func = distribution_map[dist_type]["generator"]
    return generator_func(batch_size, dist_config)


class BaseSolutionMap(BaseLitModel):
    """Base solution map model."""

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
        
    def extra_repr(self):
        return f"problem: {self.problem}\nuse_dimensionless: {self.use_dimensionless}"
    
    def _validate_nondim_methods(self):
        methods = ["nondim_u", "dim_u", "nondim_du", "dim_du"]
        for method in methods:
            if not callable(getattr(self.problem, method, None)):
                raise NotImplementedError(f"Method {method} is required for dimensionless calculations.")
    
    def _apply_nondim(self, u: torch.Tensor, deriv_mode: bool = False) -> torch.Tensor:
        if self.use_dimensionless:
            return self.problem.nondim_u(u) if not deriv_mode else self.problem.nondim_du(u)
        return u
    
    def _apply_dim(self, u: torch.Tensor, deriv_mode: bool = False) -> torch.Tensor:
        if self.use_dimensionless:
            return self.problem.dim_u(u) if not deriv_mode else self.problem.dim_du(u)
        return u

    def _initialize_hparams(self, hparams: DictConfig, default_hparams: dict):
        initialized_hparams = OmegaConf.create(default_hparams)
        return OmegaConf.merge(initialized_hparams, hparams) if hparams else initialized_hparams
    
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

    def forward(self, u0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass 
    
    def predict_sequence(self, u0: torch.Tensor, t: torch.Tensor, sequence_len: int) -> List[torch.Tensor]:
        pred_seq = [u0]
        u = u0
        for _ in range(sequence_len-1):
            u = self(u, t)
            pred_seq.append(u)
        return pred_seq
    
    def training_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx, sequence_len=2):
        u0, t, _, _ = self._unpack_batch(batch)
        pred_seq = self.predict_sequence(u0, t, sequence_len)
        batch["pred_seq"] = pred_seq
        return batch
    
    def model_step(self, batch: dict, batch_idx: int, stage: str = None):
        # unpack batch
        u0, t, true_seq, u0_unsup = self._unpack_batch(batch)

        # predict
        pred_seq = self.predict_sequence(u0, t, len(self.seq_weights))

        # initialize integrators if needed
        self._ensure_integrators_initialized()

        # compute losses
        sup_losses = self._compute_supervised_losses(pred_seq, true_seq, self.loss_hparams, self.loss_integrators)
        unsup_losses = self._compute_unsupervised_losses(u0_unsup, self.loss_hparams, self.loss_integrators)
        loss = self._compute_total_loss(sup_losses, unsup_losses)

        # compute metrics
        fitting_metrics = self._compute_fitting_metrics(pred_seq, true_seq)
        unsup_metrics = self._compute_unsupervised_losses(u0, self.metric_hparams, self.metric_integrators)

        # log losses and metrics
        metrics = {"loss": loss.detach()}
        metrics.update(self._prepare_metrics(sup_losses, suffix="_loss"))
        metrics.update(self._prepare_metrics(unsup_losses, suffix="_loss"))
        metrics.update(self._prepare_metrics(fitting_metrics))
        metrics.update(self._prepare_metrics(unsup_metrics, suffix="_err"))
        batch_size = len(u0)
        self._log_step(metrics, stage, loss, batch_size)

        return {"loss": loss, "batch_size": batch_size, "metrics": metrics}
        
    def _unpack_batch(self, batch: dict) -> tuple:
        if "supervised" in batch.keys():
            u0 = batch["supervised"]["input"]
            t = batch["supervised"]["Dt"]
            true_seq = batch["supervised"]["target_seq"]
            u0_unsup = batch["unsupervised"]["input"]
        else:
            u0 = batch["input"]
            t = batch["Dt"]
            true_seq = batch["target_seq"]
            u0_unsup = None
        return u0, t, true_seq, u0_unsup
    
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
        return instantiate_dynamical_ode_integrator(config["method"], config["stepsize"], config["nsteps"], self.problem.compute_ddx)
    
    def _compute_supervised_losses(self, pred_seq: List[torch.Tensor], true_seq: List[torch.Tensor], hparams: DictConfig, 
                                   integrators: dict, reduction: str = "mean") -> Dict[str, torch.Tensor]:
        losses = {}
        if "data_misfit" in hparams:
            loss, loss_per_steps = self._calc_data_misfit_error(pred_seq, true_seq, reduction)
            losses["data_misfit"] = loss
            for k, loss_per_step in enumerate(loss_per_steps):
                if loss_per_step is not None:
                    losses[f"data_misfit_step{k}"] = loss_per_step
        return losses

    def _compute_unsupervised_losses(self, u0: torch.Tensor, hparams: DictConfig, integrators: dict, reduction: str = "mean") -> Dict[str, torch.Tensor]:
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

        def prepare_u0(u0, batch_size):
            if u0 is None:
                if batch_size is None:
                    raise ValueError("Either u0 or batch_size should be provided.")
                return self._prepare_random_states(batch_size)
            return u0 
        
        losses = {}
        for loss_name, loss_fn in LOSS_FN_DICT.items():
            if loss_name in hparams:
                params = hparams[loss_name]
                if loss_name in {"additive", "commutative"}:
                    u0_ = prepare_u0(u0, params.get("batch_size", None))
                    s = self._prepare_random_timesteps(u0_.shape[0], params["s_dist"])
                    t = self._prepare_random_timesteps(u0_.shape[0], params["t_dist"])
                    losses[loss_name] = loss_fn(u0_, s, t, reduction)
                elif loss_name in {"numerical_residual", "residual", "additive_vv", "commutative_vv", "dyadic", "reversibility"}:
                    u0_ = prepare_u0(u0, params.get("batch_size", None))
                    t = self._prepare_random_timesteps(u0_.shape[0], params["t_dist"])
                    if loss_name in {"numerical_residual", "additive_vv", "commutative_vv"}:
                        losses[loss_name] = loss_fn(u0_, t, integrators[loss_name], reduction)
                    else:
                        losses[loss_name] = loss_fn(u0_, t, reduction)

        return losses
    
    def _compute_total_loss(self, sup_losses: Dict[str, torch.Tensor], unsup_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = sum(self.loss_hparams[loss_name]["strength"] * loss_value
               for loss_name, loss_value in {**sup_losses, **unsup_losses}.items()
               if loss_name in self.loss_hparams)
        return loss
    
    def _compute_fitting_metrics(self, pred_seq: List[torch.Tensor], true_seq: List[torch.Tensor], reduction: str = "mean") -> Dict[str, torch.Tensor]:
        metrics = {}
        for k, (u_pred, u_true) in enumerate(zip(pred_seq, true_seq)):
            errors = self.problem.compute_errors(u_pred, u_true, reduction)
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

    def _prepare_random_states(self, batch_size: int) -> torch.Tensor:
        return self.problem.random_states(batch_size).to(self.device).to(self.dtype)

    def _prepare_random_timesteps(self, batch_size: int, dist_config: DictConfig) -> torch.Tensor:
        return prepare_random_timesteps(batch_size, dist_config, self.device, self.dtype)

    def _calc_data_misfit_error(self, pred_seq: List[torch.Tensor], true_seq: List[torch.Tensor], reduction: str) -> tuple:
        errors = [None for _ in range(len(self.seq_weights))]
        for k, weight in enumerate(self.seq_weights):
            u_pred = pred_seq[k]
            u_true = true_seq[k]
            if weight != 0:
                errors[k] = self._calc_error_uw(u_pred, u_true, reduction, use_mse=False)
        stacked_errors = torch.stack(
            [errors[k] * weight for k, weight in enumerate(self.seq_weights) if errors[k] is not None]
        )
        weighted_sum = stacked_errors.sum(dim=0) / sum(self.seq_weights)
        return weighted_sum, errors

    def _calc_residual_error(self, u0: torch.Tensor, t: torch.Tensor, reduction: str) -> torch.Tensor:
        du = self.problem.compute_du(self(u0, t))
        du_pred = torch.func.vmap(torch.func.jacrev(self, argnums=1))(u0, t)  # (bs, 2*dof, 1)
        du_pred = du_pred.squeeze(-1)  # (bs, 2*dof)
        return self._calc_error_dudw(du_pred, du, reduction)

    def _calc_numerical_residual_error(self, u0: torch.Tensor, t: torch.Tensor, integrator: Integrator, reduction: str) -> torch.Tensor:
        residual = integrator.compute_residual(self(u0, t), self(u0, t+integrator.h), t)
        return self._calc_error_uw(residual, torch.zeros_like(residual), reduction, use_mse=True)
    
    def _calc_commutative_error(self, u0: torch.Tensor, s: torch.Tensor, t: torch.Tensor, reduction: str) -> torch.Tensor:
        return self._calc_error_uw(self(self(u0, t), s), self(self(u0, s), t), reduction, use_mse=True)
    
    def _calc_additive_error(self, u0: torch.Tensor, s: torch.Tensor, t: torch.Tensor, reduction: str) -> torch.Tensor:
        return self._calc_error_uw(self(self(u0, t), s), self(u0, s+t), reduction, use_mse=True)
    
    def _calc_commutative_vv_error(self, u0: torch.Tensor, t: torch.Tensor, integrator: Integrator, reduction: str) -> torch.Tensor:
        return self._calc_error_uw(integrator(self(u0, t)), self(integrator(u0), t), reduction, use_mse=True)
    
    def _calc_additive_vv_error(self, u0: torch.Tensor, t: torch.Tensor, integrator: Integrator, reduction: str) -> torch.Tensor:
        return self._calc_error_uw(integrator(self(u0, t)), self(u0, t+integrator.T), reduction, use_mse=True)

    def _calc_dyadic_error(self, u0: torch.Tensor, t: torch.Tensor, reduction: str) -> torch.Tensor:
        return self._calc_error_uw(self(self(u0, t), t), self(u0, 2*t), reduction, use_mse=True)

    def _calc_reversibiliy_error(self, u0: torch.Tensor, t: torch.Tensor, reduction: str) -> torch.Tensor:
        return self._calc_error_uw(self(self(u0, t), -t), u0, reduction, use_mse=True)

    def _calc_error_uw(self, u: torch.Tensor, w: torch.Tensor, reduction: str, use_mse: bool) -> torch.Tensor:
        u, w = self._apply_nondim(u), self._apply_nondim(w)
        if use_mse:
            return nn.functional.mse_loss(u, w, reduction=reduction)
        else:
            errors = self.loss_fn(u, w)
            if reduction == "mean":
                return errors.mean()  # scalar
            elif reduction == "sum":
                return errors.sum()  # scalar
            elif reduction == "none":
                return errors  # (bs, dim)
            else:
                raise ValueError(f"Unsupported reduction: {reduction}")
    
    def _calc_error_dudw(self, du: torch.Tensor, dw: torch.Tensor, reduction: str) -> torch.Tensor:
        du, dw = self._apply_nondim(du, deriv_mode=True), self._apply_nondim(dw, deriv_mode=True)
        return nn.functional.mse_loss(du, dw, reduction=reduction)
