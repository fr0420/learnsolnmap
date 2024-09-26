from typing import List, Dict

import hydra
import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf

from modules.default import BaseLitModel
from integrators.integrators import instantiate_symplectic_integrator, SymplecticIntegrator
from problems.default import SeparableHamiltonianSystem



DEFAULT_LOSS_HPARAMS = {}
DEFAULT_METRIC_HPARAMS = {}


class BaseSolutionMap(BaseLitModel):
    """Base fixed Dt solution map model."""

    def __init__(
            self,
            Delta_t: float,
            problem: DictConfig,
            loss: DictConfig,
            loss_hparams: DictConfig,
            metric_hparams: DictConfig = None,
            use_dimensionless: bool = True,
            use_dimensionless_for_loss: bool = True,
            **kwargs
        ) -> None:
        super(BaseSolutionMap, self).__init__(**kwargs)
        
        self.Delta_t: float = Delta_t

        self.problem: SeparableHamiltonianSystem = hydra.utils.instantiate(problem)

        self.loss_fn: nn.Module = hydra.utils.instantiate(loss)
        self.loss_hparams: DictConfig = self._initialize_hparams(loss_hparams, DEFAULT_LOSS_HPARAMS)
        self.metric_hparams: DictConfig = self._initialize_hparams(metric_hparams, DEFAULT_METRIC_HPARAMS)

        self.use_dimensionless: bool = use_dimensionless
        self.use_dimensionless_for_loss: bool = use_dimensionless_for_loss
        if self.use_dimensionless or self.use_dimensionless_for_loss:
            self._validate_nondim_methods()
        
        self.seq_weights = None
        self.loss_integrators = None
        self.metric_integrators = None

    def _validate_nondim_methods(self):
        methods = ["nondim_u", "dim_u", "nondim_du", "dim_du"]
        for method in methods:
            assert callable(getattr(self.problem, method, None)), f"Method {method} is not implemented."
    
    def _initialize_hparams(self, hparams: DictConfig, default_hparams: dict) -> DictConfig:
        initialized_hparams = OmegaConf.create(default_hparams)
        return OmegaConf.merge(initialized_hparams, hparams) if hparams else initialized_hparams

    def update_loss_hparams(self, hparams: DictConfig):
        if hparams is not None:
            self.loss_hparams = OmegaConf.merge(self.loss_hparams, hparams)
 
    def update_metric_hparams(self, hparams: DictConfig):
        if hparams is not None:
            self.metric_hparams = OmegaConf.merge(self.metric_hparams, hparams) 
       
    def set_seq_weights(self, weights: torch.Tensor):
        self.seq_weights = weights.to(self.dtype).to(self.device)

    def forward(self, u0: torch.Tensor, sequence_len: int) -> List[torch.Tensor]:
        pass 
    
    def step(self, u0: torch.Tensor) -> torch.Tensor:
        return self(u0, 2)[1]
    
    def training_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, sequence_len=6):
        u0, _, _ = self._unpack_batch(batch)
        pred_seq = self(u0, sequence_len)
        batch["pred_seq"] = pred_seq
        return batch

    def model_step(self, batch, batch_idx, stage=None):
        # unpack batch
        u0, true_seq, u0_unsup = self._unpack_batch(batch)

        # predict 
        pred_seq = self(u0, len(self.seq_weights))

        # initialize integrators if needed
        self._ensure_integrators_initialized()

        # compute losses
        sup_losses = self._compute_supervised_losses(pred_seq, true_seq, self.loss_hparams, self.loss_integrators) 
        unsup_losses = self._compute_unsupervised_losses(u0_unsup, self.loss_hparams, self.loss_integrators)
        loss = self._compute_total_loss(sup_losses, unsup_losses)
        
        # compute metrics
        fitting_metrics = self._compute_fitting_metrics(pred_seq, true_seq)
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
        if "supervised" in batch.keys():
            u0 = batch["supervised"]["input"]
            true_seq = batch["supervised"]["target_seq"]
            u0_unsup = batch["unsupervised"]["input"]
        else:
            u0 = batch["input"]
            true_seq = batch["target_seq"]
            u0_unsup = None
        return u0, true_seq, u0_unsup
    
    def _ensure_integrators_initialized(self):
        if self.loss_integrators is None:
            self.loss_integrators = self._initialize_integrators(self.loss_hparams)
        if self.metric_integrators is None:
            self.metric_integrators = self._initialize_integrators(self.metric_hparams)
    
    def _initialize_integrators(self, hparams: DictConfig) -> dict:

        def validate_integrator_params(params, key):
            assert params.get("integrator"), f"{key}.integrator is required."
            if key == "integrator_informed_misfit":
                assert params.get("fine_integrator"), f"{key}.fine_integrator is required."
                total_stepsize_integrator = params["integrator"]["stepsize"] * params["integrator"]["nsteps"]
                total_stepsize_fine = params["fine_integrator"]["stepsize"] * params["fine_integrator"]["nsteps"]
                assert total_stepsize_integrator == total_stepsize_fine, \
                    f"{key}: Integrator and fine integrator must have matching total step sizes."

        integrators = {}
        for key in ["integrator_informed_misfit", "commutative"]:
            if key in hparams:
                params = hparams[key]
                validate_integrator_params(params, key)
                integrator = self._instantiate_integrator(params["integrator"])
                if key == "integrator_informed_misfit":
                    fine_integrator = self._instantiate_integrator(params["fine_integrator"])
                    integrators[key] = (integrator, fine_integrator)
                else:
                    integrators[key] = integrator

        return integrators

    def _instantiate_integrator(self, config: DictConfig):
        return instantiate_symplectic_integrator(config["method"], config["stepsize"], config["nsteps"], self.problem.compute_ddx)
    
    def _compute_supervised_losses(self, pred_seq: List[torch.Tensor], true_seq: List[torch.Tensor], 
                                  hparams: DictConfig, integrators: dict) -> Dict[str, torch.Tensor]:
        losses = {}
        if "misfit" in hparams:
            loss, loss_per_steps = self._calc_misfit_errors(pred_seq, true_seq)
            losses["misfit"] = loss
            losses.update({f"misfit_step{t}": loss_per_steps[t] for t in range(len(loss_per_steps))})
        if "integrator_informed_misfit" in hparams:
            integrator, fine_integrator = integrators["integrator_informed_misfit"]
            loss, _ = self._calc_integrator_informed_misfit_errors(
                pred_seq, true_seq, integrator, fine_integrator, hparams["integrator_informed_misfit"]["K"])
            losses["integrator_informed_misfit"] = loss
        return losses 

    def _compute_unsupervised_losses(self, u0: torch.Tensor, hparams: DictConfig, integrators: dict) ->  Dict[str, torch.Tensor]:
        losses = {}

        if "commutative" in hparams:
            u0_ = u0 if u0 is not None else self._prepare_random_states(hparams["commutative"]["batch_size"])
            loss = self._calc_commutative_error(u0_, integrators["commutative"], hparams["commutative"]["K"])
            losses["commutative"] = loss
        
        return losses
    
    def _compute_total_loss(self, sup_losses: Dict[str, torch.Tensor], unsup_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = sum(self.loss_hparams[loss_name]["strength"] * loss_value
               for loss_name, loss_value in {**sup_losses, **unsup_losses}.items()
               if loss_name in self.loss_hparams)
        return loss
    
    def _compute_fitting_metrics(self, pred_seq: List[torch.Tensor], true_seq: List[torch.Tensor]) -> dict:
        metrics = {}
        for t, (u_pred, u_true) in enumerate(zip(pred_seq, true_seq)):
            errors = self.problem.compute_errors(u_pred, u_true, reduction="mean")
            for name, value in errors.items():
                metrics[f"{name}_step{t}"] = value
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

    def _calc_u_misfit(self, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.use_dimensionless_for_loss:
            return self.loss_fn(self.problem.nondim_u(u), self.problem.nondim_u(w))
        else:
            return self.loss_fn(u, w)

    def _prepare_random_states(self, batch_size: int) -> torch.Tensor:
        return self.problem.random_states(batch_size).to(self.device).to(self.dtype)
    
    def _calc_misfit_errors(self, pred_seq: List[torch.Tensor], true_seq: List[torch.Tensor]) -> tuple:
        errors = torch.zeros(len(self.seq_weights)).to(pred_seq[0])
        for t in range(len(self.seq_weights)):
            ut_pred = pred_seq[t]
            ut_true = true_seq[t]
            if self.seq_weights[t] != 0:
                errors[t] = self._calc_u_misfit(ut_pred, ut_true)
        weighted_sum = errors @ self.seq_weights / torch.sum(self.seq_weights)
        return weighted_sum, errors
    
    def _calc_integrator_informed_misfit_errors(self, pred_seq: List[torch.Tensor], true_seq: List[torch.Tensor], integrator: SymplecticIntegrator, 
                                              fine_integrator: SymplecticIntegrator, K: int) -> torch.Tensor: 
        errors = torch.zeros(len(self.seq_weights)).to(pred_seq[0])
        for t in range(len(self.seq_weights)):
            ut_pred = pred_seq[t]
            ut_true = true_seq[t]
            if self.seq_weights[t] != 0:
                errors[t] = self._calc_integrator_informed_misfit_error(ut_pred, ut_true, integrator, fine_integrator, K)
        weighted_sum = errors @ self.seq_weights / torch.sum(self.seq_weights)
        return weighted_sum, errors

    def _calc_integrator_informed_misfit_error(self, u_pred: torch.Tensor, u_true: torch.Tensor, integrator: SymplecticIntegrator, 
                                              fine_integrator: SymplecticIntegrator, K: int) -> torch.Tensor: 
        # 1/K * sum_{k=1}^{K} ||VV_h^k(Phi_Dt(u0)) - phi_h^k(phi_Dt(u0))||^2
        preds, targets = [], []
        u = u_pred
        for _ in range(K):
            u = integrator(u)
            preds.append(u)
        u = u_true
        for _ in range(K):
            u = fine_integrator(u)
            targets.append(u)
        errors = [self._calc_u_misfit(pred, target) for pred, target in zip(preds, targets)]
        return torch.stack(errors).mean()

    def _calc_commutative_error(self, u0: torch.Tensor, integrator: SymplecticIntegrator, K: int) -> torch.Tensor:
        # 1/K * sum_{k=1}^{K} ||VV_h^k(Phi_Dt(u0)) - Phi_Dt(VV_h^k(u0))||^2
        preds1, preds2 = [], []
        u = self.step(u0)
        for _ in range(K):
            u = integrator(u)
            preds1.append(u)
        u = u0 
        for _ in range(K):
            u = integrator(u)
            preds2.append(self.step(u))
        errors = [self._calc_u_misfit(pred1, pred2) for pred1, pred2 in zip(preds1, preds2)]
        return torch.stack(errors).mean()


class SolutionMap(BaseSolutionMap):
    """Solution map."""

    def __init__(self, network: DictConfig, **kwargs):
        super(SolutionMap, self).__init__(**kwargs)
        
        self.save_hyperparameters(logger=False)
        
        self.i2h = hydra.utils.instantiate(network.i2h)
        self.h2h = hydra.utils.instantiate(network.h2h)
        self.h2o = hydra.utils.instantiate(network.h2o)

        if self.weight_init is not None:
            self._init_weights()

    def forward(self, u0, sequence_len):
        res = []

        if self.use_dimensionless:
            u0 = self.problem.nondim_u(u0)

        hidden = self.i2h(u0)
        out = self.h2o(hidden)
        if self.use_dimensionless:
            out = self.problem.dim_u(out)
        res.append(out)

        for _ in range(sequence_len-1):
            hidden = self.h2h(hidden)
            out = self.h2o(hidden)
            if self.use_dimensionless:
                out = self.problem.dim_u(out)
            res.append(out)

        return res
    
    def freeze_encoder_decoder(self):
        for param in self.i2h.parameters():
            param.requires_grad = False
        for param in self.h2o.parameters():
            param.requires_grad = False


# class BoostedSolutionMap(BaseSolutionMap):
#     """Boosted solution map."""

#     def __init__(self, network: DictConfig, phi: DictConfig, 
#                  forward_type: str, learn_residual: bool, scaling_factor: float, **kwargs):
#         super(BoostedSolutionMap, self).__init__(**kwargs)
        
#         self.save_hyperparameters(logger=False)

#         self.net = hydra.utils.instantiate(network)
#         # self.friction = FrictionBlock()

#         if self.weight_init is not None:
#             self._init_weights()

#         checkpoint = torch.load(phi.ckpt_path, map_location="cpu")
#         self.phi = hydra.utils.instantiate(phi.module, **checkpoint["hyper_parameters"], _recursive_=False)
#         self.phi.load_state_dict(checkpoint["state_dict"], strict=False)
#         for param in self.phi.parameters():
#             param.requires_grad = False

#         self.forward_type = forward_type
#         self.learn_residual = learn_residual
#         if self.learn_residual:
#             self.scaling_factor = nn.Parameter(torch.tensor(scaling_factor), requires_grad=True)

#     def forward_1step(self, u):
#         phi_u = self.phi.forward_1step(u)
#         if self.use_dimensionless:
#             phi_u = self.nondimensionalize(phi_u)
        
#         if self.forward_type == "simple":
#             out = self.net(phi_u)
#         elif self.forward_type == "stacked":
#             if self.use_dimensionless:
#                 u = self.nondimensionalize(u)
#             out = self.net(torch.cat((u, phi_u), dim=-1))
        
#         if self.learn_residual:
#             out = phi_u + self.scaling_factor * out
        
#         if self.use_dimensionless:
#             out = self.dimensionalize(out)

#         return out
    
#     def forward(self, u0, sequence_len):
#         res = [u0]
#         u = u0 
#         for _ in range(sequence_len-1):
#             u = self.forward_1step(u)
#             res.append(u)
#         return res, None


# class CorrectionOperator(BaseSolutionMap):
#     """Correction operator."""
    
#     def __init__(self, coarse_h: float, network: DictConfig, **kwargs):
#         super(CorrectionOperator, self).__init__(**kwargs)
        
#         self.save_hyperparameters(logger=False)
        
#         self.coarse_solver = VelocityVerlet(lambda x: self.problem.compute_ddx(x), self.Delta_t, int(self.Delta_t//coarse_h))
#         self.net = hydra.utils.instantiate(network)

#         if self.weight_init is not None:
#             self._init_weights()

#     def forward_1step(self, u):
#         return self.net(self.coarse_solver(u))
    
#     def forward(self, u0, sequence_len):
#         res = []
#         u = u0 
#         for _ in range(sequence_len):
#             u = self.forward_1step(u)
#             res.append(u)
#         return res
    
    
# class CorrectionOperator2(BaseSolutionMap):
#     def __init__(self, coarse_h: float, network: DictConfig, **kwargs):
#         super(CorrectionOperator2, self).__init__(**kwargs)
        
#         self.save_hyperparameters(logger=False)

#         self.coarse_solver = VelocityVerlet(lambda x: self.problem.compute_ddx(x), self.Delta_t, int(self.Delta_t//coarse_h))
#         self.net = hydra.utils.instantiate(network)

#         if self.weight_init is not None:
#             self._init_weights()
        
#     def forward_1step(self, u):
#         C_u = self.coarse_solver(u)
#         return C_u + self.net(C_u)
#         # return C_u + self.net(u)
    
#     def forward(self, u0, sequence_len):
#         res = []
#         u = u0 
#         for _ in range(sequence_len):
#             u = self.forward_1step(u)
#             res.append(u)
#         return res


if __name__ == "__main__":

    from omegaconf import OmegaConf
    from utils.benchmark_utils import time_forward, time_backward, outputs_stats

    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        
        # compose default config and instantiate lightning module 
        cfg = hydra.compose(config_name="train", 
                            overrides=["experiment=nco", 
                                    #    "module/network=piratenet", 
                                    #    "module.network.h2h.n_linears_per_block=1",
                                    #    "module.network.h2h.n_blocks=3"
                                       ])
        # print(print(OmegaConf.to_yaml(cfg.module)))
        model = hydra.utils.instantiate(cfg.module, _recursive_=False)
        
        print(model)
        print("n_trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("device:", model.device)
        print("dtype:", model.dtype)
        
        # benchmark forward time 
        compare = time_forward(model, nsteps_list=[0, 1, 2])
        print(compare)
        
        # benchmark backward time 
        compare = time_backward(model, nsteps_list=[1, 2])
        print(compare)
        
        # benchmark outputs stats 
        stats = outputs_stats(model, nsteps=5)
        print(stats)
