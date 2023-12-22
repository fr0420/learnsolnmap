import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl

from networks.resnet import MLP, ResMLP, ResMLP2, ResMLP3
from networks.hrn import HamiltonianReversibleNetwork
from networks.equiv import EquivarianceNetwork
from networks.unet import UNet1D
from networks.attention import AttentionMLP
from networks.friction import FrictionBlock

from loss import get_loss_fn
from lr_scheduler import CustomCyclicLR

from problems import FPU, LennardJones
from solvers import VelocityVerlet


OPTIMIZER_DICT = {
    'SGD': optim.SGD,
    'AdamW': optim.AdamW,
    'Adam': optim.Adam,
    'Adadelta': optim.Adadelta,
    'Adagrad': optim.Adagrad, 
    'SparseAdam': optim.SparseAdam,
    'Adamax': optim.Adamax,
#     'NAdam': optim.NAdam,
    'ASGD': optim.ASGD,
    'LBFGS': optim.LBFGS,
    'RMSprop': optim.RMSprop,
    'Rprop': optim.Rprop
}

LR_SCHEDULER_DICT = {
    'LambdaLR': optim.lr_scheduler.LambdaLR,
    'MultiplicativeLR': optim.lr_scheduler.MultiplicativeLR,
    'StepLR': optim.lr_scheduler.StepLR,
    'MultiStepLR': optim.lr_scheduler.MultiStepLR,
#     'ConstantLR': optim.lr_scheduler.ConstantLR,
#     'LinearLR': optim.lr_scheduler.LinearLR,
    'ExponentialLR': optim.lr_scheduler.ExponentialLR,
    'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
#     'ChainedScheduler': optim.lr_scheduler.ChainedScheduler,
#     'SequentialLR': optim.lr_scheduler.SequentialLR,
    'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau,
    'CyclicLR': optim.lr_scheduler.CyclicLR,
    'OneCycleLR': optim.lr_scheduler.OneCycleLR,
    'CosineAnnealingWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts,   
    'CustomCyclicLR': CustomCyclicLR,
}


def get_model(model_name, layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale):
    """Define model"""
    
    model_fn = {
        "MLP": MLP, 
        "ResMLP": ResMLP,
        "ResMLP2": ResMLP2,
        "ResMLP3": ResMLP3,
        "HamiltonianReversibleNetwork": HamiltonianReversibleNetwork,
        "EquivarianceNetwork": EquivarianceNetwork,
        "UNet1D": UNet1D,
        "AttentionMLP": AttentionMLP,
    }[model_name]
    if "ResMLP" in model_name:
        return model_fn(layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale)
    elif model_name == "EquivarianceNetwork":
        return model_fn(7, 2, 200, 3, activation_fn, activation_kwargs, use_bn)
    elif model_name == "UNet1D":
        return model_fn(d=6, fc_hidden_nodes=500)
    elif model_name == "AttentionMLP":
        return model_fn(input_dim=12, output_dim=12, hidden_dim=200, n_hidden_layers=5, activation_fn=activation_fn, activation_kwargs=activation_kwargs)
    else:
        return model_fn(layer_sizes, activation_fn, activation_kwargs, use_bn)


def get_orthonormal_columns(m, n):
    """Get a random m x n matrix with orthonormal columns (m > n)"""
    
    mat = torch.randn(m, m)
    svd = torch.linalg.svd(mat)
    orth = svd[0] @ svd[2]
    return orth[:, :n]
    
    
NSTEPS_TO_EVAL = 10
     
        
class GenericModel(pl.LightningModule):
    def __init__(self, loss_fn="MSELoss", loss_kwargs=None, optimizer_fn="AdamW", optimizer_kwargs=None, lr_scheduler_fn=None, lr_scheduler_kwargs=None, lr_scheduler_interval="epoch", lr_mult=1.):
        super(GenericModel, self).__init__()
        
        self.loss_fn = get_loss_fn(loss_fn, **loss_kwargs) if loss_kwargs is not None else get_loss_fn(loss_fn)
        self.optimizer_fn = OPTIMIZER_DICT[optimizer_fn]
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.lr_scheduler_fn = LR_SCHEDULER_DICT[lr_scheduler_fn] if lr_scheduler_fn is not None else None
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_scheduler_interval = lr_scheduler_interval
        self.lr_mult = lr_mult

        self.training_step_outputs = [] 
        self.validation_step_outputs = []
        self.test_step_outputs = []

      #   self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        pass
        
    def on_train_epoch_end(self):
        self._shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        self._shared_epoch_end(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "test")
    
    def on_test_epoch_end(self):
        self._shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def _shared_eval_step(self, batch, batch_idx, stage=None):
        pass
    
    def _shared_epoch_end(self, outputs, stage=None):
        n_total = sum([out["batch_size"] for out in outputs])
        
        def aggregate_outputs(m):
            return torch.stack([out["metrics"][m] * out["batch_size"] for out in outputs]).sum() / n_total
        
        metrics = {m: aggregate_outputs(m) for m in outputs[0]["metrics"].keys()}
        logs = dict()
        logs[f"{stage}/batch_size"] = outputs[0]["batch_size"]
        for m in metrics.keys():
            logs["/".join([stage, m])] = metrics[m].detach().item()
        if self.trainer.is_global_zero:
            self.log_dict(logs, sync_dist=True, rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler_fn is not None:
            lr_scheduler = {
                "scheduler": self.lr_scheduler_fn(optimizer, **self.lr_scheduler_kwargs),
                "interval": self.lr_scheduler_interval,
                "monitor": "train/loss"
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer
        
    # def configure_optimizers(self):
    #     params = list(self.named_parameters())

    #     def is_last_layer(n): return 'layers.4' in n

    #     # grouped_parameters = [
    #     #     {"params": [p for n, p in params if is_last_layer(n)], 'lr': self.optimizer_kwargs['lr']*self.lr_mult},
    #     #     {"params": [p for n, p in params if not is_last_layer(n)]},
    #     # ]
    #     # optimizer = self.optimizer_fn(grouped_parameters, **self.optimizer_kwargs)
        
    #     optimizer1 = self.optimizer_fn([p for n, p in params if is_last_layer(n)], **self.optimizer_kwargs)
    #     optimizer2 = self.optimizer_fn([p for n, p in params if not is_last_layer(n)], **self.optimizer_kwargs)

    #     if self.lr_scheduler_fn is not None:
    #         lr_scheduler = {
    #             'scheduler': self.lr_scheduler_fn(optimizer2, **self.lr_scheduler_kwargs),
    #             'interval': self.lr_scheduler_interval,
    #             'monitor': 'train/loss'
    #         }
    #         return {'optimizer': optimizer1}, {'optimizer': optimizer2, 'lr_scheduler': lr_scheduler}
    #     else:
    #         return [optimizer1, optimizer2]


class SolutionMapBase(GenericModel):
    def __init__(self, H_strength=0., WS_strength=0., S_strength=0., V_strength=0., 
                 Comm_strength=0., Lagr_strength=0., sequence_len=1, **kwargs):
        super(SolutionMapBase, self).__init__(**kwargs)
            
        self.sequence_len = sequence_len
        self.weights = torch.ones(self.sequence_len)
        
        self.compute_H = None 
        self.H_strength = H_strength 
        self.WS_strength = WS_strength 
        self.S_strength = S_strength 
        self.V_strength = V_strength
        self.Comm_strength = Comm_strength
        self.Lagr_strength = Lagr_strength

    def set_sequence_weights(self, weights):
        self.weights = weights 
        self.sequence_len = len(weights)
    
    def forward(self, u0, sequence_len):
        pass 
        
    def training_step(self, batch, batch_idx):
        losses = torch.zeros(self.sequence_len).to(batch[0])
        self.weights = self.weights.type_as(losses)
        # H_losses = torch.zeros(2).to(batch[0]) 
        S_losses = torch.zeros_like(losses)
        V_losses = torch.zeros_like(losses)
        traj_errors = torch.zeros_like(losses)
        H_errors = torch.zeros_like(losses)

        u0 = batch[0]
        H0 = self.compute_H(u0)
        res, res_hs = self(u0, self.sequence_len)
        
        for t in range(self.sequence_len):
            ut_pred = res[t]
            ut_true = batch[t+1]
            
            # if t < 2: 
            #     H_losses[t] = nn.functional.mse_loss(self.compute_H(ut_pred), H0)
            if t < NSTEPS_TO_EVAL or self.weights[t] != 0:
                losses[t] = self.loss_fn(ut_pred, ut_true)
            S_losses[t] = self.compute_Lagrangian(ut_pred).mean()
            traj_errors[t] = nn.functional.mse_loss(ut_pred, ut_true)
            H_errors[t] = nn.functional.l1_loss(self.compute_H(ut_pred), H0)
             
            hs = res_hs[t] 
#             V_losses[t] = torch.stack([nn.functional.mse_loss(hs[i], hs[i-1]) for i in range(1, len(hs))]).sum() 

        loss = losses @ self.weights
        
#         loss_H = H_losses.sum()
#         loss += self.H_strength * loss_H  # disable H gradient computation for now 
        loss_S = S_losses.sum()
#         loss += self.S_strength * self.Delta_t * loss_S
        
        loss_V = V_losses.sum()
#         loss += self.V_strength * loss_V 
        
#         if isinstance(self.h2h, HamiltonianReversibleNetwork):
#             loss_WS = self.WS_strength * self.h2h.compute_weight_smoothness()
#             loss += loss_WS
            
#         d = torch.stack([torch.det(torch.autograd.functional.jacobian(self, u0, create_graph=True)) for u0 in batch[0]])
#         loss_d = torch.mean((d-1)**2)
#         loss += loss_d

        phi_F_u0 = self(self.fine_solve_dt(u0), 1)[0][0]
        F_phi_u0 = self.fine_solve_dt(self(u0, 1)[0][0])
        loss_Comm = self.loss_fn(F_phi_u0, phi_F_u0)
        # loss += self.Comm_strength * loss_Comm

        # loss_Lagr = 0.
        # u = u0 
        # for _ in range(4):
        #     phi_u = self(u, 1)[0][0]
        #     loss_Lagr += self.compute_Lagrangian(phi_u).mean() * self.fine_solver_dt.T
        #     u = self.fine_solve_dt(u)
        # loss += self.Lagr_strength * loss_Lagr

        self.log('step_loss', loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        metrics = {"loss": loss.detach()}
        for t in range(self.sequence_len):
            metrics[f'loss_step{t+1}'] = losses[t].detach()
            metrics[f'traj_err_step{t+1}'] = traj_errors[t].detach()
            metrics[f'H_err_step{t+1}'] = H_errors[t].detach()

#         metrics['loss_H'] = loss_H.detach()
        metrics['loss_S'] = loss_S.detach()
        metrics['loss_V'] = loss_V.detach()
        metrics['loss_Comm'] = loss_Comm.detach()
        # metrics['loss_Lagr'] = loss_Lagr.detach()
        
        self.training_step_outputs.append({"batch_size": len(batch[0]), "metrics": metrics})
        
        # opt1, opt2 = self.optimizers()

        # opt1.zero_grad()
        # opt2.zero_grad()
        # self.manual_backward(loss)
        # opt1.step()
        # opt2.step()
        
        # sch = self.lr_schedulers()
        # sch.step()

        return {"loss": loss, "batch_size": len(batch[0]), "metrics": metrics}

    def _shared_eval_step(self, batch, batch_idx, stage=None):
        losses = torch.zeros(self.sequence_len).to(batch[0])
        self.weights = self.weights.type_as(losses)
  #       H_losses = torch.zeros(2).to(batch[0]) 
        S_losses = torch.zeros_like(losses)
        traj_errors = torch.zeros_like(losses)
        H_errors = torch.zeros_like(losses)

        u0 = batch[0]
        H0 = self.compute_H(u0)
        res, _ = self(u0, self.sequence_len)
        
        for t in range(self.sequence_len):
            ut_pred = res[t]
            ut_true = batch[t+1]
            
  #           if t < 2: 
  #               H_losses[t] = nn.functional.mse_loss(self.compute_H(ut_pred), H0)
            if t < NSTEPS_TO_EVAL or self.weights[t] != 0:
                losses[t] = self.loss_fn(ut_pred, ut_true)
            S_losses[t] = self.compute_Lagrangian(ut_pred).mean()
            traj_errors[t] = nn.functional.mse_loss(ut_pred, ut_true)
            H_errors[t] = nn.functional.l1_loss(self.compute_H(ut_pred), H0)
        
        loss = losses @ self.weights
        
#         loss_H = H_losses.sum()
#         loss += self.H_strength * loss_H  # disable H gradient computation for now 
        loss_S = S_losses.sum()
#         loss += self.S_strength * self.Delta_t * loss_S
        
        phi_F_u0 = self(self.fine_solve_dt(u0), 1)[0][0]
        F_phi_u0 = self.fine_solve_dt(self(u0, 1)[0][0])
        loss_Comm = self.loss_fn(F_phi_u0, phi_F_u0)
        # loss += self.Comm_strength * loss_Comm
        
        # loss_Lagr = 0.
        # u = u0 
        # for _ in range(4):
        #     phi_u = self(u, 1)[0][0]
        #     loss_Lagr += self.compute_Lagrangian(phi_u).mean() * self.fine_solver_dt.T
        #     u = self.fine_solve_dt(u)
        # loss += self.Lagr_strength * loss_Lagr

        metrics = {"loss": loss.detach()}
        for t in range(self.sequence_len):
            metrics[f'loss_step{t+1}'] = losses[t].detach()
            metrics[f'traj_err_step{t+1}'] = traj_errors[t].detach()
            metrics[f'H_err_step{t+1}'] = H_errors[t].detach()
#         metrics['loss_H'] = loss_H.detach()
        metrics['loss_S'] = loss_S.detach()
        metrics['loss_Comm'] = loss_Comm.detach()
        # metrics['loss_Lagr'] = loss_Lagr.detach()

        if stage == "val": 
            self.validation_step_outputs.append({"batch_size": len(batch[0]), "metrics": metrics})
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch[0]))
        elif stage == "test": 
            self.test_step_outputs.append({"batch_size": len(batch[0]), "metrics": metrics})
        return {"loss": loss, "batch_size": len(batch[0]), "metrics": metrics}

    
class SolutionMap(SolutionMapBase):
    def __init__(self, Delta_t, h2h_model_name, h2h_layer_sizes, i2h_layer_sizes, h2o_layer_sizes, problem, problem_kwargs=None, activation_fn='ELU', activation_kwargs=None, use_bn=False, use_scale=True, init_gamma=0.0, **kwargs):
        super(SolutionMap, self).__init__(**kwargs)
        
        self.save_hyperparameters()
        
        if problem_kwargs is None:
            problem_kwargs = {}
            
        if problem == 'fpu':
            fpu = FPU(**problem_kwargs)
            self.compute_H = lambda u: fpu.compute_H(u[:, :6], u[:, 6:])
            self.compute_Lagrangian = lambda u: fpu.compute_Lagrangian(u[:, :6], u[:, 6:])
            self.input_size = 12
            self.fine_solver_dt = VelocityVerlet(lambda q: fpu.compute_q_ddot(q), Delta_t/128, 8)
        elif problem == 'lennardjones':
            lj = LennardJones(**problem_kwargs)
            self.compute_H = lambda u: lj.compute_H(u[:, :14], u[:, 14:])
            self.compute_Lagrangian = lambda u: lj.compute_Lagrangian(u[:, :14], u[:, 14:])
            self.input_size = 28
            self.fine_solver_dt = VelocityVerlet(lambda x: lj.compute_x_ddot(x), Delta_t/256, 4)
        else:
            self.compute_H = None 
            self.input_size = None

        self.Delta_t = Delta_t
        
        if activation_kwargs is None:
            activation_kwargs = {}
        
        assert h2h_layer_sizes[0] == h2h_layer_sizes[-1]
        self.hidden_size = h2h_layer_sizes[0]
        assert self.hidden_size >= self.input_size
        
        if i2h_layer_sizes is not None:
            assert h2o_layer_sizes is not None
            assert (i2h_layer_sizes[0] == self.input_size) and (i2h_layer_sizes[-1] == self.hidden_size)
            assert (h2o_layer_sizes[0] == self.hidden_size) and (h2o_layer_sizes[-1] == self.input_size)
        else:
            assert h2o_layer_sizes is None 
        
        self.h2h = get_model(h2h_model_name, h2h_layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale)
        
        W = get_orthonormal_columns(self.hidden_size, self.input_size)
        if i2h_layer_sizes is not None:
            self.i2h = get_model('MLP', i2h_layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale)
            self.h2o = get_model('MLP', h2o_layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale) 
        elif self.hidden_size == self.input_size:
            if problem == 'lennardjones':
                M_i2h = torch.diag(torch.cat([0.01*torch.ones(14), torch.ones(14)]))
                M_h2o = torch.diag(torch.cat([100.*torch.ones(14), torch.ones(14)]))
                self.i2h = nn.Linear(self.input_size, self.hidden_size, bias=False)
                self.i2h.weight = nn.Parameter(M_i2h, requires_grad=False)
                self.h2o = nn.Linear(self.hidden_size, self.input_size, bias=False)
                self.h2o.weight = nn.Parameter(M_h2o, requires_grad=False)
            else:
                self.i2h = nn.Identity()
                self.h2o = nn.Identity()
        else:
            self.i2h = nn.Linear(self.input_size, self.hidden_size, bias=False)
            self.i2h.weight = nn.Parameter(W, requires_grad=False)
            self.h2o = nn.Linear(self.hidden_size, self.input_size, bias=False)
            self.h2o.weight = nn.Parameter(W.T, requires_grad=False)
        
        self.friction = FrictionBlock(d=self.input_size//2, init_gamma=init_gamma)
    
    def forward_1step(self, u):
        u = self.i2h(u)
        u = self.h2h(u)
        u = self.h2o(u)
        u = u + self.friction(u)
        return u
    
    def forward(self, u0, sequence_len):
        res = []
        res_hs = []

        hidden = self.i2h(u0)
        for _ in range(sequence_len):
            hidden, hs = self.h2h(hidden, return_hidden=True)
            out = self.h2o(hidden)
            out = out + self.friction(out)
            res.append(out)
            res_hs.append(hs) 

        return res, res_hs

    def fine_solve_dt(self, u):
        d = self.input_size // 2
        return torch.cat(self.fine_solver_dt.solve(u[:, :d], u[:, d:]), dim=1)


class SolutionMapClean(SolutionMapBase):
    def __init__(self, model_config, Delta_t, problem, problem_kwargs=None, init_gamma=0.0, **kwargs):
        super(SolutionMapClean, self).__init__(**kwargs)
        
        self.save_hyperparameters()
        
        if problem_kwargs is None:
            problem_kwargs = {}
            
        if problem == 'fpu':
            fpu = FPU(**problem_kwargs)
            self.compute_H = lambda u: fpu.compute_H(u[:, :6], u[:, 6:])
            self.compute_Lagrangian = lambda u: fpu.compute_Lagrangian(u[:, :6], u[:, 6:])
            self.input_size = 12
            self.fine_solver_dt = VelocityVerlet(lambda q: fpu.compute_q_ddot(q), Delta_t/128, 8)
        elif problem == 'lennardjones':
            lj = LennardJones(**problem_kwargs)
            self.compute_H = lambda u: lj.compute_H(u[:, :14], u[:, 14:])
            self.compute_Lagrangian = lambda u: lj.compute_Lagrangian(u[:, :14], u[:, 14:])
            self.input_size = 28
            self.fine_solver_dt = VelocityVerlet(lambda x: lj.compute_x_ddot(x), Delta_t/256, 4)
        else:
            self.compute_H = None 
            self.input_size = None

        self.Delta_t = Delta_t
        
        networks_dict = {
            "Identity": nn.Identity,
            "MLP": MLP, 
            "ResMLP": ResMLP,
            "ResMLP2": ResMLP2,
            "ResMLP3": ResMLP3,
            "HamiltonianReversibleNetwork": HamiltonianReversibleNetwork,
            "EquivarianceNetwork": EquivarianceNetwork,
            "UNet1D": UNet1D,
            "AttentionMLP": AttentionMLP,
            }
        
        self.i2h = networks_dict[model_config.i2h_network](**model_config.i2h_config)
        self.h2h = networks_dict[model_config.h2h_network](**model_config.h2h_config)
        self.h2o = networks_dict[model_config.h2o_network](**model_config.h2o_config)
        self.friction = FrictionBlock(d=self.input_size//2, init_gamma=init_gamma)
    
    def forward_1step(self, u):
        u = self.i2h(u)
        u = self.h2h(u)
        u = self.h2o(u)
        u = u + self.friction(u)
        return u
    
    def forward(self, u0, sequence_len):
        res = []
        res_hs = []

        hidden = self.i2h(u0)
        for _ in range(sequence_len):
            hidden, hs = self.h2h(hidden, return_hidden=True)
            out = self.h2o(hidden)
            out = out + self.friction(out)
            res.append(out)
            res_hs.append(hs) 

        return res, res_hs

    def fine_solve_dt(self, u):
        d = self.input_size // 2
        return torch.cat(self.fine_solver_dt.solve(u[:, :d], u[:, d:]), dim=1)
        

class CorrectionOperator(SolutionMapBase):
    def __init__(self, Delta_t, coarse_h, model_name, layer_sizes, problem, problem_kwargs=None, activation_fn='ELU', activation_kwargs=None, use_bn=False, use_scale=True, **kwargs):
        super(CorrectionOperator, self).__init__(**kwargs)
        
        self.save_hyperparameters()
        
        self.problem = problem
        if problem_kwargs is None:
            problem_kwargs = {}
            
        if self.problem == 'fpu':
            fpu = FPU(**problem_kwargs)
            self.compute_H = lambda u: fpu.compute_H(u[:, :6], u[:, 6:])
            self.compute_Lagrangian = lambda u: fpu.compute_Lagrangian(u[:, :6], u[:, 6:])
            self.input_size = 12
            self.coarse_solver = VelocityVerlet(lambda q: fpu.compute_q_ddot(q), Delta_t, int(Delta_t//coarse_h))
        elif self.problem == 'lennardjones':
            lj = LennardJones(**problem_kwargs)
            self.compute_H = lambda u: lj.compute_H(u[:, :14], u[:, 14:])
            self.compute_Lagrangian = lambda u: lj.compute_Lagrangian(u[:, :14], u[:, 14:])
            self.input_size = 28
            self.coarse_solver = VelocityVerlet(lambda x: lj.compute_x_ddot(x), Delta_t, int(Delta_t//coarse_h))
        else:
            self.compute_H = None 
            self.input_size = None
        
        if activation_kwargs is None:
            activation_kwargs = {} 
        self.model = get_model(model_name, layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale)
        
        self.Delta_t = Delta_t
        
    def coarse_solve(self, u):
        d = self.input_size // 2
        return torch.cat(self.coarse_solver.solve(u[:, :d], u[:, d:]), dim=1)
        
    def forward_1step(self, u):
        return self.model(self.coarse_solve(u))
    
    def forward(self, u0, sequence_len):
        res = []
        u = u0 
        for _ in range(sequence_len):
            u = self.forward_1step(u)
            res.append(u)
        return res
    
    
class CorrectionOperator2(SolutionMapBase):
    def __init__(self, Delta_t, coarse_h, model_name, layer_sizes, problem, problem_kwargs=None, activation_fn='ELU', activation_kwargs=None, use_bn=False, use_scale=True, **kwargs):
        super(CorrectionOperator2, self).__init__(**kwargs)
        
        self.save_hyperparameters()
        
        self.problem = problem
        if problem_kwargs is None:
            problem_kwargs = {}
            
        if self.problem == 'fpu':
            fpu = FPU(**problem_kwargs)
            self.compute_H = lambda u: fpu.compute_H(u[:, :6], u[:, 6:])
            self.compute_Lagrangian = lambda u: fpu.compute_Lagrangian(u[:, :6], u[:, 6:])
            self.input_size = 12
            self.coarse_solver = VelocityVerlet(lambda q: fpu.compute_q_ddot(q), Delta_t, int(Delta_t//coarse_h))
        elif self.problem == 'lennardjones':
            lj = LennardJones(**problem_kwargs)
            self.compute_H = lambda u: lj.compute_H(u[:, :14], u[:, 14:])
            self.compute_Lagrangian = lambda u: lj.compute_Lagrangian(u[:, :14], u[:, 14:])
            self.input_size = 28
            self.coarse_solver = VelocityVerlet(lambda x: lj.compute_x_ddot(x), Delta_t, int(Delta_t//coarse_h))
        else:
            self.compute_H = None 
            self.input_size = None
        
        if activation_kwargs is None:
            activation_kwargs = {} 
        self.model = get_model(model_name, layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale)
        
        self.Delta_t = Delta_t
        
    def coarse_solve(self, u):
        d = self.input_size // 2
        return torch.cat(self.coarse_solver.solve(u[:, :d], u[:, d:]), dim=1)
        
    def forward_1step(self, u):
        u_c = self.coarse_solve(u)
        return u_c + self.model(u_c)
    
    def forward(self, u0, sequence_len):
        res = []
        u = u0 
        for _ in range(sequence_len):
            u = self.forward_1step(u)
            res.append(u)
        return res
