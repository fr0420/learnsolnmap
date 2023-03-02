import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from problems import FPU, LennardJones
from solvers import VelocityVerlet


ACTIVATION_DICT = {
    'ELU': nn.ELU,
    'Hardshrink': nn.Hardshrink,
    'Hardsigmoid': nn.Hardsigmoid,
    'Hardtanh': nn.Hardtanh,
    'Hardswish': nn.Hardswish,
    'LeakyReLU': nn.LeakyReLU,
    'LogSigmoid': nn.LogSigmoid,
    'MultiheadAttention': nn.MultiheadAttention,
    'PReLU': nn.PReLU,
    'ReLU': nn.ReLU,
    'ReLU6': nn.ReLU6,
    'RReLU': nn.RReLU,
    'SELU': nn.SELU,
    'CELU': nn.CELU,
    'GELU': nn.GELU,
    'Sigmoid': nn.Sigmoid,
    'SiLU': nn.SiLU,
    'Softplus': nn.Softplus,
    'Softshrink': nn.Softshrink,
    'Softsign': nn.Softsign,
    'Tanh': nn.Tanh,
    'Tanhshrink': nn.Tanhshrink,
    'Threshold': nn.Threshold
}

LOSS_FN_DICT = {
    'MSELoss': nn.MSELoss()
}

OPTIMIZER_DICT = {
    'SGD': optim.SGD,
    'AdamW': optim.AdamW,
    'Adam': optim.Adam,
    'Adadelta': optim.Adadelta,
    'Adagrad': optim.Adagrad, 
    'SparseAdam': optim.SparseAdam,
    'Adamax': optim.Adamax,
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
    'CosineAnnealingWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts    
}


class MLP(nn.Module):
    """Multi-layer perceptron"""
    
    def __init__(self, layer_sizes, activation_fn, activation_kwargs, use_bn):
        super(MLP, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList(
            [nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-1)]
        )
        self.activation = get_activation(activation_fn, **activation_kwargs)
        self.use_bn = use_bn
        if self.use_bn:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-2)]
            )
        
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


class ResMLP(nn.Module):
    """Multi-layer perceptron with residual connections between layers of equal width"""
    
    def __init__(self, layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale):
        super(ResMLP, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList(
            [nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-1)]
        )
        self.activation = get_activation(activation_fn, **activation_kwargs) 
        self.use_bn = use_bn
        if self.use_bn:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-2)]
            )
        self.scale = 1. / len(self.layers) if use_scale else 1.
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            identity = x 
            x = layer(x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.activation(x)
            if layer.in_features == layer.out_features:
                x = identity + self.scale * x
        
        output_layer = self.layers[-1]
        identity = x
        x = output_layer(x)
        if output_layer.in_features == output_layer.out_features:
            x = self.activation(x)
            x = identity + self.scale * x
                
        return x
    

class HamiltonianReversibleBlock(nn.Module):
    """Hamiltonian Reversible Block"""
    
    def __init__(self, degree_of_freedom, activation_fn, activation_kwargs):
        super(HamiltonianReversibleBlock, self).__init__()
        
        self.dof = degree_of_freedom
        self.activation = get_activation(activation_fn, **activation_kwargs)
        
        self.layer1 = nn.Linear(self.dof, self.dof)
        self.layer2 = nn.Linear(self.dof, self.dof)
        self.h = 1e-3
        
    def forward(self, x):
        
        p, q = torch.split(x, [self.dof, self.dof], dim=-1)
        
        pnew = p + self.h * torch.matmul(self.activation(self.layer1(q)), self.layer1.weight)
        qnew = q - self.h * torch.matmul(self.activation(self.layer2(pnew)), self.layer2.weight)
        
        xnew = torch.cat((pnew, qnew), dim=-1)
        
        return xnew


class HamiltonianReversibleNetwork(nn.Module):
    """Hamiltonian Reversible Network"""
    
    def __init__(self, layer_sizes, activation_fn, activation_kwargs, use_bn):
        super(HamiltonianReversibleNetwork, self).__init__()
        
        self.layer_sizes = layer_sizes
        layers = []
        for i in range(len(self.layer_sizes)-1):
            in_features = self.layer_sizes[i]
            out_features = self.layer_sizes[i+1]
            if in_features == out_features and in_features % 2 == 0:
                layers.append(HamiltonianReversibleBlock(in_features//2, activation_fn, activation_kwargs))
            else:
                layers.append(nn.Linear(in_features, out_features))
        self.layers = nn.ModuleList(layers)
        self.activation = get_activation(activation_fn, **activation_kwargs) 
        self.use_bn = use_bn
        if self.use_bn:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-2)]
            )
        
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            if not isinstance(self.layers[i], HamiltonianReversibleBlock):
                x = self.activation(x)
        x = self.layers[-1](x)
        return x
    
    def compute_weight_smoothness(self):
        s = 0.
        for layer_cur, layer_next in zip(self.layers[:-1], self.layers[1:]):
            if isinstance(layer_cur, HamiltonianReversibleBlock):
                if isinstance(layer_next, HamiltonianReversibleBlock):
                    if layer_cur.dof == layer_next.dof:
                        s += torch.linalg.norm(layer_next.layer1.weight - layer_cur.layer1.weight) 
                        s += torch.linalg.norm(layer_next.layer2.weight - layer_cur.layer2.weight) 
        return s / 1e-3
    

def get_activation(activation_fn, **kwargs):
    """Define activation function"""
    
    return ACTIVATION_DICT[activation_fn](**kwargs)


def get_loss_fn(loss_fn):
    """Define loss function"""
    
    return LOSS_FN_DICT[loss_fn]


def get_model(model_name, layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale):
    """Define model"""
    
    model_fn = {
        "MLP": MLP, 
        "ResMLP": ResMLP, 
        "HamiltonianReversibleNetwork": HamiltonianReversibleNetwork
    }[model_name]
    if model_name == "ResMLP":
        return model_fn(layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale)
    else:
        return model_fn(layer_sizes, activation_fn, activation_kwargs, use_bn)


def get_orthonormal_columns(m, n):
    """Get a random m x n matrix with orthonormal columns (m > n)"""
    
    mat = torch.randn(m, m)
    svd = torch.linalg.svd(mat)
    orth = svd[0] @ svd[2]
    return orth[:, :n]
    
    
NSTEPS_TO_EVAL = 10
        
        
class LitModel(pl.LightningModule):
    def __init__(self, model_name, layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale, loss_fn, optimizer_fn, optimizer_kwargs, lr_scheduler_fn, lr_scheduler_kwargs, lr_scheduler_interval, H_strength, WS_strength, problem):
        super(LitModel, self).__init__()
        
        self.save_hyperparameters()
        
        self.model = get_model(model_name, layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale)
        self.loss_fn = get_loss_fn(loss_fn)
        self.optimizer_fn = OPTIMIZER_DICT[optimizer_fn]
        self.optimizer_kwargs = optimizer_kwargs 
        self.lr_scheduler_fn = LR_SCHEDULER_DICT[lr_scheduler_fn] if lr_scheduler_fn is not None else None
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_scheduler_interval = lr_scheduler_interval
        
        self.H_strength = H_strength 
        if problem == 'fpu':
            fpu = FPU()
            self.compute_H = lambda u: fpu.compute_H(u[:, :6], u[:, 6:])
        elif problem == 'lennardjones':
            lj = LennardJones()
            self.compute_H = lambda u: lj.compute_H(u[:, :14]*100., u[:, 14:])
        else:
            self.compute_H = None 
        
        self.WS_strength = WS_strength 
        self.sequence_len = 1
        self.weights = None
    
    def set_sequence_weights(self, weights):
        self.weights = weights 
        self.sequence_len = len(weights)
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        losses = torch.zeros(self.sequence_len, dtype=torch.double, device=self.device)
        self.weights = self.weights.type_as(losses)
        H_losses = torch.zeros(2, dtype=torch.double, device=self.device) 
        
        u = batch[0]
        H = self.compute_H(u)
            
        for t in range(1, self.sequence_len+1):
            u = self.forward(u)
            if t <= 2: 
                H_losses[t-1] = self.loss_fn(self.compute_H(u), H)
            if t <= NSTEPS_TO_EVAL or self.weights[t-1] != 0:
                losses[t-1] = self.loss_fn(u, batch[t])
        
        loss = losses @ self.weights
        
        loss_H = H_losses.sum()
#         loss += self.H_strength * loss_H  # disable H gradient computation for now 
        
        if isinstance(self.model, HamiltonianReversibleNetwork):
            loss_WS = self.WS_strength * self.model.compute_weight_smoothness()
            loss += loss_WS
            
#         d = torch.stack([torch.det(torch.autograd.functional.jacobian(self, u0, create_graph=True)) for u0 in batch[0]])
#         loss_d = torch.mean((d-1)**2)
#         loss += loss_d
        
        self.log('step_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        metrics = {'loss': loss.detach()}
        for t, l in enumerate(losses[:NSTEPS_TO_EVAL]):
            metrics[f'loss_step{t+1}'] = l.detach()
        metrics['loss_H'] = loss_H.detach()
        return {'loss': loss, 'batch_size': len(batch[0]), 'metrics': metrics}
  
    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'train')
        
    def validation_step(self, batch, batch_idx):
        loss, losses = self._shared_eval_step(batch, batch_idx)   
        metrics = {'loss': loss}
        for t, l in enumerate(losses[:NSTEPS_TO_EVAL]):
            metrics[f'loss_step{t+1}'] = l 
        return {'batch_size': len(batch[0]), 'metrics': metrics}
    
    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'val')
    
    def test_step(self, batch, batch_idx):
        loss, losses = self._shared_eval_step(batch, batch_idx)    
        metrics = {'loss': loss}
        for t, l in enumerate(losses[:NSTEPS_TO_EVAL]):
            metrics[f'loss_step{t+1}'] = l
        return {'batch_size': len(batch[0]), 'metrics': metrics}
    
    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'test')
        
    def _shared_eval_step(self, batch, batch_idx):
        losses = torch.zeros(self.sequence_len, dtype=torch.double, device=self.device)
        self.weights = self.weights.type_as(losses)
        
        u = batch[0]
        
        for t in range(1, self.sequence_len+1):
            u = self.forward(u)
            if t <= NSTEPS_TO_EVAL or self.weights[t-1] != 0:
                losses[t-1] = self.loss_fn(u, batch[t])
        
        loss = losses @ self.weights
        
        return loss, losses
    
    def _shared_epoch_end(self, outputs, stage=None):
        n_total = sum([out['batch_size'] for out in outputs])
        
        def aggregate_outputs(m):
            return torch.stack([out['metrics'][m] * out['batch_size'] for out in outputs]).sum() / n_total
        
        metrics = {m: aggregate_outputs(m) for m in outputs[0]['metrics'].keys()}
        logs = dict()
        for m in metrics.keys():
            logs['/'.join([stage, m])] = metrics[m].detach().item()
        if self.trainer.is_global_zero:
            self.log_dict(logs, rank_zero_only=True)
        
    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler_fn is not None:
            lr_scheduler = {
                'scheduler': self.lr_scheduler_fn(optimizer, **self.lr_scheduler_kwargs),
                'interval': self.lr_scheduler_interval,
                'monitor': 'train/loss'
            }
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            return optimizer

        
     
        
class GenericModel(pl.LightningModule):
    def __init__(self, loss_fn='MSELoss', optimizer_fn='AdamW', optimizer_kwargs=None, lr_scheduler_fn=None, lr_scheduler_kwargs=None, lr_scheduler_interval='epoch'):
        super(GenericModel, self).__init__()
        
        self.loss_fn = get_loss_fn(loss_fn)
        self.optimizer_fn = OPTIMIZER_DICT[optimizer_fn]
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.lr_scheduler_fn = LR_SCHEDULER_DICT[lr_scheduler_fn] if lr_scheduler_fn is not None else None
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_scheduler_interval = lr_scheduler_interval

    def training_step(self, batch, batch_idx):
        pass
        
    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'train')
        
    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'test')
        
    def _shared_eval_step(self, batch, batch_idx):
        pass
    
    def _shared_epoch_end(self, outputs, stage=None):
        n_total = sum([out['batch_size'] for out in outputs])
        
        def aggregate_outputs(m):
            return torch.stack([out['metrics'][m] * out['batch_size'] for out in outputs]).sum() / n_total
        
        metrics = {m: aggregate_outputs(m) for m in outputs[0]['metrics'].keys()}
        logs = dict()
        for m in metrics.keys():
            logs['/'.join([stage, m])] = metrics[m].detach().item()
        if self.trainer.is_global_zero:
            self.log_dict(logs, rank_zero_only=True)
        
    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler_fn is not None:
            lr_scheduler = {
                'scheduler': self.lr_scheduler_fn(optimizer, **self.lr_scheduler_kwargs),
                'interval': self.lr_scheduler_interval,
                'monitor': 'train/loss'
            }
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            return optimizer


class SolutionMapBase(GenericModel):
    def __init__(self, H_strength=0., WS_strength=0., S_strength=0., **kwargs):
        super(SolutionMapBase, self).__init__(**kwargs)
            
        self.sequence_len = 1
        self.weights = None
        
        self.compute_H = None 
        self.H_strength = H_strength 
        self.WS_strength = WS_strength 
        self.S_strength = S_strength 
        
    def set_sequence_weights(self, weights):
        self.weights = weights 
        self.sequence_len = len(weights)
    
    def get_sequence_predictions(self, u0, sequence_len):
        pass 
        
    def training_step(self, batch, batch_idx):
        losses = torch.zeros(self.sequence_len, dtype=torch.double, device=self.device)
        self.weights = self.weights.type_as(losses)
        H_losses = torch.zeros(2, dtype=torch.double, device=self.device) 
        S_losses = torch.zeros(self.sequence_len, dtype=torch.double, device=self.device)
        
        u0 = batch[0]
        H0 = self.compute_H(u0)
        res = self.get_sequence_predictions(u0, self.sequence_len)
        
        for t in range(1, self.sequence_len+1):
            ut_pred = res[t-1]
            ut_true = batch[t]
            
            if t <= 2: 
                H_losses[t-1] = self.loss_fn(self.compute_H(ut_pred), H0)
            if t <= NSTEPS_TO_EVAL or self.weights[t-1] != 0:
                losses[t-1] = self.loss_fn(ut_pred, ut_true)
            S_losses[t-1] = self.compute_Lagrangian(ut_pred).mean()
            
        loss = losses @ self.weights
        
        loss_H = H_losses.sum()
#         loss += self.H_strength * loss_H  # disable H gradient computation for now 
        loss_S = S_losses.sum()
        loss += self.S_strength * self.Delta_t * loss_S
        
#         if isinstance(self.h2h, HamiltonianReversibleNetwork):
#             loss_WS = self.WS_strength * self.h2h.compute_weight_smoothness()
#             loss += loss_WS
            
#         d = torch.stack([torch.det(torch.autograd.functional.jacobian(self, u0, create_graph=True)) for u0 in batch[0]])
#         loss_d = torch.mean((d-1)**2)
#         loss += loss_d
        
        self.log('step_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        metrics = {'loss': loss.detach()}
        for t, l in enumerate(losses[:NSTEPS_TO_EVAL]):
            metrics[f'loss_step{t+1}'] = l.detach()
        metrics['loss_H'] = loss_H.detach()
        metrics['loss_S'] = loss_S.detach()
        return {'loss': loss, 'batch_size': len(batch[0]), 'metrics': metrics}
  
    def _shared_eval_step(self, batch, batch_idx):
        losses = torch.zeros(self.sequence_len, dtype=torch.double, device=self.device)
        self.weights = self.weights.type_as(losses)
        H_losses = torch.zeros(2, dtype=torch.double, device=self.device) 
        S_losses = torch.zeros(self.sequence_len, dtype=torch.double, device=self.device)
        
        u0 = batch[0]
        H0 = self.compute_H(u0)
        res = self.get_sequence_predictions(u0, self.sequence_len)
        
        for t in range(1, self.sequence_len+1):
            ut_pred = res[t-1]
            ut_true = batch[t]
            
            if t <= 2: 
                H_losses[t-1] = self.loss_fn(self.compute_H(ut_pred), H0)
            if t <= NSTEPS_TO_EVAL or self.weights[t-1] != 0:
                losses[t-1] = self.loss_fn(ut_pred, ut_true)
            S_losses[t-1] = self.compute_Lagrangian(ut_pred).mean()

        loss = losses @ self.weights
        
        loss_H = H_losses.sum()
#         loss += self.H_strength * loss_H  # disable H gradient computation for now 
        loss_S = S_losses.sum()
        loss += self.S_strength * self.Delta_t * loss_S
        
        metrics = {'loss': loss.detach()}
        for t, l in enumerate(losses[:NSTEPS_TO_EVAL]):
            metrics[f'loss_step{t+1}'] = l.detach()
        metrics['loss_H'] = loss_H.detach()
        metrics['loss_S'] = loss_S.detach()
        return {'loss': loss, 'batch_size': len(batch[0]), 'metrics': metrics}

    
class SolutionMap(SolutionMapBase):
    def __init__(self, Delta_t, h2h_model_name, h2h_layer_sizes, i2h_layer_sizes, h2o_layer_sizes, problem, problem_kwargs=None, activation_fn='ELU', activation_kwargs=None, use_bn=False, use_scale=True, **kwargs):
        super(SolutionMap, self).__init__(**kwargs)
        
        self.save_hyperparameters()
        
        if problem_kwargs is None:
            problem_kwargs = {}
            
        if problem == 'fpu':
            fpu = FPU(**problem_kwargs)
            self.compute_H = lambda u: fpu.compute_H(u[:, :6], u[:, 6:])
            self.compute_Lagrangian = lambda u: fpu.compute_Lagrangian(u[:, :6], u[:, 6:])
            self.input_size = 12
        elif problem == 'lennardjones':
            lj = LennardJones(**problem_kwargs)
            self.compute_H = lambda u: lj.compute_H(u[:, :14]*100., u[:, 14:])
            self.compute_Lagrangian = lambda u: lj.compute_Lagrangian(u[:, :14]*100., u[:, 14:])
            self.input_size = 28
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
            self.i2h = nn.Identity()
            self.h2o = nn.Identity()
        else:
            self.i2h = nn.Linear(self.input_size, self.hidden_size, bias=False)
            self.i2h.weight = nn.Parameter(W, requires_grad=False)
            self.h2o = nn.Linear(self.hidden_size, self.input_size, bias=False)
            self.h2o.weight = nn.Parameter(W.T, requires_grad=False)
        
    def forward(self, u):
        u = self.i2h(u)
        u = self.h2h(u)
        u = self.h2o(u)
        return u
    
    def get_sequence_predictions(self, u0, sequence_len):
        res = []
        hidden = self.i2h(u0)
        for _ in range(sequence_len):
            hidden = self.h2h(hidden)
            res.append(self.h2o(hidden))
        return res


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
            self.compute_H = lambda u: lj.compute_H(u[:, :14]*100., u[:, 14:])
            self.compute_Lagrangian = lambda u: lj.compute_Lagrangian(u[:, :14]*100., u[:, 14:])
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
        if self.problem == 'fpu':
            return torch.cat(self.coarse_solver.solve(u[:, :d], u[:, d:]), dim=1)
        elif self.problem == 'lennardjones':
            v, x = self.coarse_solver.solve(u[:, :d]*100., u[:, d:])
            return torch.cat((v/100., x), dim=1)
        else: 
            pass 
        
    def forward(self, u):
        return self.model(self.coarse_solve(u))
    
    def get_sequence_predictions(self, u0, sequence_len):
        res = []
        u = u0 
        for _ in range(sequence_len):
            u = self.forward(u)
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
            self.compute_H = lambda u: lj.compute_H(u[:, :14]*100., u[:, 14:])
            self.compute_Lagrangian = lambda u: lj.compute_Lagrangian(u[:, :14]*100., u[:, 14:])
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
        if self.problem == 'fpu':
            return torch.cat(self.coarse_solver.solve(u[:, :d], u[:, d:]), dim=1)
        elif self.problem == 'lennardjones':
            v, x = self.coarse_solver.solve(u[:, :d]*100., u[:, d:])
            return torch.cat((v/100., x), dim=1)
        else: 
            pass 
        
    def forward(self, u):
        u_c = self.coarse_solve(u)
        return u_c + self.model(u_c)
    
    def get_sequence_predictions(self, u0, sequence_len):
        res = []
        u = u0 
        for _ in range(sequence_len):
            u = self.forward(u)
            res.append(u)
        return res
