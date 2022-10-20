import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from problems import FPU, LennardJones

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
        self.h = 1e-2
        
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
            x = self.activation(x)
        x = self.layers[-1](x)
        return x
    
    
    

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


NSTEPS_TO_EVAL = 10
        
        
class LitModel(pl.LightningModule):
    def __init__(self, model_name, layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale, loss_fn, optimizer_fn, optimizer_kwargs, lr_scheduler_fn, lr_scheduler_kwargs, lr_scheduler_interval, H_strength, problem):
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
