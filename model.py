import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl


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





class MLP(pl.LightningModule):
    def __init__(self, layer_sizes, activation_fn, activation_kwargs, use_bn, loss_fn, with_one_step_loss, with_two_step_loss, optimizer_fn, optimizer_kwargs, 
                 lr_scheduler_fn, lr_scheduler_kwargs, lr_scheduler_interval):
        super().__init__()
        
        self.save_hyperparameters()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList(
            [nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-1)]
        )
        self.activation = ACTIVATION_DICT[activation_fn](**activation_kwargs) 
        self.use_bn = use_bn
        if self.use_bn:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-2)]
            )
        self.loss_fn = LOSS_FN_DICT[loss_fn]
        self.with_one_step_loss = with_one_step_loss
        self.with_two_step_loss = with_two_step_loss
        self.optimizer_fn = OPTIMIZER_DICT[optimizer_fn]
        self.optimizer_kwargs = optimizer_kwargs 
        self.lr_scheduler_fn = LR_SCHEDULER_DICT[lr_scheduler_fn] if lr_scheduler_fn is not None else None
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_scheduler_interval = lr_scheduler_interval
        
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x
  
    def training_step(self, batch, batch_idx):
        u0, u1, u2 = batch
        f_u0 = self.forward(u0)
        f_f_u0 = self.forward(f_u0)
        loss_1step = self.loss_fn(f_u0, u1)
        loss_2step = self.loss_fn(f_f_u0, u2)
        loss = 0.
        if self.with_one_step_loss: 
            loss += loss_1step
        if self.with_two_step_loss:
            loss += loss_2step
        self.log('step_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return {'loss': loss, 'batch_size': len(batch[0]),
                'metrics': {'loss': loss, 'loss_1step': loss_1step, 'loss_2step': loss_2step}}
  
    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'train')
        
    def validation_step(self, batch, batch_idx):
        loss, loss_1step, loss_2step = self._shared_eval_step(batch, batch_idx)    
        return {'batch_size': len(batch[0]), 
                'metrics': {'loss': loss, 'loss_1step': loss_1step, 'loss_2step': loss_2step}}
    
    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'val')
    
    def test_step(self, batch, batch_idx):
        loss, loss_1step, loss_2step = self._shared_eval_step(batch, batch_idx)    
        return {'batch_size': len(batch[0]), 
                'metrics': {'loss': loss, 'loss_1step': loss_1step, 'loss_2step': loss_2step}}
    
    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'test')
        
    def _shared_eval_step(self, batch, batch_idx):
        u0, u1, u2 = batch
        f_u0 = self.forward(u0)
        f_f_u0 = self.forward(f_u0)
        loss_1step = self.loss_fn(f_u0, u1)
        loss_2step = self.loss_fn(f_f_u0, u2)
        loss = 0.
        if self.with_one_step_loss: 
            loss += loss_1step
        if self.with_two_step_loss:
            loss += loss_2step
        return loss, loss_1step, loss_2step
    
    def _shared_epoch_end(self, outputs, stage=None):
        n_total = sum([out['batch_size'] for out in outputs])
        
        def aggregate_outputs(m):
            return torch.stack([out['metrics'][m] * out['batch_size'] for out in outputs]).sum() / n_total
        
        metrics = {m: aggregate_outputs(m) for m in outputs[0]['metrics'].keys()}
        logs = dict()
        for m in metrics.keys():
            logs['/'.join([stage, m])] = metrics[m].detach().item()
        self.log_dict(logs)
        
    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler_fn is not None:
            lr_scheduler = {
                'scheduler': self.lr_scheduler_fn(optimizer, **self.lr_scheduler_kwargs),
                'interval': self.lr_scheduler_interval
            }
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            return optimizer

    def get_progress_bar_dict(self):
        # don't show the version number 
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items 



def get_activation(activation_fn, kwargs):
    '''Define activation function'''
    
    return ACTIVATION_DICT[activation_fn](**kwargs)


def get_loss_fn(loss_fn):
    
    return LOSS_FN_DICT[loss_fn]



Omega = 300
C0 = 0.25 * Omega**2

def compute_H(u):
    p, q = u[:, :6], u[:, 6:]
    K = 0.5 * torch.sum(p**2, axis=1)
    dq_stiff = q[:, 1::2] - q[:, ::2]
    dq_soft = torch.stack((q[:, 0], q[:, 2]-q[:, 1], q[:, 4]-q[:, 3], -q[:, 5]), dim=1)
    U = C0 * torch.sum(dq_stiff ** 2, axis=1) + torch.sum(dq_soft ** 4, axis=1)
    return K + U


class ResMLP(pl.LightningModule):
    def __init__(self, layer_sizes, activation_fn, activation_kwargs, use_bn, loss_fn, optimizer_fn, optimizer_kwargs, lr_scheduler_fn,
                 lr_scheduler_kwargs, lr_scheduler_interval, H_strength):
        super().__init__()
        
        self.save_hyperparameters()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList(
            [nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-1)]
        )
        self.activation = get_activation(activation_fn, activation_kwargs) 
        self.use_bn = use_bn
        if self.use_bn:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-2)]
            )
        self.loss_fn = get_loss_fn(loss_fn)
        self.optimizer_fn = OPTIMIZER_DICT[optimizer_fn]
        self.optimizer_kwargs = optimizer_kwargs 
        self.lr_scheduler_fn = LR_SCHEDULER_DICT[lr_scheduler_fn] if lr_scheduler_fn is not None else None
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_scheduler_interval = lr_scheduler_interval
        
        self.H_strength = H_strength 
        
        self.sequence_len = 1
        self.weights = None
    
    def set_sequence_weights(self, weights):
        self.weights = weights 
        self.sequence_len = len(weights)
        
    def forward(self, x):
        x = self.layers[0](x)
        if self.use_bn:
            x = self.bn_layers[0](x)
        x = self.activation(x)
        for i in range(1, len(self.layers)-1):
            identity = x 
            x = self.layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.activation(x)
            x += identity
        x = self.layers[-1](x)
        return x
        
    def training_step(self, batch, batch_idx):
        losses = torch.zeros(self.sequence_len, dtype=torch.double, device=self.device)

        u = batch[0]
        H = compute_H(u)
            
        for t in range(1, self.sequence_len+1):
            u = self.forward(u)
            if t <= 5 or self.weights[t-1] != 0:
                losses[t-1] = self.loss_fn(u, batch[t])
        
        loss = losses @ self.weights
        
        loss_H = self.loss_fn(compute_H(u), H)
        loss += self.H_strength * loss_H
        
#         d = torch.stack([torch.det(torch.autograd.functional.jacobian(self, u0, create_graph=True)) for u0 in batch[0]])
#         loss_d = torch.mean((d-1)**2)
#         loss += loss_d
        
        self.log('step_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        metrics = {'loss': loss.detach()}
        for t, l in enumerate(losses[:5]):
            metrics[f'loss_{t+1}step'] = l.detach()
        metrics['loss_H'] = loss_H.detach()
        return {'loss': loss, 'batch_size': len(batch[0]), 'metrics': metrics}
  
    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'train')
        
    def validation_step(self, batch, batch_idx):
        loss, losses = self._shared_eval_step(batch, batch_idx)   
        metrics = {'loss': loss}
        for t, l in enumerate(losses[:5]):
            metrics[f'loss_{t+1}step'] = l 
        return {'batch_size': len(batch[0]), 'metrics': metrics}
    
    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'val')
    
    def test_step(self, batch, batch_idx):
        loss, losses = self._shared_eval_step(batch, batch_idx)    
        metrics = {'loss': loss}
        for t, l in enumerate(losses[:5]):
            metrics[f'loss_{t+1}step'] = l
        return {'batch_size': len(batch[0]), 'metrics': metrics}
    
    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, 'test')
        
    def _shared_eval_step(self, batch, batch_idx):
        losses = torch.zeros(self.sequence_len, dtype=torch.double, device=self.device)

        u = batch[0]
        
        for t in range(1, self.sequence_len+1):
            u = self.forward(u)
            if t <= 5 or self.weights[t-1] != 0:
                losses[t-1] = self.loss_fn(u, batch[t])
        
        loss = losses @ self.weights
        
#         losses = []
#         u = batch[0]
#         for t in range(1, 6):
#             u = self.forward(u)
#             losses.append(self.loss_fn(u, batch[t]))
        return loss, losses
    
    def _shared_epoch_end(self, outputs, stage=None):
        n_total = sum([out['batch_size'] for out in outputs])
        
        def aggregate_outputs(m):
            return torch.stack([out['metrics'][m] * out['batch_size'] for out in outputs]).sum() / n_total
        
        metrics = {m: aggregate_outputs(m) for m in outputs[0]['metrics'].keys()}
        logs = dict()
        for m in metrics.keys():
            logs['/'.join([stage, m])] = metrics[m].detach().item()
        self.log_dict(logs)
        
    def configure_optimizers(self):
        optimizer = self.optimizer_fn(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler_fn is not None:
            lr_scheduler = {
                'scheduler': self.lr_scheduler_fn(optimizer, **self.lr_scheduler_kwargs),
                'interval': self.lr_scheduler_interval
            }
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            return optimizer

    def get_progress_bar_dict(self):
        # don't show the version number 
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items 

    
    

# import math 



# def get_conv_layer(in_channels, out_channels, kernel_size):
#     '''
#     Define a stride 1 convolution layer which keeps the input image size unchanged
#     (kernel_size is required to be an odd number because Conv2d allows only symmetric padding) 
#     '''
    
#     return nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2) 


# def get_downscale_layer(in_channels, out_channels, kernel_size):
#     '''
#     Define a stride 2 convolution layer which halves the input image size
#     '''
#     return nn.Conv1d(out_channels, out_channels, kernel_size, stride=2, padding=math.ceil(kernel_size/2-1))


# def get_batchnorm_layer(num_features):
#     return nn.BatchNorm1d(num_features)
    


# class Concatenate(nn.Module):
#     '''Concatenate two inputs along the channel dimension'''
#     def __init__(self):
#         super(Concatenate, self).__init__()
     
#     def forward(self, x1, x2):
#         x = torch.cat((x1, x2), dim=1)
#         return x
    
    
# class Down(nn.Module):
#     '''Downscaling with [Convolution, Batch Norm, Activation] * n, Stride 2 Convolution, Batch Norm, Activation'''
    
#     def __init__(self, in_channels, out_channels, kernel_size=3, n_conv=1, use_bn=True, downscale=True, activation='leakyrelu'):
#         super(Down, self).__init__()
        
#         self.n_conv = n_conv
#         self.use_bn = use_bn
#         self.downscale = downscale
#         self.activation = get_activation(activation)
        
#         # conv layers 
#         self.conv_layers = nn.ModuleList(
#             [get_conv_layer(in_channels, out_channels, kernel_size)] + \
#             [get_conv_layer(out_channels, out_channels, kernel_size) for _ in range(self.n_conv-1)]
#         )
        
#         # downsampling layer 
#         if self.downscale:
#             self.down_layer = get_downscale_layer(out_channels, out_channels, kernel_size)
        
#         # batch norm layers 
#         if self.use_bn:
#             self.bn_layers = nn.ModuleList([get_batchnorm_layer(out_channels) for _ in range(self.n_conv)])
#             if self.downscale:
#                 self.down_bn = get_batchnorm_layer(out_channels)
        
#     def forward(self, x):
        
#         for i in range(self.n_conv):
#             x = self.conv_layers[i](x)
#             if self.use_bn:
#                 x = self.bn_layers[i](x)
#             x = self.activation(x)

#         before_downscale = x
#         if self.downscale:
#             x = self.down_layer(x)
#             if self.use_bn:
#                 x = self.down_bn(x)
#             x = self.activation(x)
        
#         return x, before_downscale


# class Up(nn.Module):
#     '''Upscaling with [Bilinear 2x Upsampling, Convolution, Batch Norm, Activation], Merge, [Convolution, Batch Norm, Activation] * n'''
    
#     def __init__(self, in_channels, out_channels, kernel_size=3, n_conv=1, use_bn=True, activation='leakyrelu'):
#         super(Up, self).__init__()
        
#         self.n_conv = n_conv
#         self.use_bn = use_bn
#         self.activation = get_activation(activation)
        
#         # bilinear upsampling layer 
#         self.up_layer = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        
#         # halve the channel dimension with a conv layer 
#         self.conv0 = get_conv_layer(in_channels, out_channels, kernel_size)
        
#         # concatenate layer
#         self.concat = Concatenate()
        
#         # conv layers 
#         self.conv_layers = nn.ModuleList(
#             [get_conv_layer(in_channels, out_channels, kernel_size)] + \
#             [get_conv_layer(out_channels, out_channels, kernel_size) for _ in range(self.n_conv-1)]
#         )
        
#         # batch norm layers 
#         if self.use_bn:
#             self.bn0 = get_batchnorm_layer(out_channels)
#             self.bn_layers = nn.ModuleList([get_batchnorm_layer(out_channels) for _ in range(self.n_conv)])        
    
#     def forward(self, x, xskip):
        
#         x = self.up_layer(x)
#         x = self.conv0(x)
#         if self.use_bn:
#             x = self.bn0(x)
#         x = self.activation(x)
        
#         x = self.concat(x, xskip)
        
#         for i in range(self.n_conv):
#             x = self.conv_layers[i](x)
#             if self.use_bn:
#                 x = self.bn_layers[i](x)
#             x = self.activation(x)
        
#         return x

    
# def compute_H(u):
#     p, q = u[:, 0], u[:, 1]
#     K = 0.5 * torch.sum(p**2, axis=1)
#     dq_stiff = q[:, 1::2] - q[:, ::2]
#     dq_soft = torch.stack((q[:, 0], q[:, 2]-q[:, 1], q[:, 4]-q[:, 3], -q[:, 5]), dim=1)
#     U = C0 * torch.sum(dq_stiff ** 2, axis=1) + torch.sum(dq_soft ** 4, axis=1)
#     return K + U
    
# class UNet1D(pl.LightningModule):
#     """A 1D U-Net without sigmoid output activation"""
    
#     def __init__(self, in_channels, out_channels=1, n_blocks=5, k=4, kernel_size=3, use_bn=True, n_conv=1, activation='leakyrelu',
#                 loss_fn='MSELoss', optimizer_fn='Adam', optimizer_kwargs=None, lr_scheduler_fn=None, lr_scheduler_kwargs=None, lr_scheduler_interval='step', H_strength=0):
        
#         super(UNet1D, self).__init__()
        
        
#         self.save_hyperparameters()
        
#         self.in_channels = in_channels 
#         self.out_channels = out_channels 
#         self.n_blocks = n_blocks 
#         self.k = k 
#         self.kernel_size = kernel_size 
#         self.use_bn = use_bn
#         self.n_conv = n_conv
#         self.activation = activation 
        
#         down_blocks = []
#         up_blocks = []
        
#         for i in range(self.n_blocks+1):
#             n_in = self.in_channels if i == 0 else n_out
#             n_out = 2**i * self.k
#             downscale = False if i == self.n_blocks else True 
#             down = Down(
#                 in_channels=n_in, 
#                 out_channels=n_out, 
#                 kernel_size=self.kernel_size,
#                 n_conv=self.n_conv,
#                 use_bn=self.use_bn,
#                 downscale=downscale,
#                 activation=self.activation
#             )
#             down_blocks.append(down)
        
#         for i in range(self.n_blocks):
#             n_in = n_out
#             n_out = n_in // 2
#             up = Up(
#                 in_channels=n_in, 
#                 out_channels=n_out, 
#                 kernel_size=self.kernel_size,
#                 n_conv=self.n_conv,
#                 use_bn=self.use_bn,
#                 activation=self.activation
#             )
#             up_blocks.append(up)
            
#         self.down_blocks = nn.ModuleList(down_blocks)
#         self.up_blocks = nn.ModuleList(up_blocks)
        
#         self.fc_layer = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(30, 1000),
#             get_activation(self.activation),
#             nn.Linear(1000, 1000),
#             get_activation(self.activation),
#             nn.Linear(1000, 30),
#             nn.Unflatten(1, (10, 3))
#         )
        
#         self.final_conv = get_conv_layer(in_channels=self.k, out_channels=self.out_channels, kernel_size=1)
            
# #         if self.use_bn:
# #             self.final_bn = nn.BatchNorm2d(self.out_channels)
        
#         self.loss_fn = LOSS_FN_DICT[loss_fn]
#         self.optimizer_fn = OPTIMIZER_DICT[optimizer_fn]
#         self.optimizer_kwargs = optimizer_kwargs 
#         self.lr_scheduler_fn = LR_SCHEDULER_DICT[lr_scheduler_fn] if lr_scheduler_fn is not None else None
#         self.lr_scheduler_kwargs = lr_scheduler_kwargs
#         self.lr_scheduler_interval = lr_scheduler_interval
        
#         self.H_strength = H_strength 
        
#         self.sequence_len = 1
#         self.weights = None
    
#     def set_sequence_weights(self, weights):
#         self.weights = weights 
#         self.sequence_len = len(weights)
        
#     def forward(self, x):
        
#         xs = []
        
#         for i, down in enumerate(self.down_blocks):
#             x, before_downscale = down(x)
#             xs.append(before_downscale)
            
#         x = self.fc_layer(x)
        
#         for i, up in enumerate(self.up_blocks):
#             x = up(x, xs[-2-i])
            
#         logits = self.final_conv(x) 

#         return logits
    
#     def training_step(self, batch, batch_idx):
#         losses = torch.zeros(self.sequence_len, dtype=torch.double, device=self.device)

#         u = batch[0]
#         H = compute_H(u)
            
#         for t in range(1, self.sequence_len+1):
#             u = self.forward(u)
#             if t <= 5 or self.weights[t-1] != 0:
#                 losses[t-1] = self.loss_fn(u, batch[t])
        
#         loss = losses @ self.weights
        
#         loss_H = self.loss_fn(compute_H(u), H)
#         loss += self.H_strength * loss_H
        
# #         d = torch.stack([torch.det(torch.autograd.functional.jacobian(self, u0, create_graph=True)) for u0 in batch[0]])
# #         loss_d = torch.mean((d-1)**2)
# #         loss += loss_d
        
#         self.log('step_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
#         metrics = {'loss': loss.detach()}
#         for t, l in enumerate(losses[:5]):
#             metrics[f'loss_{t+1}step'] = l.detach()
#         metrics['loss_H'] = loss_H.detach()
#         return {'loss': loss, 'batch_size': len(batch[0]), 'metrics': metrics}
  
#     def training_epoch_end(self, outputs):
#         self._shared_epoch_end(outputs, 'train')
        
#     def validation_step(self, batch, batch_idx):
#         loss, losses = self._shared_eval_step(batch, batch_idx)   
#         metrics = {'loss': loss}
#         for t, l in enumerate(losses[:5]):
#             metrics[f'loss_{t+1}step'] = l 
#         return {'batch_size': len(batch[0]), 'metrics': metrics}
    
#     def validation_epoch_end(self, outputs):
#         self._shared_epoch_end(outputs, 'val')
    
#     def test_step(self, batch, batch_idx):
#         loss, losses = self._shared_eval_step(batch, batch_idx)    
#         metrics = {'loss': loss}
#         for t, l in enumerate(losses[:5]):
#             metrics[f'loss_{t+1}step'] = l
#         return {'batch_size': len(batch[0]), 'metrics': metrics}
    
#     def test_epoch_end(self, outputs):
#         self._shared_epoch_end(outputs, 'test')
        
#     def _shared_eval_step(self, batch, batch_idx):
#         losses = torch.zeros(self.sequence_len, dtype=torch.double, device=self.device)

#         u = batch[0]
        
#         for t in range(1, self.sequence_len+1):
#             u = self.forward(u)
#             if t <= 5 or self.weights[t-1] != 0:
#                 losses[t-1] = self.loss_fn(u, batch[t])
        
#         loss = losses @ self.weights
        
# #         losses = []
# #         u = batch[0]
# #         for t in range(1, 6):
# #             u = self.forward(u)
# #             losses.append(self.loss_fn(u, batch[t]))
#         return loss, losses
    
#     def _shared_epoch_end(self, outputs, stage=None):
#         n_total = sum([out['batch_size'] for out in outputs])
        
#         def aggregate_outputs(m):
#             return torch.stack([out['metrics'][m] * out['batch_size'] for out in outputs]).sum() / n_total
        
#         metrics = {m: aggregate_outputs(m) for m in outputs[0]['metrics'].keys()}
#         logs = dict()
#         for m in metrics.keys():
#             logs['/'.join([stage, m])] = metrics[m].detach().item()
#         self.log_dict(logs)
        
#     def configure_optimizers(self):
#         optimizer = self.optimizer_fn(self.parameters(), **self.optimizer_kwargs)
#         if self.lr_scheduler_fn is not None:
#             lr_scheduler = {
#                 'scheduler': self.lr_scheduler_fn(optimizer, **self.lr_scheduler_kwargs),
#                 'interval': self.lr_scheduler_interval
#             }
#             return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
#         else:
#             return optimizer

#     def get_progress_bar_dict(self):
#         # don't show the version number 
#         items = super().get_progress_bar_dict()
#         items.pop('v_num', None)
#         return items 
