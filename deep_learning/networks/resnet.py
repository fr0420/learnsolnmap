import torch
from torch import nn
from networks.activation import get_activation


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
        
    def forward(self, x, return_hidden=False):
        hs = []
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.activation(x)
            if return_hidden:
                hs.append(x)
        x = self.layers[-1](x)

        if return_hidden:
            return x, hs 
        else:         
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
        
    def forward(self, x, return_hidden=False):
        hs = [] 
        for i, layer in enumerate(self.layers[:-1]):
            identity = x 
            x = layer(x)
            # if self.use_bn:
            #     x = self.bn_layers[i](x)
            x = self.activation(x)
            if layer.in_features == layer.out_features:
                x = identity + self.scale * x
            if self.use_bn:
                x = self.bn_layers[i](x)
            if return_hidden: 
                hs.append(x) 
        
        output_layer = self.layers[-1]
        identity = x
        x = output_layer(x)
        if output_layer.in_features == output_layer.out_features:
            x = self.activation(x)
            x = identity + self.scale * x

        if return_hidden:
            return x, hs 
        else:         
            return x


class ResMLP2(nn.Module):
    """Multi-layer perceptron with residual connections between layers of equal width"""
    
    def __init__(self, layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale):
        super(ResMLP2, self).__init__()
        
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
        
    def forward(self, x, return_hidden=False):
        init_x = x
        hs = [] 
        for i, layer in enumerate(self.layers[:-1]):
            identity = x 
            x = layer(x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.activation(x)
            if layer.in_features == layer.out_features:
                x = identity + self.scale * x
            if return_hidden: 
                hs.append(x) 
        
        output_layer = self.layers[-1]
        identity = x
        x = output_layer(x)
        if output_layer.in_features == output_layer.out_features:
            x = self.activation(x)
            x = identity + self.scale * x

        if return_hidden:
            return init_x + x, hs 
        else:
            return init_x + x
        

class ResMLP3(nn.Module):
    """Multi-layer perceptron with residual connections between layers of equal width"""

    def __init__(self, layer_sizes, activation_fn, activation_kwargs, use_bn, use_scale):
        super(ResMLP3, self).__init__()

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

    def forward(self, x, return_hidden=False):
        hs = []
        for i, layer in enumerate(self.layers[:-1]):
            identity = x
            x = layer(x)
            # if self.use_bn:
            #     x = self.bn_layers[i](x)
            x = self.activation(x)
            if layer.in_features == layer.out_features:
                x = identity + self.scale * x
            if self.use_bn:
                x = self.bn_layers[i](x)
            if return_hidden:
                hs.append(x)

        output_layer = self.layers[-1]
        identity = x
        # x = output_layer(x)
        x = torch.matmul(x, torch.t(output_layer.weight))
        if output_layer.in_features == output_layer.out_features:
            x = self.activation(x)
            x = identity + self.scale * x

        if return_hidden:
            return x, hs
        else:
            return x

