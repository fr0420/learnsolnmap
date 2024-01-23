import torch
from torch import nn


class ResidualBlock(nn.Module):
    """A residual block."""

    def __init__(self, n_features, activation, n_linears, scaling_factor, use_bn):
        super(ResidualBlock, self).__init__()
        
        self.linears = nn.ModuleList(
            [nn.Linear(n_features, n_features) for _ in range(n_linears)]
        )
        self.activation = activation
        self.scaling_factor = scaling_factor
        self.use_bn = use_bn
        if self.use_bn:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(n_features) for _ in range(n_linears)]
            )

    def forward(self, x):
        pass
    

class PostActivationResidualBlock(ResidualBlock):
    """A post-activation residual block with [Linear, Batch Norm, Activation] * n, Addition."""

    def __init__(self, n_features, activation, n_linears=1, scaling_factor=1., use_bn=False):
        super(PostActivationResidualBlock, self).__init__(
            n_features, activation, n_linears, scaling_factor, use_bn
        )

    def forward(self, x):
        identity = x

        for i in range(len(self.linears)):
            x = self.linears[i](x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.activation(x)

        x = identity + self.scaling_factor * x

        return x


class PreActivationResidualBlock(ResidualBlock):
    """A pre-activation residual block with [Batch Norm, Activation, Linear] * n, Addition."""

    def __init__(self, n_features, activation, n_linears=1, scaling_factor=1., use_bn=False):
        super(PreActivationResidualBlock, self).__init__(
            n_features, activation, n_linears, scaling_factor, use_bn
        )

    def forward(self, x):
        identity = x

        for i in range(len(self.linears)):
            x = self.activation(x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.linears[i](x)
        x = identity + self.scaling_factor * x

        return x


class ResNet(nn.Module):
    """Residual network."""
    
    def __init__(self, input_dim, output_dim, hidden_dim, activation, 
                 n_blocks, n_linears_per_block=1, use_bn=False, use_scale=True, 
                 use_big_skip=False, block_type="pre-act"):
        super(ResNet, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        scale = 1. / n_blocks if use_scale else 1.
        self.block_type = block_type
        if block_type == "pre-act":
            self.res_blocks = nn.ModuleList(
                [PreActivationResidualBlock(hidden_dim, activation, n_linears_per_block, scale, use_bn)
                for _ in range(n_blocks)])
        elif block_type == "post-act":
            self.res_blocks = nn.ModuleList(
                [PostActivationResidualBlock(hidden_dim, activation, n_linears_per_block, scale, use_bn)
                for _ in range(n_blocks)])
        else:
            raise Exception("Invalid block_type. Must be one of ['pre-act', 'post-act].")
        
        self.activation = activation
        self.use_big_skip = use_big_skip
        
    def forward(self, x, return_hidden=False):
        init_x = x
        hs = [] 
        
        x = self.input_layer(x)
        if self.block_type == "post-act":
            x = self.activation(x)
        hs.append(x)

        for res_block in self.res_blocks:
            x = res_block(x)
            hs.append(x) 

        if self.block_type == "pre-act":
            x = self.activation(x)
        x = self.output_layer(x)
        if self.use_big_skip:
            x = init_x + x

        if return_hidden:
            return x, hs 
        else:         
            return x


class ResMLP(nn.Module):
    """Multi-layer perceptron with residual connections between layers of equal width."""
    
    def __init__(self, layer_sizes, activation, use_bn=False, use_scale=True, use_big_skip=False):
        super(ResMLP, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList(
            [nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-1)]
        )
        self.activation = activation
        self.use_bn = use_bn
        if self.use_bn:
            self.bn_layers = nn.ModuleList(
                [nn.BatchNorm1d(self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-2)]
            )
        self.scale = 1. / len(self.layers) if use_scale else 1.
        self.use_big_skip = use_big_skip

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
            # if self.use_bn:
                # x = self.bn_layers[i](x)
            if return_hidden: 
                hs.append(x) 
        
        output_layer = self.layers[-1]
        identity = x
        x = output_layer(x)
        if output_layer.in_features == output_layer.out_features:
            x = self.activation(x)
            x = identity + self.scale * x

        if self.use_big_skip:
            x = init_x + x

        if return_hidden:
            return x, hs 
        else:         
            return x
