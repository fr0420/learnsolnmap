import torch
from torch import nn
from networks.basics import BranchedMLP


class ModifiedMLP(nn.Module):
    """A modified MLP with gate operations."""

    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden_layers, activation, 
                 use_branched_output_layers=False):
        super(ModifiedMLP, self).__init__()

        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.encoder2 = nn.Linear(input_dim, hidden_dim)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.use_branched_output_layers = use_branched_output_layers
        if self.use_branched_output_layers:
            # use two branches, one for velocity and one for position 
            branches_input_dims = [hidden_dim//2, hidden_dim//2]
            branches_output_dims = [output_dim//2, output_dim//2]
            self.output_layer = BranchedMLP(branches_input_dims, branches_output_dims, hidden_dim//2, 2, activation)
        else:
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x, return_hidden=False):

        u = self.activation(self.encoder1(x))
        v = self.activation(self.encoder2(x))

        hs = [u, v]

        x = self.activation(self.input_layer(x))
        x = (1 - x) * u + x * v 
        hs.append(x)

        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            x = (1 - x) * u + x * v
            hs.append(x)
        
        if self.use_branched_output_layers:
            xs = x.chunk(2, dim=-1)
            xs = self.output_layer(xs)
            x = torch.cat(xs, dim=-1)
        else:
            x = self.output_layer(x)

        if return_hidden:
            return x, hs
        else:
            return x
    

class PirateNetBlock(nn.Module):
    """A residual block in a Pirate Net."""

    def __init__(self, n_features, activation, n_linears=3):
        super(PirateNetBlock, self).__init__()
        
        self.linears = nn.ModuleList(
            [nn.Linear(n_features, n_features) for _ in range(n_linears)]
        )
        self.activation = activation
        self.alpha = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, u, v):
        identity = x
        for layer in self.linears[:-1]:
            x = self.activation(layer(x))
            x = (1 - x) * u + x * v
        x = self.activation(self.linears[-1](x))
        x = self.alpha * x + (1 - self.alpha) * identity
        return x


class PirateNet(nn.Module):
    """A Pirate Net. 
    
    Reference: Wang et al., PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks, 2024
    """
    def __init__(self, input_dim, output_dim, hidden_dim, activation, 
                 n_blocks, n_linears_per_block=3, use_branched_output_layers=False):
        super(PirateNet, self).__init__()

        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.encoder2 = nn.Linear(input_dim, hidden_dim)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.use_branched_output_layers = use_branched_output_layers
        if self.use_branched_output_layers:
            # use two branches, one for velocity and one for position 
            branches_input_dims = [hidden_dim//2, hidden_dim//2]
            branches_output_dims = [output_dim//2, output_dim//2]
            self.output_layer = BranchedMLP(branches_input_dims, branches_output_dims, 
                                            hidden_dim//2, 2, activation, use_output_bias=False)
        else:
            self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        
        self.res_blocks = nn.ModuleList(
            [PirateNetBlock(hidden_dim, activation, n_linears_per_block) for _ in range(n_blocks)]
        )
        self.activation = activation
        
    def forward(self, x, return_hidden=False):
        u = self.activation(self.encoder1(x))
        v = self.activation(self.encoder2(x))
        hs = [u, v]

        x = self.activation(self.input_layer(x))
        hs.append(x)

        for block in self.res_blocks:
            x = block(x, u, v)
            hs.append(x)
        
        if self.use_branched_output_layers:
            xs = x.chunk(2, dim=-1)
            xs = self.output_layer(xs)
            x = torch.cat(xs, dim=-1)
        else:
            x = self.output_layer(x)

        if return_hidden:
            return x, hs 
        else:         
            return x