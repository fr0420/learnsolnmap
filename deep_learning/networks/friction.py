import torch
from torch import nn

    
class FrictionBlock(nn.Module):
    """Friction Block"""
    
    def __init__(self, d, init_gamma):
        super(FrictionBlock, self).__init__()
        
        self.d = d
        self.gamma = nn.Parameter(torch.tensor(init_gamma), requires_grad=True)
        
    def forward(self, x):
        
        p, q = torch.split(x, [self.d, self.d], dim=-1)
        dp = - self.gamma**2 * p
        dq = torch.zeros_like(q)

        return torch.cat((dp, dq), dim=-1)
    