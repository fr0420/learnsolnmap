import torch 
import numpy as np


class LennardJones:

    def __init__(self):
        self.Natoms = 7
        self.d = 2 
        self.MASS = 66.34e-27
        self.kB = 1.380658e-23
        self.EPSILON = 119.8*self.kB
        self.SIGMA = 0.341
        
    def LJ_potential(self, r):
        return 4 * self.EPSILON * ((self.SIGMA/r)**(12) - (self.SIGMA/r)**(6))
    
    def compute_Lagrangian(self, v, x):
        return self.compute_K(v) - self.compute_U(x)
    
    def compute_H(self, v, x):
        return self.compute_U(x) + self.compute_K(v)
    
    def compute_U(self, x):
        x_reshaped = x.reshape(shape=(-1, self.Natoms, self.d))
        pairwise_dist = torch.cdist(x_reshaped.contiguous(), x_reshaped.contiguous(), p=2)
        U = torch.triu(self.LJ_potential(pairwise_dist), diagonal=1).sum(dim=(-2, -1))
        return U 
    
    def compute_K(self, v):
        K = 0.5 * self.MASS * torch.sum(v**2, dim=-1)
        return K 
    
    def compute_x_ddot(self, x):
        x_reshaped = x.reshape(shape=(-1, self.Natoms, self.d))
        pairwise_dist = torch.cdist(x_reshaped, x_reshaped, p=2)
        pairwise_diff = x_reshaped.unsqueeze(-2) - x_reshaped.unsqueeze(-3)
        fac = 2*self.SIGMA**(12) * pairwise_dist**(-14) - self.SIGMA**6 * pairwise_dist**(-8) 
        for i in range(len(fac)):
            fac[i].fill_diagonal_(0.)
        x_ddot = 24*self.EPSILON/self.MASS * torch.sum(fac.unsqueeze(-1) * pairwise_diff, dim=-2)
        return x_ddot.flatten(start_dim=-2)
    
    
class FPU:
    def __init__(self, Omega=300):
        self.Omega = Omega
        self.C0 = 0.25 * self.Omega**2
    
    def compute_Lagrangian(self, p, q):
        return self.compute_K(p) - self.compute_U(q)
        
    def compute_H(self, p, q):
        return self.compute_U(q) + self.compute_K(p)
    
    def compute_U(self, q):
        dq_stiff = q[:, 1::2] - q[:, ::2]
        dq_soft = torch.stack((q[:, 0], q[:, 2]-q[:, 1], q[:, 4]-q[:, 3], -q[:, 5]), dim=1)
        U = self.C0 * torch.sum(dq_stiff**2, dim=-1) + torch.sum(dq_soft**4, dim=-1)
        return U
    
    def compute_K(self, p):
        K = 0.5 * torch.sum(p**2, dim=-1)
        return K
    
    def compute_q_ddot(self, q):
        dq_stiff = q[:, 1::2] - q[:, ::2]
        dq_soft = torch.stack((q[:, 0], q[:, 2]-q[:, 1], q[:, 4]-q[:, 3], -q[:, 5]), dim=1)

        a_r = - 2 * self.C0 * dq_stiff + 4 * dq_soft[:, 1:]**3
        a_l = 2 * self.C0 * dq_stiff - 4 * dq_soft[:, :-1]**3
        q_ddot = torch.stack((a_l, a_r), dim=-1)
    
        return q_ddot.flatten(start_dim=1)
    
    def Lambda_transform(self, p, q):
        dq_stiff = 0.5 * self.Omega * (q[:, 1::2] - q[:, ::2])
        dq_soft = torch.stack((q[:, 0], q[:, 2]-q[:, 1], q[:, 4]-q[:, 3], -q[:, 5]), dim=1)**2
        return torch.cat((p / 2**0.5, dq_stiff, dq_soft), dim=1)
    
    def Lambda2_transform(self, p, q):
        dq_stiff = 0.5 * self.Omega * (q[:, 1::2] - q[:, ::2])
        dq_soft = torch.stack((q[:, 0], q[:, 2]-q[:, 1], q[:, 4]-q[:, 3], -q[:, 5]), dim=1)**2
        return torch.cat((p / 2**0.5, dq_stiff, dq_soft, q[:, 0]), dim=1)