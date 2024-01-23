import torch 
import numpy as np


class SeparableHamiltonianSystem:

    def __init___(self):
        pass 

    def compute_Hamiltonian(self, v, x):
        return self.compute_U(x) + self.compute_K(v)
    
    def compute_Lagrangian(self, v, x):
        return self.compute_K(v) - self.compute_U(x)
    
    def compute_U(self, x):
        pass

    def compute_K(self, v):
        pass

    def compute_ddx(self, x):
        pass

    def transform_to_energy_components(self, v, x):
        pass


class ArgonCrystal(SeparableHamiltonianSystem):

    def __init__(self):
        self.dof = 14
        self.Natoms = 7
        self.d = 2 
        self.MASS = 66.34e-27
        self.kB = 1.380658e-23
        self.EPSILON = 119.8*self.kB
        self.SIGMA = 0.341
        
    def LJ_potential(self, r):
        return 4 * self.EPSILON * ((self.SIGMA/r)**(12) - (self.SIGMA/r)**(6))

    def compute_U(self, x):
        x_reshaped = x.reshape(shape=(-1, self.Natoms, self.d))
        pairwise_dist = torch.cdist(x_reshaped.contiguous(), x_reshaped.contiguous(), p=2)
        U = torch.triu(self.LJ_potential(pairwise_dist), diagonal=1).sum(dim=(-2, -1))
        return U 
    
    def compute_K(self, v):
        K = 0.5 * self.MASS * torch.sum(v**2, dim=-1)
        return K 
    
    def compute_ddx(self, x):
        x_reshaped = x.reshape(shape=(-1, self.Natoms, self.d))
        pairwise_dist = torch.cdist(x_reshaped, x_reshaped, p=2)
        pairwise_diff = x_reshaped.unsqueeze(-2) - x_reshaped.unsqueeze(-3)
        fac = 2*self.SIGMA**(12) * pairwise_dist**(-14) - self.SIGMA**6 * pairwise_dist**(-8) 
        for i in range(len(fac)):
            fac[i].fill_diagonal_(0.)
        x_ddot = 24*self.EPSILON/self.MASS * torch.sum(fac.unsqueeze(-1) * pairwise_diff, dim=-2)
        return x_ddot.flatten(start_dim=-2)
    
    
class FPU(SeparableHamiltonianSystem):

    def __init__(self, Omega=300):
        self.dof = 6
        self.Omega = Omega
        self.C0 = 0.25 * self.Omega**2

    def compute_U(self, q):
        # assert shape of q 
        dq_stiff = q[:, 1::2] - q[:, ::2]
        dq_soft = torch.stack((q[:, 0], q[:, 2]-q[:, 1], q[:, 4]-q[:, 3], -q[:, 5]), dim=1)
        U = self.C0 * torch.sum(dq_stiff**2, dim=-1) + torch.sum(dq_soft**4, dim=-1)
        return U
    
    def compute_K(self, p):
        # assert shape of p
        K = 0.5 * torch.sum(p**2, dim=-1)
        return K
    

    def compute_I(self, p, q):
        dq_stiff = q[:, 1::2] - q[:, ::2]
        dp_stiff = p[:, 1::2] - p[:, ::2]
        I = 0.25 * dp_stiff**2 + self.C0 * dq_stiff**2
        I_tot = torch.sum(I, dim=-1)
        return torch.column_stack((I, I_tot))


    def compute_ddx(self, q):
        # assert shape of q 
        dq_stiff = q[:, 1::2] - q[:, ::2]
        dq_soft = torch.stack((q[:, 0], q[:, 2]-q[:, 1], q[:, 4]-q[:, 3], -q[:, 5]), dim=1)
        dq_soft_cubic = dq_soft**3

        a_r = - 2 * self.C0 * dq_stiff + 4 * dq_soft_cubic[:, 1:]
        a_l = 2 * self.C0 * dq_stiff - 4 * dq_soft_cubic[:, :-1]
        ddq = torch.stack((a_l, a_r), dim=-1)
    
        return ddq.flatten(start_dim=1)
    
    def transform_to_energy_components(self, p, q):
        # assert shape of p, q
        dq_stiff = 0.5 * self.Omega * (q[:, 1::2] - q[:, ::2])
        dq_soft = torch.stack((q[:, 0], q[:, 2]-q[:, 1], q[:, 4]-q[:, 3], -q[:, 5]), dim=1)**2
        return torch.cat((p / 2**0.5, dq_stiff, dq_soft), dim=-1)

    def default_initial_states(self):
        p0 = np.zeros(self.dof)
        q0 = np.zeros(self.dof)
        p0[1] = np.sqrt(2)
        q0[0] = (1. - 1. / self.Omega) / np.sqrt(2.)
        q0[1] = (1. + 1. / self.Omega) / np.sqrt(2.)
        
        # for all states, U = 1 + 3 * \omega^{-2} + 0.5 * \omega^{-4}
        states = [
            np.concatenate([p0, q0]),  # K = 1
            np.concatenate([p0/np.sqrt(2.), q0]),  # K = 0.5
            np.concatenate([p0*np.sqrt(2.), q0])   # K = 2
        ]
        return torch.stack([torch.tensor(s) for s in states])  # tensor dtype is torch.float64