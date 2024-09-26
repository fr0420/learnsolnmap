import torch 
import numpy as np
import matplotlib.pyplot as plt
from problems.default import SeparableHamiltonianSystem


def _tri_flatten(tri, indicies_func, offset):
    N = tri.size(-1)
    indicies = indicies_func(N, N, offset)
    indicies = N * indicies[0] + indicies[1]
    return tri.flatten(-2)[..., indicies]


def tril_flatten(tril, offset):
    return _tri_flatten(tril, torch.tril_indices, offset)


def triu_flatten(triu, offset):
    return _tri_flatten(triu, torch.triu_indices, offset)


class ArgonCrystal(SeparableHamiltonianSystem):
    """Frozen Argon crystal problem."""

    def __init__(self):
        self.dof = 14
        self.Natoms = 7
        self.d = 2 
        self.MASS = 66.34e-27  # [kg]
        self.kB = 1.380658e-23  # [J / K]
        self.EPSILON = 119.8*self.kB  # [J] = [kg * nm^2 / ns^2]
        self.SIGMA = 0.341  # [nm]

        self.EPSILON_div_kB = 119.8  # [K]
        self.MASS_div_kB = self.MASS / self.kB  # [K * ns^2 / nm^2]
        self.C0 = (self.EPSILON/self.MASS)**0.5  # [nm / ns]

        self.bounds = None  # TODO 
        
    def default_initial_states(self):
        """Generate initial states."""
        
        # initial positions [nm]
        x0 = torch.tensor(
            [0.0, 0.0, 0.02, 0.39, 0.34, 0.17, 0.36, -0.21, -0.02, -0.4, -0.35, -0.16, -0.31, 0.21]
        )
        
        # initial velocities [nm/ns]
        v0_1 = torch.tensor(
            [-30.0, -20.0, 50.0, -90.0, -70.0, -60.0, 90.0, 40.0, 80.0, 90.0, -40.0, 100.0, -80.0, -60.0]
        )
        v0_2 = torch.tensor(
            [-130.0, -20.0, 150.0, -90.0, -70.0, -60.0, 90.0, 40.0, 80.0, 90.0, -40.0, 100.0, -80.0, -60.0]
        )
        v0_3 = torch.tensor(
            [0.0, -20.0, 20.0, -90.0, -50.0, -60.0, 70.0, 40.0, 80.0, 90.0, -40.0, 20.0, -80.0, -20.0]
        )

        states = [
            torch.cat([v0_1, x0], dim=-1),  # H0 = -1260 kB
            torch.cat([v0_2, x0], dim=-1),  # H0 = -1174 kB
            torch.cat([v0_3, x0], dim=-1)   # H0 = -1312 kB
        ]
        return torch.stack(states)

    def LJ_potential(self, r):
        """Lennard-Jones potential (divided by kB)."""
        return 4 * self.EPSILON_div_kB * ((self.SIGMA/r)**(12) - (self.SIGMA/r)**(6))

    def compute_U(self, x):
        """Compute potential energy (divided by kB)."""
        x_reshaped = x.view(-1, self.Natoms, self.d)
        pairwise_dist = torch.cdist(x_reshaped, x_reshaped, p=2)  # (-1, Natoms, Natoms)
        U = torch.triu(self.LJ_potential(pairwise_dist), diagonal=1).sum(dim=(-2, -1))
        return U
    
    def compute_K(self, v):
        """Compute kinetic energy (divided by kB)."""
        K = 0.5 * self.MASS_div_kB * torch.sum(v**2, dim=-1)
        return K
    
    def compute_ddx(self, x):
        """Compute second derivative of x with respect to time (force/mass)."""
        x_reshaped = x.view(-1, self.Natoms, self.d)
        pairwise_dist = torch.cdist(x_reshaped, x_reshaped, p=2)
        pairwise_diff = x_reshaped.unsqueeze(-2) - x_reshaped.unsqueeze(-3)
        fac = 2*self.SIGMA**(12) * pairwise_dist**(-14) - self.SIGMA**6 * pairwise_dist**(-8) 
        for i in range(len(fac)):
            fac[i].fill_diagonal_(0.)
        x_ddot = 24*self.EPSILON/self.MASS * torch.sum(fac.unsqueeze(-1) * pairwise_diff, dim=-2)
        return x_ddot.flatten(start_dim=-2)
    
    def transform_to_energy_components(self, u_nd):
        """Transform canonical variables to variables whose squared l2-norm = Hamiltonian + constant."""
        v, x = u_nd.chunk(2, dim=-1)
        x_reshaped = x.view(-1, self.Natoms, self.d)
        pairwise_dist = torch.cdist(x_reshaped, x_reshaped, p=2)  # (-1, Natoms, Natoms)
        d = triu_flatten(pairwise_dist, offset=1) # (-1, Natoms * (Natoms-1) / 2)
        return torch.cat((v / 2**0.5, 2 * (1/d)**6 - 1), dim=-1)
    
    def transform_to_energy_components_anchored(self, u_nd):
        """Transform canonical variables to variables whose squared l2-norm = Hamiltonian + constant."""
        v, x = u_nd.chunk(2, dim=-1)
        x_reshaped = x.view(-1, self.Natoms, self.d)
        pairwise_dist = torch.cdist(x_reshaped, x_reshaped, p=2)  # (-1, Natoms, Natoms)
        d = triu_flatten(pairwise_dist, offset=1) # (-1, Natoms * (Natoms-1) / 2)
        return torch.cat((v / 2**0.5, 2 * (1/d)**6, x), dim=-1)
        # return torch.cat((v / 2**0.5, 1 / d, x), dim=-1)
        # return torch.cat((v / 2**0.5, d, x), dim=-1)
    
    def nondimensionalize(self, u):
        v, x = u.chunk(2, dim=-1)
        return torch.cat((v / self.C0, x / self.SIGMA), dim=-1)
    
    def dimensionalize(self, u_nd):
        v_nd, x_nd = u_nd.chunk(2, dim=-1)
        return torch.cat((v_nd * self.C0, x_nd * self.SIGMA), dim=-1)
    
    def compute_temperature(self, v):
        """Compute temperature."""
        return 0.5 * self.MASS_div_kB * torch.sum(v**2, dim=-1) / self.Natoms

    def compute_quantities(self, u):
        """Compute useful quantities accessed by model trainer."""
        v, x = u.chunk(2, dim=-1)
        return {
            "H": self.compute_Hamiltonian(u),
            "T": self.compute_temperature(v)
        }

    def plot_trajectory_in_xy_plane(self, trajectory):
        """Plot trajectory in xy-plane."""
        # trajectory: (traj_len, 2 * dof)

        x = trajectory[:, 14:].cpu().numpy()
        # c = np.arange(len(trajectory))
        markers = [".", "^", "s", "o", "*", "+", "h"]

        fig, ax = plt.subplots()
        for i in range(7):
            x1 = x[:, i*2]
            x2 = x[:, i*2+1]
            ax.scatter(x1, x2, s=2, marker=markers[i])
        for i in range(7):
            x1 = x[0, i*2]
            x2 = x[0, i*2+1]
            ax.scatter(x1, x2, c="r", s=4, marker=markers[i])
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")
        # plt.grid()

        return fig
