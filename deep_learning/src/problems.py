import torch 
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict
from torch import Tensor
from matplotlib.figure import Figure


def _tri_flatten(tri, indicies_func, offset):
    N = tri.size(-1)
    indicies = indicies_func(N, N, offset)
    indicies = N * indicies[0] + indicies[1]
    return tri.flatten(-2)[..., indicies]


def tril_flatten(tril, offset):
    return _tri_flatten(tril, torch.tril_indices, offset)


def triu_flatten(triu, offset):
    return _tri_flatten(triu, torch.triu_indices, offset)


class SeparableHamiltonianSystem:
    """Separable Hamiltonian system."""

    def __init___(self):
        pass 
    
    def default_initial_states(self):
        """Generate initial states."""
        pass

    def compute_Hamiltonian(self, u):
        """Compute total energy / Hamiltonian."""
        v, x = u.chunk(2, dim=-1)
        return self.compute_U(x) + self.compute_K(v)
    
    def compute_Lagrangian(self, u):
        """Compute Lagrangian."""
        v, x = u.chunk(2, dim=-1)
        return self.compute_K(v) - self.compute_U(x)
    
    def compute_U(self, x):
        """Compute potential energy."""
        pass

    def compute_K(self, v):
        """Compute kinetic energy."""
        pass

    def compute_ddx(self, x):
        """Compute second derivative with respect to x (force/mass)."""
        pass

    def transform_to_energy_components(self, u):
        """Transform canonical variables to variables whose squared l2-norm = Hamiltonian."""
        pass
    
    def compute_quantities(self, u):
        """Compute useful quantities accessed by model trainer."""
        pass 
    
    def plot_energy_profile(self, trajectory: Tensor) -> Figure:
        """Plot energy profile."""
        # trajectory: (traj_len, 2 * dof)

        n_grid = np.arange(len(trajectory))
        quantities = self.compute_quantities(trajectory)
        quantities = {key: quantities[key].cpu().numpy() for key in quantities.keys()}
        init_vals = [vals[0] for _, vals in quantities.items()]
        min_val = min(init_vals)
        max_val = max(init_vals)
        val_range = max_val - min_val

        fig, ax = plt.subplots()
        for key in quantities.keys():
            ax.plot(n_grid, quantities[key], lw=2, label=key)
        ax.set_xlim(n_grid[0], n_grid[-1])
        ax.set_ylim(min_val - 0.1*val_range, max_val + 0.1*val_range)
        ax.set_xlabel("n")
        ax.set_ylabel("energy")
        ax.legend()

        return fig

    def plot_trajectories(self, trajectories: Tensor) -> Dict[str, Figure]:
        """Create plots for several trajectories."""
        # trajectories: (n_traj, traj_len, 2 * dof)
        pass


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
        """Compute second derivative with respect to x (force/mass)."""
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

    def plot_trajectory_in_xy_plane(self, trajectory: Tensor):
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
    

class FPU(SeparableHamiltonianSystem):
    """Fermi-Pasta-Ulam problem."""

    def __init__(self, Omega=300):
        self.dof = 6
        self.Omega = Omega
        self.C0 = 0.25 * self.Omega**2

    def default_initial_states(self):
        """Generate initial states."""
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
    
    def compute_U(self, q):
        """Compute potential energy."""
        # assert shape of q 
        dq_stiff = q[:, 1::2] - q[:, ::2]
        dq_soft = torch.stack((q[:, 0], q[:, 2]-q[:, 1], q[:, 4]-q[:, 3], -q[:, 5]), dim=1)
        U = self.C0 * torch.sum(dq_stiff**2, dim=-1) + torch.sum(dq_soft**4, dim=-1)
        return U
    
    def compute_K(self, p):
        """Compute kinetic energy."""
        # assert shape of p
        K = 0.5 * torch.sum(p**2, dim=-1)
        return K

    def compute_ddx(self, q):
        """Compute second derivative with respect to x (force/mass)."""
        # assert shape of q 
        dq_stiff = q[:, 1::2] - q[:, ::2]
        dq_soft = torch.stack((q[:, 0], q[:, 2]-q[:, 1], q[:, 4]-q[:, 3], -q[:, 5]), dim=1)
        dq_soft_cubic = dq_soft**3

        a_r = - 2 * self.C0 * dq_stiff + 4 * dq_soft_cubic[:, 1:]
        a_l = 2 * self.C0 * dq_stiff - 4 * dq_soft_cubic[:, :-1]
        ddq = torch.stack((a_l, a_r), dim=-1)
    
        return ddq.flatten(start_dim=1)
    
    def transform_to_energy_components(self, u):
        """Transform canonical variables to variables whose squared l2-norm = Hamiltonian."""
        # assert shape of p, q
        p, q = u.chunk(2, dim=-1)
        dq_stiff = 0.5 * self.Omega * (q[:, 1::2] - q[:, ::2])
        dq_soft = torch.stack((q[:, 0], q[:, 2]-q[:, 1], q[:, 4]-q[:, 3], -q[:, 5]), dim=1)**2
        return torch.cat((p / 2**0.5, dq_stiff, dq_soft), dim=-1)

    def transform_to_energy_components_anchored(self, u):
        """Transform canonical variables to variables whose squared l2-norm = Hamiltonian."""
        # assert shape of p, q
        p, q = u.chunk(2, dim=-1)
        dq_stiff = 0.5 * self.Omega * (q[:, 1::2] - q[:, ::2])
        dq_soft = torch.stack((q[:, 0], q[:, 2]-q[:, 1], q[:, 4]-q[:, 3], -q[:, 5]), dim=1)**2
        return torch.cat((p / 2**0.5, dq_stiff, dq_soft, q), dim=-1)

    def compute_I(self, u):
        """Compute energy of stiff springs."""
        p, q = u.chunk(2, dim=-1)
        dq_stiff = q[:, 1::2] - q[:, ::2]
        dp_stiff = p[:, 1::2] - p[:, ::2]
        I = 0.25 * dp_stiff**2 + self.C0 * dq_stiff**2
        I_tot = torch.sum(I, dim=-1)
        return torch.column_stack((I, I_tot))

    def compute_T0(self, p):
        """Compute total kinetic energy of the mass center motion of stiff springs."""
        y0 = p[:, 1::2] + p[:, ::2]
        return 0.25 * torch.sum(y0**2, dim=-1)

    def compute_T1(self, p):
        """Compute total kinetic energy of the relative motion of masses joined by stiff springs."""
        y1 = p[:, 1::2] - p[:, ::2]
        return 0.25 * torch.sum(y1**2, dim=-1)

    def compute_quantities(self, u):
        """Compute useful quantities accessed by model trainer."""
        I = self.compute_I(u)
        return {
            "H": self.compute_Hamiltonian(u),
            "I_1": I[:, 0],
            "I_2": I[:, 1],
            "I_3": I[:, 2],
            "I_tot": I[:, -1]
        }


class NonlinearCoupledOscillators(SeparableHamiltonianSystem):
    """Nonlinear coupled oscillators."""

    def __init__(self, epsilon=0.01):
        self.dof = 2
        self.epsilon = epsilon

    def default_initial_states(self):
        """Generate initial states."""
        v1_ic = 0.
        v2_ic = 0.
        x1_ic = 1.5
        
        states = [
            np.array([v1_ic, v2_ic, x1_ic, 1.5]),       # H = 1.1299632
            np.array([v1_ic, v2_ic, x1_ic, 0.0]),       # H = 1.125
            np.array([v1_ic, v2_ic, x1_ic, 0.15]),      # H = 1.12475757
            np.array([v1_ic, v2_ic, x1_ic, 0.3]),       # H = 1.12345866
            np.array([v1_ic, v2_ic, x1_ic, 0.4]),       # H = 1.12212885
            np.array([v1_ic, v2_ic, x1_ic, 1.0]),       # H = 1.11561614
            np.array([v1_ic, v2_ic, x1_ic, 1.45]),      # H = 1.12738068
            np.array([v1_ic, v2_ic, x1_ic, 1.55]),      # H = 1.13277722
            np.array([v1_ic, v2_ic, 1.8,   1.8]),       # H = 1.66191484
            np.array([1.8,   0.008, 1.8,   1.8]),       # H = 3.28191484
            np.array([v1_ic, v2_ic, 1.0,   11.45256]),  # H = 1.1299617
            np.array([v1_ic, v2_ic, -1.45433,   5.]),   # H = 1.1299629
        ]
        return torch.stack([torch.tensor(s) for s in states])  # tensor dtype is torch.float64
    
    def compute_U(self, x):
        """Compute potential energy."""
        x1, x2 = x[..., 0], x[..., 1]
        U = 0.5 * (x1**2 + x2**2 * self.epsilon) + self.epsilon * x1 * x2 * torch.sin(2*(x1 + x2))
        return U
    
    def compute_K(self, v):
        """Compute kinetic energy."""
        v1, v2 = v[..., 0], v[..., 1]
        K = 0.5 * (v1**2 + v2**2 / self.epsilon)
        return K

    def compute_ddx(self, x):
        """Compute second derivative with respect to x (force/mass)."""
        x1, x2 = x[..., 0], x[..., 1]
        sine = torch.sin(2*(x1+x2))
        cosine = torch.cos(2*(x1+x2))
        ddx1 = - x1 - self.epsilon * x2 * (sine + 2 * x1 * cosine)
        ddx2 = - self.epsilon**2 * x2 - self.epsilon**2 * x1 * (sine + 2 * x2 * cosine)
        return torch.stack([ddx1, ddx2], dim=-1)
    
    def transform_to_energy_components(self, u_nd):
        """Transform canonical variables to variables whose squared l2-norm = Hamiltonian."""
        p, q = u_nd.chunk(2, dim=-1)
        p1, p2 = p[..., 0], p[..., 1]
        q1, q2 = q[..., 0], q[..., 1]
        return torch.stack([p1, p2 * self.epsilon**0.5, q1, q2 * self.epsilon**0.5], dim=-1)

    def compute_quantities(self, u):
        """Compute useful quantities accessed by model trainer."""
        v, x = u.chunk(2, dim=-1)
        return {
            "H": self.compute_Hamiltonian(u),
            "U": self.compute_U(x),
            "K": self.compute_K(v),
        }

    def nondimensionalize(self, u):
        """Convert (v1, v2, x1, x2) to (p1, p2, q1, q2) by scaling v2 with 1/epsilon."""
        u_nd = u.clone()
        u_nd[..., 1] = u_nd[..., 1] / self.epsilon
        return u_nd
    
    def dimensionalize(self, u_nd):
        """Convert (p1, p2, q1, q2) to (v1, v2, x1, x2) by scaling p2 with epsilon."""
        u = u_nd.clone()
        u[..., 1] = u[..., 1] * self.epsilon
        return u

    def plot_trajectories(self, trajectories: Tensor) -> Dict[str, Figure]:
        """Create plots for several trajectories."""

        figures = {}

        for i, traj in enumerate(trajectories):
            figures[f"traj{i+1}_energy_profile"] = self.plot_energy_profile(traj)

        # Phase space plot for trajectories 1-7
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for traj in trajectories[1:8]:
            ax1.plot(traj[:, 2], traj[:, 0], "-", lw=1)
            ax2.plot(traj[:, 3], traj[:, 1], "-", lw=1)
        ax1.set_xlim(-2.2, 2.2)
        ax1.set_ylim(-2.2, 2.2)
        ax1.set_xlabel("x1")
        ax1.set_ylabel("v1")
        ax1.set_title("Oscillator 1")
        ax2.set_xlim(-1., 1.6)
        ax2.set_ylim(-0.015, 0.015)
        ax2.set_xlabel("x2")
        ax2.set_ylabel("v2")
        ax2.set_title("Oscillator 2")

        figures["group1_trajectories"] = fig
        
        # Phase space plot for trajectories 0, 10, 11
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for traj in trajectories[(0, 10, 11), ...]:
            ax1.plot(traj[:, 2], traj[:, 0], "-", lw=1)
            ax2.plot(traj[:, 3], traj[:, 1], "-", lw=1)
        ax1.set_xlim(-2.2, 2.2)
        ax1.set_ylim(-2.2, 2.2)
        ax1.set_xlabel("x1")
        ax1.set_ylabel("v1")
        ax1.set_title("Oscillator 1")
        ax2.set_xlim(-15., 15)
        ax2.set_ylim(-0.15, 0.15)
        ax2.set_xlabel("x2")
        ax2.set_ylabel("v2")
        ax2.set_title("Oscillator 2")

        figures["group2_trajectories"] = fig

        return figures


if __name__ == "__main__":
    
    # prob = ArgonCrystal()
    # prob = FPU()
    prob = NonlinearCoupledOscillators()

    # print(prob.SIGMA / prob.C0)
    
    u0 = prob.default_initial_states()
    v0, x0 = u0.chunk(2, dim=-1)
    print("u0:")
    print(f"mean: {torch.mean(v0)} \t var: {torch.var(v0)}")
    print(f"mean: {torch.mean(x0)} \t var: {torch.var(x0)}")

    print(prob.compute_quantities(u0))
    print(prob.compute_ddx(x0))
    # v0 = torch.randn(3, 14) * 100
    # x0 = torch.randn(3, 14)
    # print(torch.var(v0))
    # print(torch.var(x0))
    # u0 = torch.cat([v0, x0], -1)
    
    # u0_nd = prob.nondimensionalize(u0)
    # v0, x0 = u0_nd.chunk(2, dim=-1)
    # print("u0 non-dim:")
    # print(f"mean: {torch.mean(v0)} \t var: {torch.var(v0)}")
    # print(f"mean: {torch.mean(x0)} \t var: {torch.var(x0)}")

    # z = prob.transform_to_energy_components(u0_nd)
    # print((torch.sum(z**2, -1) - 21) * prob.EPSILON_div_kB)
    # print(z.shape)
    # print(z)
    # print(f"mean: {torch.mean(z)} \t var: {torch.var(z)}")