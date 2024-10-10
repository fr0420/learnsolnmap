import torch 
import numpy as np
import matplotlib.pyplot as plt

from problems.default import SeparableHamiltonianSystem
from typing import Dict


class NonlinearCoupledOscillators(SeparableHamiltonianSystem):
    """Nonlinear coupled oscillators."""

    def __init__(self, epsilon: float = 0.01) -> None:
        """Initialize the system."""
        super().__init__()
        
        # System parameters
        self.dof = 2
        self.epsilon = epsilon
        self.bounds = [
            (-1.6, 1.6), 
            (-self.epsilon*16, self.epsilon*16), 
            (-1.6, 1.6), 
            (-16, 16)
        ]  # bounds for v1, v2, x1, x2
        
        # Characteristic length scales and time scales for nondimensionalization
        # char_len = (1, 1), char_time = (1, 1/epsilon) effectively converts (v1, v2, x1, x2) to (p1, p2, q1, q2)
        self.char_len1 = 1.
        self.char_len2 = 1. 
        self.char_time1 = 1.
        self.char_time2 = 1. / self.epsilon
        self.char_vel1 = self.char_len1 / self.char_time1
        self.char_vel2 = self.char_len2 / self.char_time2
        self.char_acc1 = self.char_vel1 / self.char_time1
        self.char_acc2 = self.char_vel2 / self.char_time2

    def __repr__(self) -> str:
        return "NCO(epsilon={})".format(self.epsilon)

    def default_initial_states(self) -> torch.Tensor:
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
            np.array([v1_ic, v2_ic, x1_ic, 4.0]),      # H = 1.14500059
            np.array([v1_ic, v2_ic, x1_ic, 5.0]),      # H = 1.28151253
            np.array([v1_ic, v2_ic, x1_ic, 6.0]),      # H = 1.3635259
            np.array([v1_ic, v2_ic, x1_ic, 9.0]),      # H = 1.6429485
        ]
        return torch.stack([torch.tensor(s) for s in states])  # tensor dtype is torch.float64
    
    def compute_U(self, x: torch.Tensor) -> torch.Tensor:
        """Compute potential energy."""
        x1, x2 = x[..., 0], x[..., 1]
        U = 0.5 * (x1**2 + x2**2 * self.epsilon) + self.epsilon * x1 * x2 * torch.sin(2*(x1 + x2))
        return U
    
    def compute_K(self, v: torch.Tensor) -> torch.Tensor:
        """Compute kinetic energy."""
        v1, v2 = v[..., 0], v[..., 1]
        K = 0.5 * (v1**2 + v2**2 / self.epsilon)
        return K

    def compute_ddx(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute second derivative of x with respect to time (force/mass)."""
        x1, x2 = x[..., 0], x[..., 1]
        sine = torch.sin(2*(x1+x2))
        cosine = torch.cos(2*(x1+x2))
        ddx1 = - x1 - self.epsilon * x2 * (sine + 2 * x1 * cosine)
        ddx2 = - self.epsilon**2 * x2 - self.epsilon**2 * x1 * (sine + 2 * x2 * cosine)
        return torch.stack([ddx1, ddx2], dim=-1)
    
    def transform_to_energy_components(self, u_nd: torch.Tensor) -> torch.Tensor:
        """Transform canonical variables to variables whose squared l2-norm = Hamiltonian."""
        p, q = u_nd.chunk(2, dim=-1)
        p1, p2 = p[..., 0], p[..., 1]
        q1, q2 = q[..., 0], q[..., 1]
        return torch.stack([p1, p2 * self.epsilon**0.5, q1, q2 * self.epsilon**0.5], dim=-1)

    def compute_quantities(self, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute useful quantities accessed by model trainer."""
        v, x = u.chunk(2, dim=-1)
        return {
            "H": self.compute_Hamiltonian(u),
            "U": self.compute_U(x),
            "K": self.compute_K(v),
        }
    
    def compute_errors(self, u: torch.Tensor, u_true: torch.Tensor, reduction: str = "none") -> Dict[str, torch.Tensor]:
        """Compute errors between predicted and true states."""

        # Nondimensionalize inputs for traj error computation
        u_nd = self.nondim_u(u)
        u_true_nd = self.nondim_u(u_true)
        
        # Compute trajectory errors
        diff_squares = (u_nd - u_true_nd)**2
        abs_traj_errors = diff_squares.sum(dim=-1).sqrt()
        rel_traj_errors = abs_traj_errors / torch.sum(u_true_nd**2, dim=-1).sqrt()
        osc1_abs_traj_errors = diff_squares[..., (0, 2)].sum(dim=-1).sqrt()
        osc1_rel_traj_errors = osc1_abs_traj_errors / torch.sum(u_true_nd[..., (0, 2)]**2, dim=-1).sqrt()
        osc2_abs_traj_errors = diff_squares[..., (1, 3)].sum(dim=-1).sqrt()
        osc2_rel_traj_errors = osc2_abs_traj_errors / torch.sum(u_true_nd[..., (1, 3)]**2, dim=-1).sqrt()
        
        # Compute Hamiltonian errors
        H = self.compute_Hamiltonian(u)
        H_true = self.compute_Hamiltonian(u_true)
        abs_H_errors = torch.abs(H - H_true)
        rel_H_errors = abs_H_errors / torch.abs(H_true)

        # Apply reduction
        if reduction == "mean":
            reduction_fn = torch.mean
        elif reduction == "sum":
            reduction_fn = torch.sum
        elif reduction == "none":
            reduction_fn = lambda x: x
        else:
            raise ValueError(f"Invalid reduction type: {reduction}. Choose from 'mean', 'sum', or 'none'.")

        # Return error metrics
        return {
            "abs_traj_err": reduction_fn(abs_traj_errors),
            "rel_traj_err": reduction_fn(rel_traj_errors),
            "abs_H_err": reduction_fn(abs_H_errors),
            "rel_H_err": reduction_fn(rel_H_errors),
            "osc1_abs_traj_err": reduction_fn(osc1_abs_traj_errors),
            "osc1_rel_traj_err": reduction_fn(osc1_rel_traj_errors),
            "osc2_abs_traj_err": reduction_fn(osc2_abs_traj_errors),
            "osc2_rel_traj_err": reduction_fn(osc2_rel_traj_errors),
        }

    def nondim_u(self, u: torch.Tensor) -> torch.Tensor:
        """Nondimensionalize u = (v1, v2, x1, x2)."""
        u_nd = u.clone()
        u_nd[..., 0] = u_nd[..., 0] / self.char_vel1
        u_nd[..., 1] = u_nd[..., 1] / self.char_vel2
        u_nd[..., 2] = u_nd[..., 2] / self.char_len1
        u_nd[..., 3] = u_nd[..., 3] / self.char_len2
        return u_nd
    
    def dim_u(self, u_nd: torch.Tensor) -> torch.Tensor:
        """Dimensionalize u_nd = (v1_nd, v2_nd, x1_nd, x2_nd)."""
        u = u_nd.clone()
        u[..., 0] = u[..., 0] * self.char_vel1
        u[..., 1] = u[..., 1] * self.char_vel2
        u[..., 2] = u[..., 2] * self.char_len1
        u[..., 3] = u[..., 3] * self.char_len2
        return u

    def nondim_du(self, du: torch.Tensor) -> torch.Tensor:
        """Nondimensionalize du = (dv1, dv2, dx1, dx2)."""
        du_nd = du.clone()
        du_nd[..., 0] = du_nd[..., 0] / self.char_acc1
        du_nd[..., 1] = du_nd[..., 1] / self.char_acc2
        du_nd[..., 2] = du_nd[..., 2] / self.char_vel1
        du_nd[..., 3] = du_nd[..., 3] / self.char_vel2
        return du_nd

    def dim_du(self, du_nd: torch.Tensor) -> torch.Tensor:
        """Dimensionalize du_nd = (dv1_nd, dv2_nd, dx1_nd, dx2_nd)."""
        du = du_nd.clone()
        du[..., 0] = du[..., 0] * self.char_acc1
        du[..., 1] = du[..., 1] * self.char_acc2
        du[..., 2] = du[..., 2] * self.char_vel1
        du[..., 3] = du[..., 3] * self.char_vel2
        return du
    
    def plot_trajectories(self, trajectories: torch.Tensor) -> Dict[str, plt.Figure]:
        """Plot trajectories."""

        figures = {}

        # for i, traj in enumerate(trajectories):
        #     figures[f"traj{i+1}_energy_profile"] = self.plot_energy_profile(traj)

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

        # Phase space plot for trajectories 12, 13
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for traj in trajectories[(12, 13, 14, 15), ...]:
            ax1.plot(traj[:, 2], traj[:, 0], "-", lw=1)
            ax2.plot(traj[:, 3], traj[:, 1], "-", lw=1)
        ax1.set_xlim(-2.2, 2.2)
        ax1.set_ylim(-2.2, 2.2)
        ax1.set_xlabel("x1")
        ax1.set_ylabel("v1")
        ax1.set_title("Oscillator 1")
        ax2.set_xlim(-10., 10)
        ax2.set_ylim(-0.1, 0.1)
        ax2.set_xlabel("x2")
        ax2.set_ylabel("v2")
        ax2.set_title("Oscillator 2")

        figures["group3_trajectories"] = fig

        return figures
