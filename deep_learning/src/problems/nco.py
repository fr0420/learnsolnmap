import torch 
import numpy as np
import matplotlib.pyplot as plt

from problems.default import SeparableHamiltonianSystem
from typing import Dict, List, Optional


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
        self.random_params_bounds = {"epsilon": (0.001, 0.1)}

        # Characteristic length scales and time scales for nondimensionalization
        self.char_len = 1.
        self.char_time = 1.
        self.char_mass = 1.
        self.char_vel = self.char_len / self.char_time  # O(1)
        self.char_acc = self.char_vel / self.char_time  # O(1)

        # char_len = (1, 1), char_time = (1, 1/epsilon) effectively converts (v1, v2, x1, x2) to (p1, p2, q1, q2)
        # self.char_len1 = 1.
        # self.char_len2 = 1. 
        # self.char_time1 = 1.
        # self.char_time2 = 1. / self.epsilon
        # self.char_vel1 = self.char_len1 / self.char_time1  # O(1)
        # self.char_vel2 = self.char_len2 / self.char_time2  # O(epsilon)
        # self.char_acc1 = self.char_vel1 / self.char_time1  # O(1)
        # self.char_acc2 = self.char_vel2 / self.char_time2  # O(epsilon^2)

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

    def compute_U(self, x: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute potential energy."""
        x1, x2 = x[..., 0], x[..., 1]
        eps = p["epsilon"] if p is not None and "epsilon" in p else self.epsilon
        U = 0.5 * (x1**2 + x2**2 * eps) + eps * x1 * x2 * torch.sin(2*(x1 + x2))
        return U
    
    def compute_K(self, v: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute kinetic energy."""
        v1, v2 = v[..., 0], v[..., 1]
        eps = p["epsilon"] if p is not None and "epsilon" in p else self.epsilon
        K = 0.5 * (v1**2 + v2**2 / eps)
        return K

    def compute_ddx(self, x: torch.Tensor, t: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute second derivative of x with respect to time (force/mass)."""
        x1, x2 = x[..., (0,)], x[..., (1,)]
        sine = torch.sin(2*(x1+x2))
        cosine = torch.cos(2*(x1+x2))
        eps = p["epsilon"] if p is not None and "epsilon" in p else self.epsilon
        ddx1 = - x1 - eps * x2 * (sine + 2 * x1 * cosine)
        ddx2 = - eps**2 * x2 - eps**2 * x1 * (sine + 2 * x2 * cosine)
        return torch.cat([ddx1, ddx2], dim=-1)
    
    def compute_du(self, u: torch.Tensor, t: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute time derivative of u = (v, x).""" 
        v, x = u.chunk(2, dim=-1)
        dv = self.compute_ddx(x, t, p)
        dx = v
        return torch.cat((dv, dx), dim=-1)
    
    def compute_ddu(self, u: torch.Tensor, t: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute second derivative of u = (v, x) with respect to time."""
        v, x = u.chunk(2, dim=-1)
        x1, x2 = x[..., (0,)], x[..., (1,)]
        v1, v2 = v[..., (0,)], v[..., (1,)]
        sine = torch.sin(2*(x1+x2))
        cosine = torch.cos(2*(x1+x2))
        eps = p["epsilon"] if p is not None and "epsilon" in p else self.epsilon

        # Computue hessian of the coupling potential Uc(x1, x2) =  x1 * x2 * sin(2*(x1 + x2))
        d2Ucdx1dx1 = 4 * x2 * cosine - 4 * x1 * x2 * sine
        d2Ucdx2dx2 = 4 * x1 * cosine - 4 * x1 * x2 * sine
        d2Ucdx1dx2 = (1 - 4 * x1 * x2) * sine + 2 * (x1 + x2) * cosine
        ddv1 = - v1 - eps * (v1 * d2Ucdx1dx1 + v2 * d2Ucdx1dx2)
        ddv2 = - eps**2 * v2 - eps**2 * (v1 * d2Ucdx1dx2 + v2 * d2Ucdx2dx2)
        ddv =  torch.cat([ddv1, ddv2], dim=-1)
        ddx = self.compute_ddx(x, t, p)
        return torch.cat((ddv, ddx), dim=-1)
    
    def transform_to_energy_components(self, u_nd: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Transform nondimensionalized variables to energy-based variables."""
        v, x = u_nd.chunk(2, dim=-1)
        v1, v2 = v[..., (0,)], v[..., (1,)]
        x1, x2 = x[..., (0,)], x[..., (1,)]
        eps = p["epsilon"] if p is not None and "epsilon" in p else self.epsilon
        return torch.cat((v1, v2 / eps**0.5, x1, x2 / eps**0.5), dim=-1)
        # return torch.cat((v1, v2/eps, x1, x2), dim=-1)

    def compute_quantities(self, u: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Compute useful quantities accessed by model trainer."""
        v, x = u.chunk(2, dim=-1)
        return {
            "H": self.compute_Hamiltonian(u, p),
            "U": self.compute_U(x, p),
            "K": self.compute_K(v, p),
        }
    
    def compute_errors(self, u: torch.Tensor, u_true: torch.Tensor, p: Optional[Dict[str, torch.Tensor]], reduction: str = "none") -> Dict[str, torch.Tensor]:
        """Compute errors between predicted and true states."""

        # Nondimensionalize inputs for traj error computation
        u_nd = self.nondim_u(u, p)
        u_true_nd = self.nondim_u(u_true, p)
        
        # Compute trajectory errors
        diff_squares = (u_nd - u_true_nd)**2
        abs_traj_errors = diff_squares.sum(dim=-1).sqrt()
        rel_traj_errors = abs_traj_errors / torch.sum(u_true_nd**2, dim=-1).sqrt()
        osc1_abs_traj_errors = diff_squares[..., (0, 2)].sum(dim=-1).sqrt()
        osc1_rel_traj_errors = osc1_abs_traj_errors / torch.sum(u_true_nd[..., (0, 2)]**2, dim=-1).sqrt()
        osc2_abs_traj_errors = diff_squares[..., (1, 3)].sum(dim=-1).sqrt()
        osc2_rel_traj_errors = osc2_abs_traj_errors / torch.sum(u_true_nd[..., (1, 3)]**2, dim=-1).sqrt()
        
        # Compute Hamiltonian errors
        H = self.compute_Hamiltonian(u, p)
        H_true = self.compute_Hamiltonian(u_true, p)
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

    def vx_to_pq(self, vx: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Convert (v1, v2, x1, x2) to (p1, p2, q1, q2)."""
        eps = p["epsilon"] if p is not None and "epsilon" in p else self.epsilon
        q1, q2 = vx[..., (2,)], vx[..., (3,)]
        p1, p2 = vx[..., (0,)], vx[..., (1,)] / eps 
        return torch.cat([p1, p2, q1, q2], dim=-1)
    
    def pq_to_vx(self, pq: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Convert (p1, p2, q1, q2) to (v1, v2, x1, x2)."""
        eps = p["epsilon"] if p is not None and "epsilon" in p else self.epsilon
        x1, x2 = pq[..., (2,)], pq[..., (3,)]
        v1, v2 = pq[..., (0,)], pq[..., (1,)] * eps
        return torch.cat([v1, v2, x1, x2], dim=-1)
    
    def to_slow_fast_variables(self, pq: torch.Tensor) -> torch.Tensor:
        """Convert (p1, p2, q1, q2) to (ps, qs, pf, qf)."""
        p1, p2, q1, q2 = pq.chunk(4, dim=-1)
        return torch.cat([p2, q2, p1, q1], dim=-1)

    def to_original_variables(self, pqsf: torch.Tensor) -> torch.Tensor:
        """Convert (ps, qs, pf, qf) to (p1, p2, q1, q2)."""
        ps, qs, pf, qf = pqsf.chunk(4, dim=-1)
        return torch.cat([pf, ps, qf, qs], dim=-1)

    def nondim_u(self, u: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Nondimensionalize u = (v1, v2, x1, x2)."""
        u_nd = self.vx_to_pq(u, p)
        # u_nd = self.to_slow_fast_variables(u_nd)
        return u_nd
    
    def dim_u(self, u_nd: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Dimensionalize u_nd = (p1_nd, p2_nd, q1_nd, q2_nd)."""
        u = u_nd.clone()
        # u = self.to_original_variables(u)
        u = self.pq_to_vx(u, p)
        return u

    def nondim_du(self, du: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Nondimensionalize du = (dv1, dv2, dx1, dx2)."""
        du_nd = self.vx_to_pq(du, p)
        # du_nd = self.to_slow_fast_variables(du_nd)
        return du_nd

    def dim_du(self, du_nd: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Dimensionalize du_nd = (dp1_nd, dp2_nd, dq1_nd, dq2_nd)."""
        du = du_nd.clone()
        # du = self.to_original_variables(du)
        du = self.pq_to_vx(du, p)
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
