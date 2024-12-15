import torch 
import numpy as np
import matplotlib.pyplot as plt

from problems.default import SeparableHamiltonianSystem
from typing import Dict


class ThreeBody(SeparableHamiltonianSystem):
    """Three-body system."""

    def __init__(self, m1=100.0, m2=1.0, m3=0.001, G=1.0) -> None:
        """Initialize the system."""
        super().__init__()
        
        # System parameters
        self.dof = 9
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.G = G
        self.bounds = [
            (-0.01, 0.01), (-0.01, 0.01), (-1.0e-06, 1.0e-06), 
            (-1.0, 1.0), (-1.0, 1.0), (-0.0004, 0.0003), 
            (-2.0, 2.0), (-2.0, 2.0), (-0.25, 0.27), 
            (-1.0, 1.0), (-1.0, 1.0), (-0.0001, 0.0001), 
            (-104, 100), (-102, 102), (-0.011, 0.011), 
            (-113, 109), (-111, 111), (-0.68, 0.68)
        ]  # bounds for v, x

    def __repr__(self) -> str:
        return f"ThreeBody(m1={self.m1}, m2={self.m2}, m3={self.m3}, G={self.G})"

    def default_initial_states(self) -> torch.Tensor:
        """Generate initial states."""
        x = np.zeros((9,))
        v = np.zeros((9,))
        x[0:3] =   [-1.00102,      0.,     0. ]
        x[3:6] =   [100.,    0.,     0. ]
        x[6:9] =   [102.,    0.,     0. ]
        v[0:3] =   [0.,     -0.010001,     -0.000001   ]
        v[3:6] =   [0.,     1.,     0.   ]
        v[6:9] =   [0.,     0.1,    0.1   ]
        
        states = [ np.concatenate([v, x]),]

        return torch.stack([torch.tensor(s) for s in states])  # tensor dtype is torch.float64
    
    def compute_U(self, x: torch.Tensor) -> torch.Tensor:
        """Compute potential energy."""
        x1, x2, x3 = x[..., 0:3], x[..., 3:6], x[..., 6:9]
        U = -self.G * (self.m1 * self.m2 / torch.norm(x1 - x2, dim=-1) \
                     + self.m1 * self.m3 / torch.norm(x1 - x3, dim=-1) \
                     + self.m2 * self.m3 / torch.norm(x2 - x3, dim=-1))
        return U
    
    def compute_K(self, v: torch.Tensor) -> torch.Tensor:
        """Compute kinetic energy."""
        v1, v2, v3 = v[..., 0:3], v[..., 3:6], v[..., 6:9]
        K = 0.5 * (self.m1 * torch.sum(v1**2, dim=-1) \
                 + self.m2 * torch.sum(v2**2, dim=-1) \
                 + self.m3 * torch.sum(v3**2, dim=-1))
        return K

    def compute_ddx(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute second derivative of x with respect to time (force/mass)."""
        x1, x2, x3 = x[..., 0:3], x[..., 3:6], x[..., 6:9]
        r12 = torch.norm(x2 - x1, dim=-1, keepdim=True)
        r13 = torch.norm(x3 - x1, dim=-1, keepdim=True)
        r23 = torch.norm(x3 - x2, dim=-1, keepdim=True)
        ddx1 = - self.G * (self.m2 * (x1 - x2) / r12**3 + self.m3 * (x1 - x3) / r13**3)
        ddx2 = - self.G * (self.m1 * (x2 - x1) / r12**3 + self.m3 * (x2 - x3) / r23**3)
        ddx3 = - self.G * (self.m1 * (x3 - x1) / r13**3 + self.m2 * (x3 - x2) / r23**3)
        return torch.cat([ddx1, ddx2, ddx3], dim=-1)

    def transform_to_energy_components(self, u_nd: torch.Tensor) -> torch.Tensor:
        """Transform canonical variables to variables whose squared l2-norm = Hamiltonian."""
        p, q = u_nd.chunk(2, dim=-1)
        p1, p2, p3 = p[..., 0:3], p[..., 3:6], p[..., 6:9]
        q1, q2, q3 = q[..., 0:3], q[..., 3:6], q[..., 6:9]
        r12 = torch.norm(q2 - q1, dim=-1, keepdim=True)
        r13 = torch.norm(q3 - q1, dim=-1, keepdim=True)
        r23 = torch.norm(q3 - q2, dim=-1, keepdim=True)
        return torch.cat((
            p1 / (2*self.m1)**0.5, p2 / (2*self.m2)**0.5, p3 / (2*self.m3)**0.5, 
            torch.sqrt(self.m1*self.m2/r12), torch.sqrt(self.m1*self.m3/r13), torch.sqrt(self.m2*self.m3/r23)), dim=-1)

    def transform_to_energy_components_anchored(self, u_nd: torch.Tensor) -> torch.Tensor:
        """Transform canonical variables to variables whose squared l2-norm = Hamiltonian."""
        p, q = u_nd.chunk(2, dim=-1)
        p1, p2, p3 = p[..., 0:3], p[..., 3:6], p[..., 6:9]
        q1, q2, q3 = q[..., 0:3], q[..., 3:6], q[..., 6:9]
        r12 = torch.norm(q2 - q1, dim=-1, keepdim=True)
        r13 = torch.norm(q3 - q1, dim=-1, keepdim=True)
        r23 = torch.norm(q3 - q2, dim=-1, keepdim=True)
        return torch.cat((
            p1 / (2*self.m1)**0.5, p2 / (2*self.m2)**0.5, p3 / (2*self.m3)**0.5, 
            torch.sqrt(self.m1*self.m2/r12), torch.sqrt(self.m1*self.m3/r13), torch.sqrt(self.m2*self.m3/r23),
            q1, q2, q3), dim=-1)

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
        abs_traj_errors_p = diff_squares[:, :9].sum(dim=-1).sqrt()
        abs_traj_errors_q = diff_squares[:, 9:].sum(dim=-1).sqrt()
        abs_traj_errors_p1 = diff_squares[:, :3].sum(dim=-1).sqrt()
        abs_traj_errors_p2 = diff_squares[:, 3:6].sum(dim=-1).sqrt()
        abs_traj_errors_p3 = diff_squares[:, 6:9].sum(dim=-1).sqrt()
        abs_traj_errors_q1 = diff_squares[:, 9:12].sum(dim=-1).sqrt()
        abs_traj_errors_q2 = diff_squares[:, 12:15].sum(dim=-1).sqrt()
        abs_traj_errors_q3 = diff_squares[:, 15:].sum(dim=-1).sqrt()
        rel_traj_errors = abs_traj_errors / torch.sum(u_true_nd**2, dim=-1).sqrt()
        rel_traj_errors_p = abs_traj_errors_p / torch.sum(u_true_nd[:, :9]**2, dim=-1).sqrt()
        rel_traj_errors_q = abs_traj_errors_q / torch.sum(u_true_nd[:, 9:]**2, dim=-1).sqrt()
        rel_traj_errors_p1 = abs_traj_errors_p1 / torch.sum(u_true_nd[:, :3]**2, dim=-1).sqrt()
        rel_traj_errors_p2 = abs_traj_errors_p2 / torch.sum(u_true_nd[:, 3:6]**2, dim=-1).sqrt()
        rel_traj_errors_p3 = abs_traj_errors_p3 / torch.sum(u_true_nd[:, 6:9]**2, dim=-1).sqrt()
        rel_traj_errors_q1 = abs_traj_errors_q1 / torch.sum(u_true_nd[:, 9:12]**2, dim=-1).sqrt()
        rel_traj_errors_q2 = abs_traj_errors_q2 / torch.sum(u_true_nd[:, 12:15]**2, dim=-1).sqrt()
        rel_traj_errors_q3 = abs_traj_errors_q3 / torch.sum(u_true_nd[:, 15:]**2, dim=-1).sqrt()

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
            "abs_traj_err_p": reduction_fn(abs_traj_errors_p),
            "rel_traj_err_p": reduction_fn(rel_traj_errors_p),
            "abs_traj_err_q": reduction_fn(abs_traj_errors_q),
            "rel_traj_err_q": reduction_fn(rel_traj_errors_q),
            "abs_traj_err_p1": reduction_fn(abs_traj_errors_p1),
            "rel_traj_err_p1": reduction_fn(rel_traj_errors_p1),
            "abs_traj_err_p2": reduction_fn(abs_traj_errors_p2),
            "rel_traj_err_p2": reduction_fn(rel_traj_errors_p2),
            "abs_traj_err_p3": reduction_fn(abs_traj_errors_p3),
            "rel_traj_err_p3": reduction_fn(rel_traj_errors_p3),
            "abs_traj_err_q1": reduction_fn(abs_traj_errors_q1),
            "rel_traj_err_q1": reduction_fn(rel_traj_errors_q1),
            "abs_traj_err_q2": reduction_fn(abs_traj_errors_q2),
            "rel_traj_err_q2": reduction_fn(rel_traj_errors_q2),
            "abs_traj_err_q3": reduction_fn(abs_traj_errors_q3),
            "rel_traj_err_q3": reduction_fn(rel_traj_errors_q3),
            "abs_H_err": reduction_fn(abs_H_errors),
            "rel_H_err": reduction_fn(rel_H_errors),
        }
    
    def plot_trajectories(self, trajectories: torch.Tensor) -> Dict[str, plt.Figure]:
        """Plot trajectories."""

        figures = {}

        for i, traj in enumerate(trajectories):
            figures[f"traj{i+1}_energy_profile"] = self.plot_energy_profile(traj)

        # 3D trajectory plot for traj 1
        traj = trajectories[0]
        _, x = traj.chunk(2, dim=-1)
        x1, x2, x3 = x[..., 0:3], x[..., 3:6], x[..., 6:9]
        x1, x2, x3 = x1.cpu().numpy(), x2.cpu().numpy(), x3.cpu().numpy()
        axis_labels = [('$q_{1x}$', '$q_{1y}$', '$q_{1z}$'), 
                       ('$q_{2x}$', '$q_{2y}$', '$q_{2z}$'),
                       ('$q_{3x}$', '$q_{3y}$', '$q_{3z}$')]
        
        fig = plt.figure(figsize=(12, 4))
        axes = fig.subplots(1, 3, subplot_kw={"projection": "3d"}) 
        axes[0].plot(x1[:, 0], x1[:, 1], x1[:, 2], "-", lw=1)
        axes[1].plot(x2[:, 0], x2[:, 1], x2[:, 2], "-", lw=1)
        axes[2].plot(x3[:, 0], x3[:, 1], x3[:, 2], "-", lw=1)
        for i, ax in enumerate(axes):
            ax.set_xlabel(axis_labels[i][0])
            ax.set_ylabel(axis_labels[i][1])
            ax.set_zlabel(axis_labels[i][2])
            ax.set_title(f"Body {i+1}")
        axes[0].set_xlim(-1.1, 1.1)
        axes[0].set_ylim(-1.1, 1.1)
        axes[0].set_zlim(-1.1e-4, 1.1e-4)
        axes[1].set_xlim(-1.1e2, 1.1e2)
        axes[1].set_ylim(-1.1e2, 1.1e2)
        axes[1].set_zlim(-1.1e-2, 1.1e-2)
        axes[2].set_xlim(-1.1e2, 1.1e2)
        axes[2].set_ylim(-1.1e2, 1.1e2)
        axes[2].set_zlim(-1.1, 1.1)
        fig.tight_layout()
        figures["traj1_xyz"] = fig

        # xy-plane trajectory plot for traj 1
        traj = trajectories[0]
        _, x = traj.chunk(2, dim=-1)
        x1, x2, x3 = x[..., 0:3], x[..., 3:6], x[..., 6:9]
        x1, x2, x3 = x1.cpu().numpy(), x2.cpu().numpy(), x3.cpu().numpy()
        axis_labels = [('$q_{1x}$', '$q_{1y}$'), 
                       ('$q_{2x}$', '$q_{2y}$'),
                       ('$q_{3x}$', '$q_{3y}$')]
        
        fig = plt.figure(figsize=(12, 4))
        axes = fig.subplots(1, 3) 
        axes[0].plot(x1[:, 0], x1[:, 1], "-", lw=1)
        axes[1].plot(x2[:, 0], x2[:, 1], "-", lw=1)
        axes[2].plot(x3[:, 0], x3[:, 1], "-", lw=1)
        for i, ax in enumerate(axes):
            ax.set_xlabel(axis_labels[i][0])
            ax.set_ylabel(axis_labels[i][1])
            ax.set_title(f"Body {i+1}")
        axes[0].set_xlim(-1.1, 1.1)
        axes[0].set_ylim(-1.1, 1.1)
        axes[1].set_xlim(-1.1e2, 1.1e2)
        axes[1].set_ylim(-1.1e2, 1.1e2)
        axes[2].set_xlim(-1.1e2, 1.1e2)
        axes[2].set_ylim(-1.1e2, 1.1e2)
        fig.tight_layout()
        figures["traj1_xy"] = fig

        return figures

    def nondim_u(self, u: torch.Tensor) -> torch.Tensor:
        """Nondimensionalize the input states."""
        # Convert v,x to p,q
        v, x = u.chunk(2, dim=-1)
        v1, v2, v3 = v[..., 0:3], v[..., 3:6], v[..., 6:9]
        p1 = self.m1 * v1
        p2 = self.m2 * v2
        p3 = self.m3 * v3
        u_nd = torch.cat([p1, p2, p3, x], dim=-1)
        return u_nd
    
    def dim_u(self, u_nd: torch.Tensor) -> torch.Tensor:
        """Dimensionalize the input states."""
        # Convert p,q to v,x
        p, q = u_nd.chunk(2, dim=-1)
        p1, p2, p3 = p[..., 0:3], p[..., 3:6], p[..., 6:9]
        v1 = p1 / self.m1
        v2 = p2 / self.m2
        v3 = p3 / self.m3
        u = torch.cat([v1, v2, v3, q], dim=-1)
        return u
    
    def nondim_du(self, du: torch.Tensor) -> torch.Tensor:
        """Nondimensionalize the input state derivatives."""
        # Convert dv, dx to dp, dq
        dv, dx = du.chunk(2, dim=-1)
        dv1, dv2, dv3 = dv[..., 0:3], dv[..., 3:6], dv[..., 6:9]
        dp1 = self.m1 * dv1
        dp2 = self.m2 * dv2
        dp3 = self.m3 * dv3
        du_nd = torch.cat([dp1, dp2, dp3, dx], dim=-1)
        return du_nd
    
    def dim_du(self, du_nd: torch.Tensor) -> torch.Tensor:
        """Dimensionalize the input state derivatives."""
        # Convert dp, dq to dv, dx
        dp, dq = du_nd.chunk(2, dim=-1)
        dp1, dp2, dp3 = dp[..., 0:3], dp[..., 3:6], dp[..., 6:9]
        dv1 = dp1 / self.m1
        dv2 = dp2 / self.m2
        dv3 = dp3 / self.m3
        du = torch.cat([dv1, dv2, dv3, dq], dim=-1)
        return du
