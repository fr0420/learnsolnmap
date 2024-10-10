from typing import Dict
import torch 
import numpy as np
from problems.default import SeparableHamiltonianSystem


class FPU(SeparableHamiltonianSystem):
    """Fermi-Pasta-Ulam problem."""

    def __init__(self, omega=300.0):
        self.dof = 6
        self.omega = omega
        self.C0 = 0.25 * self.omega**2
        self.bounds = [(-2, 2) for _ in range(self.dof)] + [(-1, 1) for _ in range(self.dof)]  # bounds for p, q
    
    def __repr__(self) -> str:
        return "FPU(omega={})".format(self.omega)

    def default_initial_states(self):
        """Generate initial states."""
        p0 = np.zeros(self.dof)
        q0 = np.zeros(self.dof)
        p0[1] = np.sqrt(2)
        q0[0] = (1. - 1. / self.omega) / np.sqrt(2.)
        q0[1] = (1. + 1. / self.omega) / np.sqrt(2.)
        
        # for all states, U = 1 + 3 * \omega^{-2} + 0.5 * \omega^{-4}
        states = [
            np.concatenate([p0, q0]),  # K = 1
            np.concatenate([p0/np.sqrt(2.), q0]),  # K = 0.5
            np.concatenate([p0*np.sqrt(2.), q0])   # K = 2
        ]
        return torch.stack([torch.tensor(s) for s in states])  # tensor dtype is torch.float64
    
    def compute_U(self, q: torch.Tensor) -> torch.Tensor:
        """Compute potential energy."""
        # assert shape of q 
        dq_stiff = q[..., 1::2] - q[..., ::2]
        dq_soft = torch.stack((q[..., 0], q[..., 2]-q[..., 1], q[..., 4]-q[..., 3], -q[..., 5]), dim=-1)
        U = self.C0 * torch.sum(dq_stiff**2, dim=-1) + torch.sum(dq_soft**4, dim=-1)
        return U
    
    def compute_K(self, p: torch.Tensor) -> torch.Tensor:
        """Compute kinetic energy."""
        # assert shape of p
        K = 0.5 * torch.sum(p**2, dim=-1)
        return K

    def compute_ddx(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute second derivative of x with respect to time (force/mass)."""
        # assert shape of q 
        dq_stiff = q[..., 1::2] - q[..., ::2]
        dq_soft = torch.stack((q[..., 0], q[..., 2]-q[..., 1], q[..., 4]-q[..., 3], -q[..., 5]), dim=-1)
        dq_soft_cubic = dq_soft**3

        a_r = - 2 * self.C0 * dq_stiff + 4 * dq_soft_cubic[..., 1:]
        a_l = 2 * self.C0 * dq_stiff - 4 * dq_soft_cubic[..., :-1]
        ddq = torch.stack((a_l, a_r), dim=-1)
    
        return ddq.flatten(start_dim=1)
    
    def transform_to_energy_components(self, u: torch.Tensor) -> torch.Tensor:
        """Transform canonical variables to variables whose squared l2-norm = Hamiltonian."""
        # assert shape of p, q
        p, q = u.chunk(2, dim=-1)
        dq_stiff = 0.5 * self.omega * (q[..., 1::2] - q[..., ::2])
        dq_soft = torch.stack((q[..., 0], q[..., 2]-q[..., 1], q[..., 4]-q[..., 3], -q[..., 5]), dim=-1)**2
        return torch.cat((p / 2**0.5, dq_stiff, dq_soft), dim=-1)

    def transform_to_energy_components_anchored(self, u: torch.Tensor) -> torch.Tensor:
        """Transform canonical variables to variables whose squared l2-norm = Hamiltonian."""
        # assert shape of p, q
        p, q = u.chunk(2, dim=-1)
        dq_stiff = 0.5 * self.omega * (q[..., 1::2] - q[..., ::2])
        dq_soft = torch.stack((q[..., 0], q[..., 2]-q[..., 1], q[..., 4]-q[..., 3], -q[..., 5]), dim=-1)**2
        return torch.cat((p / 2**0.5, dq_stiff, dq_soft, q), dim=-1)

    def compute_I(self, u: torch.Tensor) -> torch.Tensor:
        """Compute energy of stiff springs."""
        p, q = u.chunk(2, dim=-1)
        dq_stiff = q[..., 1::2] - q[..., ::2]
        dp_stiff = p[..., 1::2] - p[..., ::2]
        I = 0.25 * dp_stiff**2 + self.C0 * dq_stiff**2
        I_tot = torch.sum(I, dim=-1)
        return torch.column_stack((I, I_tot))

    def compute_T0(self, p: torch.Tensor) -> torch.Tensor:
        """Compute total kinetic energy of the mass center motion of stiff springs."""
        y0 = p[..., 1::2] + p[..., ::2]
        return 0.25 * torch.sum(y0**2, dim=-1)

    def compute_T1(self, p: torch.Tensor) -> torch.Tensor:
        """Compute total kinetic energy of the relative motion of masses joined by stiff springs."""
        y1 = p[..., 1::2] - p[..., ::2]
        return 0.25 * torch.sum(y1**2, dim=-1)

    def compute_quantities(self, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute useful quantities accessed by model trainer."""
        I = self.compute_I(u)
        return {
            "H": self.compute_Hamiltonian(u),
            "I_1": I[..., 0],
            "I_2": I[..., 1],
            "I_3": I[..., 2],
            "I_tot": I[..., -1]
        }

    def compute_errors(self, u: torch.Tensor, u_true: torch.Tensor, reduction: str = "none") -> Dict[str, torch.Tensor]:
        """Compute errors between predicted and true states."""
        
        # Compute trajectory errors
        diff_squares = (u - u_true)**2
        abs_traj_errors = diff_squares.sum(dim=-1).sqrt()
        rel_traj_errors = abs_traj_errors / torch.sum(u_true**2, dim=-1).sqrt()

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
        }
