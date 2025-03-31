from typing import Dict, Optional
import torch 
import numpy as np
from problems.default import SeparableHamiltonianSystem
from utils.sampling_utils import sample_shell


def synthetic_hmc(n_samples, y0_bounds=(1, 2), y1_bounds=(1.3, 2.0), x0_bounds=(1, 2), x1_bounds=None, omega=50.):
    if x1_bounds is None:
        x1_bounds = (y1_bounds[0] / omega, y1_bounds[1] / omega)
    r0_bounds = [y0_bounds] * 3 + [x0_bounds] * 3
    r1_bounds = [y1_bounds] * 3 + [x1_bounds] * 3
    y0x0 = sample_shell(r0_bounds, n_samples)
    y1x1 = sample_shell(r1_bounds, n_samples)
    y0, x0 = y0x0[:, :3], y0x0[:, 3:]
    y1, x1 = y1x1[:, :3], y1x1[:, 3:]
    return y0, y1, x0, x1
    # fpu = FPU(omega)
    # return fpu.slow_fast_to_original(y0, y1, x0, x1)


class FPU(SeparableHamiltonianSystem):
    """Fermi-Pasta-Ulam problem."""

    def __init__(self, omega=300.0):
        """Initialize the system."""
        super().__init__()

        # System parameters
        self.dof = 6
        self.omega = omega
        # self.C0 = 0.25 * self.omega**2
        self.bounds = [(-2, 2) for _ in range(self.dof)] + [(-1, 1) for _ in range(self.dof)]  # bounds for p, q
        self.random_params_bounds = {"omega": (10, 300)}

    def __repr__(self) -> str:
        return "FPU(omega={})".format(self.omega)

    def get_omega(self, params: Optional[Dict[str, torch.Tensor]]):
        """Helper function to get omega from params or default."""
        return params["omega"] if params is not None and "omega" in params else self.omega
    
    def default_initial_states(self) -> torch.Tensor:
        """Generate initial states using default omega."""
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

    def compute_U(self, q: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute potential energy."""
        dq_stiff = q[..., 1::2] - q[..., ::2]
        dq_soft = torch.stack((q[..., 0], q[..., 2]-q[..., 1], q[..., 4]-q[..., 3], -q[..., 5]), dim=-1)
        omega = self.get_omega(params)
        U = 0.25 * omega**2 * torch.sum(dq_stiff**2, dim=-1, keepdim=True) + torch.sum(dq_soft**4, dim=-1, keepdim=True)
        return U.squeeze(-1)
    
    def compute_K(self, p: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute kinetic energy."""
        K = 0.5 * torch.sum(p**2, dim=-1)
        return K

    def compute_ddx(self, q: torch.Tensor, t: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute second derivative of x (q) with respect to time (force/mass)."""
        dq_stiff = q[..., 1::2] - q[..., ::2]
        dq_soft = torch.stack((q[..., 0], q[..., 2]-q[..., 1], q[..., 4]-q[..., 3], -q[..., 5]), dim=-1)
        dq_soft_cubic = dq_soft**3

        omega = self.get_omega(params)
        omega_sq = omega**2
        a_r = - 0.5 * omega_sq * dq_stiff + 4 * dq_soft_cubic[..., 1:]
        a_l = 0.5 * omega_sq * dq_stiff - 4 * dq_soft_cubic[..., :-1]
        ddq = torch.stack((a_l, a_r), dim=-1)
    
        return ddq.flatten(start_dim=-2)
    
    def compute_du(self, u: torch.Tensor, t: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute time derivative of u = (p, q).""" 
        p, q = u.chunk(2, dim=-1)
        dp = self.compute_ddx(q, t, params)
        dq = p
        return torch.cat((dp, dq), dim=-1)
    
    def compute_ddu(self, u: torch.Tensor, t: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute second time derivative of u = (p, q)."""
        p, q = u.chunk(2, dim=-1)
        ddq = self.compute_ddx(q, t, params)
        ddp = torch.matmul(-self.compute_hessian_U(q, params), p.unsqueeze(-1)).squeeze(-1)
        return torch.cat((ddp, ddq), dim=-1)

    def compute_hessian_U(self, q: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute Hessian of potential energy."""
        omega = self.get_omega(params)
        omega_sq = omega**2
        diagonal = torch.stack((
            q[..., 0], 
            q[..., 2]-q[..., 1], 
            q[..., 2]-q[..., 1],
            q[..., 4]-q[..., 3], 
            q[..., 4]-q[..., 3],
            -q[..., 5]
        ), dim=-1)
        diagonal = 12 * diagonal**2 + 0.5 * omega_sq
        hessian = torch.diag_embed(diagonal)
        hessian[..., 0, (1,)] = hessian[..., 1, (0,)] = - 0.5 * omega_sq
        hessian[..., 2, (3,)] = hessian[..., 3, (2,)] = - 0.5 * omega_sq
        hessian[..., 4, (5,)] = hessian[..., 5, (4,)] = - 0.5 * omega_sq
        hessian[..., 1, 2] = hessian[..., 2, 1] = - 12 * (q[..., 2] - q[..., 1])**2 
        hessian[..., 3, 4] = hessian[..., 4, 3] = - 12 * (q[..., 4] - q[..., 3])**2
        return hessian
    
    def to_slow_fast_variables(self, u: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Convert canonical variables (p, q) to slow and fast variables (y0, x0, y1, x1)."""
        p, q = u.chunk(2, dim=-1)
        sqrt2 = 2**0.5
        y0 = (p[..., 1::2] + p[..., ::2]) / sqrt2
        y1 = (p[..., 1::2] - p[..., ::2]) / sqrt2
        x0 = (q[..., 1::2] + q[..., ::2]) / sqrt2
        x1 = (q[..., 1::2] - q[..., ::2]) / sqrt2
        return torch.cat((y0, x0, y1, x1), dim=-1)
    
    def to_original_variables(self, u: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Convert slow and fast variables (y0, x0, y1, x1) to canonical variables (p, q)."""
        y0, x0, y1, x1 = u.chunk(4, dim=-1)
        sqrt2 = 2**0.5
        p_even = (y0 + y1) / sqrt2
        p_odd = (y0 - y1) / sqrt2
        q_even = (x0 + x1) / sqrt2
        q_odd = (x0 - x1) / sqrt2
        p = torch.stack((p_odd, p_even), dim=-1).flatten(start_dim=-2)
        q = torch.stack((q_odd, q_even), dim=-1).flatten(start_dim=-2)
        return torch.cat((p, q), dim=-1)

    def transform_to_energy_components(self, u: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Transform slow and fast variables to energy-based variables."""
        y0, x0, y1, x1 = u.chunk(4, dim=-1)
        omega = self.get_omega(params)
        return torch.cat((y0, x0, y1, x1 * omega), dim=-1)

    # def transform_to_energy_components(self, u: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
    #     """Transform canonical variables to energy-based variables."""
    #     u_slow_fast = self.to_slow_fast_variables(u, params)
    #     y0, x0, y1, x1 = u_slow_fast.chunk(4, dim=-1)
    #     omega = self.get_omega(params)
    #     return torch.cat((y0, x0, y1, x1 * omega), dim=-1)

    # def transform_to_energy_components_anchored(self, u: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
    #     """Transform canonical variables to variables whose squared l2-norm = Hamiltonian."""
    #     p, q = u.chunk(2, dim=-1)
    #     dq_stiff = 0.5 * self.omega * (q[..., 1::2] - q[..., ::2])
    #     dq_soft = torch.stack((q[..., 0], q[..., 2]-q[..., 1], q[..., 4]-q[..., 3], -q[..., 5]), dim=-1)**2
    #     return torch.cat((p / 2**0.5, dq_stiff, dq_soft, q), dim=-1)
    #     # return torch.cat((p / 2**0.5, dq_stiff, q), dim=-1)

    # def transform_to_energy_components_anchored(self, u: torch.Tensor) -> torch.Tensor:
    #     """Transform canonical variables to variables whose squared l2-norm = Hamiltonian."""
    #     p, q = u.chunk(2, dim=-1)
    #     dq_stiff = (q[..., 1::2] - q[..., ::2])
    #     dq_soft = torch.stack((q[..., 0], q[..., 2]-q[..., 1], q[..., 4]-q[..., 3], -q[..., 5]), dim=-1)
    #     return torch.cat((p, dq_stiff, dq_soft, q), dim=-1)
    
    def compute_I(self, u: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute energy of stiff springs."""
        p, q = u.chunk(2, dim=-1)
        dq_stiff = q[..., 1::2] - q[..., ::2]
        dp_stiff = p[..., 1::2] - p[..., ::2]
        omega = self.get_omega(params)
        I = 0.25 * (dp_stiff**2 + omega**2 * dq_stiff**2)
        I_tot = torch.sum(I, dim=-1)
        return torch.column_stack((I, I_tot))

    def compute_T0(self, p: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute total kinetic energy of the mass center motion of stiff springs."""
        y0 = p[..., 1::2] + p[..., ::2]
        return 0.25 * torch.sum(y0**2, dim=-1)

    def compute_T1(self, p: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute total kinetic energy of the relative motion of masses joined by stiff springs."""
        y1 = p[..., 1::2] - p[..., ::2]
        return 0.25 * torch.sum(y1**2, dim=-1)
    
    def compute_quantities(self, u: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Compute useful quantities accessed by model trainer."""
        I = self.compute_I(u, params)
        return {
            "H": self.compute_Hamiltonian(u, params),
            "I_1": I[..., 0],
            "I_2": I[..., 1],
            "I_3": I[..., 2],
            "I_tot": I[..., -1]
        }

    def compute_errors(self, u: torch.Tensor, u_true: torch.Tensor, params: Optional[Dict[str, torch.Tensor]], reduction: str = "none") -> Dict[str, torch.Tensor]:
        """Compute errors between predicted and true states."""
        
        # Compute trajectory errors
        diff_squares = (u - u_true)**2
        abs_traj_errors = diff_squares.sum(dim=-1).sqrt()
        rel_traj_errors = abs_traj_errors / torch.sum(u_true**2, dim=-1).sqrt()

        # Compute trajectory errors in slow and fast variables
        u_slow_fast = self.to_slow_fast_variables(u, params)
        u_true_slow_fast = self.to_slow_fast_variables(u_true, params)
        diff_squares_slow_fast = (u_slow_fast - u_true_slow_fast)**2
        abs_traj_errors_slow_fast = diff_squares_slow_fast.sum(dim=-1).sqrt()
        rel_traj_errors_slow_fast = abs_traj_errors_slow_fast / torch.sum(u_true_slow_fast**2, dim=-1).sqrt()
        abs_traj_errors_y0 = diff_squares_slow_fast[:, :3].sum(dim=-1).sqrt()
        abs_traj_errors_x0 = diff_squares_slow_fast[:, 3:6].sum(dim=-1).sqrt()
        abs_traj_errors_y1 = diff_squares_slow_fast[:, 6:9].sum(dim=-1).sqrt()
        abs_traj_errors_x1 = diff_squares_slow_fast[:, 9:].sum(dim=-1).sqrt()
        rel_traj_errors_y0 = abs_traj_errors_y0 / torch.sum(u_true_slow_fast[:, :3]**2, dim=-1).sqrt()
        rel_traj_errors_x0 = abs_traj_errors_x0 / torch.sum(u_true_slow_fast[:, 3:6]**2, dim=-1).sqrt()
        rel_traj_errors_y1 = abs_traj_errors_y1 / torch.sum(u_true_slow_fast[:, 6:9]**2, dim=-1).sqrt()
        rel_traj_errors_x1 = abs_traj_errors_x1 / torch.sum(u_true_slow_fast[:, 9:]**2, dim=-1).sqrt()

        # Compute Hamiltonian errors
        H = self.compute_Hamiltonian(u, params)
        H_true = self.compute_Hamiltonian(u_true, params)
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
            "abs_traj_err_slow_fast": reduction_fn(abs_traj_errors_slow_fast),
            "rel_traj_err_slow_fast": reduction_fn(rel_traj_errors_slow_fast),
            "abs_traj_err_y0": reduction_fn(abs_traj_errors_y0),
            "rel_traj_err_y0": reduction_fn(rel_traj_errors_y0),
            "abs_traj_err_y1": reduction_fn(abs_traj_errors_y1),
            "rel_traj_err_y1": reduction_fn(rel_traj_errors_y1),
            "abs_traj_err_x0": reduction_fn(abs_traj_errors_x0),
            "rel_traj_err_x0": reduction_fn(rel_traj_errors_x0),
            "abs_traj_err_x1": reduction_fn(abs_traj_errors_x1),
            "rel_traj_err_x1": reduction_fn(rel_traj_errors_x1),
        }
    
    def nondim_u(self, u: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        return self.to_slow_fast_variables(u, params)
    
    def dim_u(self, u_nd: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        return self.to_original_variables(u_nd, params)
    
    def nondim_du(self, du: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        return self.to_slow_fast_variables(du, params)
    
    def dim_du(self, du_nd: torch.Tensor, params: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        return self.to_original_variables(du_nd, params)
    