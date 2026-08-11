import os
import numpy as np 
import pandas as pd
from ..core import States, Trajectory, Dataset
from .base import BaseProblem
from ..constants import BASE_OUTPUT_DIR

# NCO-specific constants
EPSILON_VAL1 = 0.01

DT_0_01 = 0.01
DT_0_05 = 0.05

# Mapping from epsilon values to directory names
EPSILON_TO_DIR = {
    0.01: "eps=1e-2",
}

def _build_nco_filepath(epsilon, initial_condition_idx, timestamp, category='constant_energy'):
    """Build filepath for NCO trajectory files."""
    if epsilon not in EPSILON_TO_DIR:
        raise ValueError(f"Epsilon {epsilon} not found in EPSILON_TO_DIR")
    
    if category == 'constant_energy':
        dirname = f"const_energy_{initial_condition_idx}"
    elif category == 'fixed_q1':
        dirname = f"202407110236_{initial_condition_idx}"
    elif category == 'vv_constant_energy':
        dirname = f"const_energy_{initial_condition_idx}"
    else:
        raise ValueError(f"Unknown category {category}")
    
    return os.path.join(
        BASE_OUTPUT_DIR, 
        "nco", 
        EPSILON_TO_DIR[epsilon], 
        dirname,
        timestamp,
        "ref", 
        "k=0" if category != 'vv_constant_energy' else "",
        "u.csv"
    ).replace("//", "/")

# Reference trajectory filepaths for different epsilon values and initial condition indices
# Each entry contains the filepath and time step (dt) for the reference trajectory
CONSTANT_ENERGY_REF_TRAJ_FILEPATHS = {
    EPSILON_VAL1: {
        1: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 1, "202405091928", "constant_energy"),
            "dt": DT_0_05,
        },
        2: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 2, "202406151238", "constant_energy"),
            "dt": DT_0_05,
        },
        3: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 3, "202405301406", "constant_energy"),
            "dt": DT_0_05,
        }
    }
}

FIXED_Q1_REF_TRAJ_FILEPATHS = {
    EPSILON_VAL1: {
        1: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 1, "", "fixed_q1"),
            "dt": DT_0_05,
        },
        2: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 2, "", "fixed_q1"),
            "dt": DT_0_05,
        },
        3: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 3, "", "fixed_q1"),
            "dt": DT_0_05,
        },
        4: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 4, "", "fixed_q1"),
            "dt": DT_0_05,
        },
        5: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 5, "", "fixed_q1"),
            "dt": DT_0_05,
        },
        6: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 6, "", "fixed_q1"),
            "dt": DT_0_05,
        },
        7: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 7, "", "fixed_q1"),
            "dt": DT_0_05,
        },
        8: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 8, "", "fixed_q1"),
            "dt": DT_0_05,
        },
        9: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 9, "", "fixed_q1"),
            "dt": DT_0_05,
        },
        10: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 10, "", "fixed_q1"),
            "dt": DT_0_05,
        },
        11: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 11, "", "fixed_q1"),
            "dt": DT_0_05,
        }   
    }
}

# Reference trajectory filepaths generated with Velocity-Verlet method with dt=0.01
VV_0_01_CONSTANT_ENERGY_REF_TRAJ_FILEPATHS = {
    EPSILON_VAL1: {
        1: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 1, "202503242022", "vv_constant_energy"),
            "dt": DT_0_01,
        },
        2: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 2, "202503242240", "vv_constant_energy"),
            "dt": DT_0_01,
        },
        3: {
            "filepath": _build_nco_filepath(EPSILON_VAL1, 3, "202503242247", "vv_constant_energy"),
            "dt": DT_0_01,
        },
    }
}


class NCO(BaseProblem):

    def __init__(self, epsilon=0.01):
        super().__init__(dof=2)
        self.epsilon = epsilon
    
    def __repr__(self):
        return f"NCO(epsilon={self.epsilon})"

    def __eq__(self, other):
        if isinstance(other, NCO):
            return self.epsilon == other.epsilon
        return NotImplemented
    
    @classmethod
    def get_reference_filepaths(cls, category='constant_energy'):
        """
        Get reference trajectory filepaths for NCO problem.
        
        Parameters:
        -----------
        category : str, optional
            The category of reference trajectories to retrieve. Options:
            - 'constant_energy': Constant energy initial conditions (default)
            - 'fixed_q1': Fixed q1 initial conditions
            - 'vv_constant_energy': Velocity-Verlet with constant energy initial conditions
            
        Returns:
        --------
        dict
            Dictionary containing reference trajectory filepaths organized by
            initial condition indices.
        """
        if category == 'constant_energy':
            return CONSTANT_ENERGY_REF_TRAJ_FILEPATHS
        elif category == 'fixed_q1':
            return FIXED_Q1_REF_TRAJ_FILEPATHS
        elif category == 'vv_constant_energy':
            return VV_0_01_CONSTANT_ENERGY_REF_TRAJ_FILEPATHS
        else:
            raise ValueError(f"Unknown category '{category}'. Available categories: {cls.get_available_reference_categories()}")
    
    @classmethod
    def get_available_reference_categories(cls):
        """
        Get list of available reference trajectory categories for NCO.
        
        Returns:
        --------
        list
            List of available category names for reference trajectories.
        """
        return ['constant_energy', 'fixed_q1', 'vv_constant_energy']
    
    def get_pq(self, u):
        p1, p2, q1, q2 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        return p1, p2/self.epsilon, q1, q2
    
    def get_vx(self, u):
        v1, v2, x1, x2 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        return v1, v2, x1, x2
    
    def vx_to_pq(self, vx):
        v1, v2, x1, x2 = vx[..., 0], vx[..., 1], vx[..., 2], vx[..., 3]
        return np.stack([v1, v2 / self.epsilon, x1, x2], axis=-1)
    
    def pq_to_vx(self, pq):
        p1, p2, q1, q2 = pq[..., 0], pq[..., 1], pq[..., 2], pq[..., 3]
        return np.stack([p1, p2 * self.epsilon, q1, q2], axis=-1)
    
    def compute_kinetic_energy(self, u):
        v1, v2, _, _ = self.get_vx(u)
        return 0.5 * v1**2 + 0.5 * v2**2 / self.epsilon
    
    def compute_potential_energy(self, u):
        _, _, x1, x2 = self.get_vx(u)
        return 0.5 * x1**2 + 0.5 * x2**2 * self.epsilon + self.epsilon * x1 * x2 * np.sin(2*(x1 + x2))
    
    def compute_hamiltonian(self, u):
        return self.compute_kinetic_energy(u) + self.compute_potential_energy(u)

    def compute_interaction_energy(self, u):
        _, _, x1, x2 = self.get_vx(u)
        return self.epsilon * x1 * x2 * np.sin(2*(x1 + x2))
    
    def compute_energies(self, u):
        v1, v2, x1, x2 = self.get_vx(u)
        K1 = 0.5 * v1**2
        K2 = 0.5 * v2**2 / self.epsilon
        V1 = 0.5 * x1**2
        V2 = 0.5 * x2**2 * self.epsilon
        V12 = self.epsilon * x1 * x2 * np.sin(2*(x1 + x2))
        return {
            "H": K1+K2+V1+V2+V12, 
            "K1+V1": K1+V1, 
            "K2+V2": K2+V2, 
            "K2+V2+V12": K2+V2+V12,
            # "K1": K1, 
            # "K2": K2, 
            # "V1": V1, 
            # "V2": V2, 
            "V12": V12
        }

    def compute_errors(self, u, ref_u):
        # Rescale the data for traj error computation: p2 = v2 * m2 = v2 / epsilon
        pq = u * np.array([1., 1/self.epsilon, 1., 1.])
        ref_pq = ref_u * np.array([1., 1/self.epsilon, 1., 1.])

        # Compute trajectory errors
        diff_squares = (pq - ref_pq)**2
        abs_traj_err = np.sqrt(np.sum(diff_squares, axis=-1))
        rel_traj_err = abs_traj_err / np.linalg.norm(ref_pq, axis=-1)
        osc1_abs_traj_err = np.sqrt(np.sum(diff_squares[..., (0, 2)], axis=-1))
        osc1_rel_traj_err = osc1_abs_traj_err / np.linalg.norm(ref_pq[..., (0, 2)], axis=-1)
        osc2_abs_traj_err = np.sqrt(np.sum(diff_squares[..., (1, 3)], axis=-1))
        osc2_rel_traj_err = osc2_abs_traj_err / np.linalg.norm(ref_pq[..., (1, 3)], axis=-1)

        # Compute Hamiltonian errors 
        H = self.compute_hamiltonian(u)
        ref_H = self.compute_hamiltonian(ref_u)
        abs_H_err = np.abs(H - ref_H)
        rel_H_err = abs_H_err / np.abs(ref_H)
        
        return {
            "abs_traj_err": abs_traj_err, 
            "rel_traj_err": rel_traj_err,
            "osc1_abs_traj_err": osc1_abs_traj_err, 
            "osc1_rel_traj_err": osc1_rel_traj_err,
            "osc2_abs_traj_err": osc2_abs_traj_err, 
            "osc2_rel_traj_err": osc2_rel_traj_err,
            "abs_H_err": abs_H_err, 
            "rel_H_err": rel_H_err
        }

    def compute_du(self, u):
        """Compute the time derivative of (p, q)."""
        p1, p2, q1, q2 = self.get_pq(u)

        dUdq1 = q2 * np.sin(2*q1+2*q2) + q1 * q2 * 2 * np.cos(2*q1+2*q2)
        dUdq2 = q1 * np.sin(2*q1+2*q2) + q1 * q2 * 2 * np.cos(2*q1+2*q2)

        dp1 = - q1 - self.epsilon * dUdq1
        dp2 = - self.epsilon * q2 - self.epsilon * dUdq2
        dq1 = p1
        dq2 = self.epsilon * p2

        return np.stack([dp1, dp2, dq1, dq2], axis=-1)
    
    def compute_ddu(self, u):
        """Compute the second time derivative of (p, q)."""
        if u.ndim == 1:
            u = u.reshape(1, -1)
        jac = self.compute_jacobian(u) 
        du = self.compute_du(u)
        ddu = np.einsum('bij,bj->bi', jac, du)
        if len(u) == 1:
            ddu = ddu[0]
        return ddu
        
    def compute_jacobian(self, u):
        """Compute the Jacobian of the vector field (dp/dt, dq/dt) with respect to (p, q)."""
        hessian_H = self.compute_hessian_H(u)
        J_inv = np.zeros((4, 4))
        J_inv[:2, 2:] = -np.eye(2)
        J_inv[2:, :2] = np.eye(2)
        jacobian = np.einsum('ij,bjk->bik', J_inv, hessian_H)
        return jacobian

    def compute_stiffness(self, u, zero_tol=1e-6, ratio=False):
        """Compute the stiffness measure of the vector field (dp/dt, dq/dt) at the point (p, q)."""
        jacobian = self.compute_jacobian(u)
        stiffness_values = np.zeros(len(u))
        for i in range(len(u)):
            try:
                eigenvalues = np.linalg.eigvals(jacobian[i])
                eigenvalues_magnitudes = np.abs(eigenvalues)
                # print(eigenvalues)
                nonzero_vals = eigenvalues_magnitudes[eigenvalues_magnitudes > zero_tol]
                if ratio: 
                    stiffness_values[i] = np.max(nonzero_vals) / np.min(nonzero_vals)
                else:
                    stiffness_values[i] = np.max(nonzero_vals)
            except np.linalg.LinAlgError:
                stiffness_values[i] = np.nan
            except ValueError:
                stiffness_values[i] = np.nan
        return stiffness_values

    def compute_stiffness2(self, u, zero_tol=1e-14, ratio=False):
        """Compute the stiffness measure of the vector field (dp/dt, dq/dt) at the point (p, q)."""
        # Eigenvalues of the jacobian = \pm sqrt(eigenvalues of Hessian_H[:2, :2] * Hessian_H[2:, 2:])
        hessian_H = self.compute_hessian_H(u)
        matrices = hessian_H[:, :2, :2] @ hessian_H[:, 2:, 2:]
        stiffness_values = np.zeros(len(u))
        for i in range(len(u)):
            try:
                eigenvalues = np.linalg.eigvals(matrices[i])
                eigenvalues_magnitudes = np.abs(eigenvalues)
                nonzero_vals = eigenvalues_magnitudes[eigenvalues_magnitudes > zero_tol]
                sqrt_vals = np.sqrt(nonzero_vals)
                if ratio: 
                    stiffness_values[i] = np.max(sqrt_vals) / np.min(sqrt_vals)
                else:
                    stiffness_values[i] = np.max(sqrt_vals)
            except np.linalg.LinAlgError:
                stiffness_values[i] = np.nan
                print("LinAlgError")
            except ValueError:
                stiffness_values[i] = np.nan
                print("ValueError")
        return stiffness_values
    
    def compute_ddx(self, u):
        """
        ddx1 = - x1 - epsilon * dU/dx1
        ddx2 = - epsilon^2 * x2 - epsilon^2 * dU/dx2
        """
        _, _, x1, x2 = self.get_vx(u)

        dUdx1 = x2 * np.sin(2*x1+2*x2) + x1 * x2 * 2 * np.cos(2*x1+2*x2)
        dUdx2 = x1 * np.sin(2*x1+2*x2) + x1 * x2 * 2 * np.cos(2*x1+2*x2)

        ddx1 = - x1 - self.epsilon * dUdx1
        ddx2 = - self.epsilon**2 * x2 - self.epsilon**2 * dUdx2
        return np.stack([ddx1, ddx2], axis=-1)
    
    def compute_grad_H(self, u):
        """
        dH/dp1 = p1
        dH/dp2 = epsilon * p2 
        dH/dq1 = q1 + epsilon * dU/dq1
        dH/dq2 = epsilon * q2 + epsilon * dU/dq2 

        U(q1, q2) = q1 * q2 * sin(2*q1+2*q2)
        dU/dq1 = q2 * sin(2*q1+2*q2) + q1 * q2 * 2 * cos(2*q1+2*q2)
        dU/dq2 = q1 * sin(2*q1+2*q2) + q1 * q2 * 2 * cos(2*q1+2*q2)
        """
        p1, p2, q1, q2 = self.get_pq(u)

        dUdq1 = q2 * np.sin(2*q1+2*q2) + q1 * q2 * 2 * np.cos(2*q1+2*q2)
        dUdq2 = q1 * np.sin(2*q1+2*q2) + q1 * q2 * 2 * np.cos(2*q1+2*q2)

        dHdp1 = p1
        dHdp2 = self.epsilon * p2
        dHdq1 = q1 + self.epsilon * dUdq1
        dHdq2 = self.epsilon * q2 + self.epsilon * dUdq2

        return np.stack([dHdp1, dHdp2, dHdq1, dHdq2], axis=-1)
    
    def compute_hessian_H(self, u):
        """
        Compute the Hessian of the Hamiltonian H with respect to (p1, p2, q1, q2).
        
        [[d^2H/dp1^2, d^2H/dp1dp2, d^2H/dp1dq1, d^2H/dp1dq2],
         [d^2H/dp2dp1, d^2H/dp2^2, d^2H/dp2dq1, d^2H/dp2dq2],
         [d^2H/dq1dp1, d^2H/dq1dp2, d^2H/dq1^2, d^2H/dq1dq2],
         [d^2H/dq2dp1, d^2H/dq2dp2, d^2H/dq2dq1, d^2H/dq2^2]]
        
        d2H/dp1dp2 = d2H/dp1dq1 = d2H/dp1dq2 = 0
        d2H/dp2dp1 = d2H/dp2dq1 = d2H/dp2dq2 = 0
        d2H/dq1dp1 = d2H/dq1dp2 = 0 
        d2H/dq2dp1 = d2H/dq2dp2 = 0
        d2H/dp1dp1 = 1
        d2H/dp2dp2 = epsilon
        d2H/dq1dq1 = 1 + epsilon * d2U/dq1dq1
        d2H/dq2dq2 = epsilon + epsilon * d2U/dq2dq2
        d2H/dq1dq2 = epsilon * d2U/dq1dq2
        d2H/dq2dq1 = epsilon * d2U/dq2dq1

        U(q1, q2) = q1 * q2 * sin(2*q1+2*q2)
        d2U/dq1dq1 = q2 * 4 * cos(2*q1+2*q2) - q1 * q2 * 4 * sin(2*q1+2*q2)
        d2U/dq2dq2 = q1 * 4 * cos(2*q1+2*q2) - q1 * q2 * 4 * sin(2*q1+2*q2)
        d2U/dq1dq2 = d2U/dq2dq1 = sin(2*q1+2*q2) + (q1+q2) * 2 * cos(2*q1+2*q2) - q1 * q2 * 4 * sin(2*q1+2*q2)
        """
        if u.ndim == 1:
            u = u.reshape(1, -1)

        p1, p2, q1, q2 = self.get_pq(u)

        S = np.sin(2*q1+2*q2)
        C = np.cos(2*q1+2*q2)
        d2Udq1dq1 = q2 * 4 * C - q1 * q2 * 4 * S
        d2Udq2dq2 = q1 * 4 * C - q1 * q2 * 4 * S
        d2Udq1dq2 = S + (q1+q2) * 2 * C - q1 * q2 * 4 * S
        
        d2Hdp1dp1 = np.ones_like(p1)
        d2Hdp2dp2 = np.ones_like(p2) * self.epsilon
        d2Hdq1dq1 = np.ones_like(q1) + self.epsilon * d2Udq1dq1
        d2Hdq2dq2 = np.ones_like(q2) * self.epsilon + self.epsilon * d2Udq2dq2
        d2Hdq1dq2 = d2Hdq2dq1 = self.epsilon * d2Udq1dq2

        hessian = np.zeros((len(u), 4, 4))
        hessian[:, 0, 0] = d2Hdp1dp1
        hessian[:, 1, 1] = d2Hdp2dp2
        hessian[:, 2, 2] = d2Hdq1dq1
        hessian[:, 3, 3] = d2Hdq2dq2
        hessian[:, 2, 3] = d2Hdq1dq2
        hessian[:, 3, 2] = d2Hdq2dq1

        return hessian
    
    def compute_d3Hdq1dq1dq1(self, u):
        """
        d3H/dq1dq1dq1 = epsilon * d3U/dq1dq1dq1

        U(q1, q2) = q1 * q2 * sin(2*q1+2*q2)
        d2U/dq1dq1 = q2 * 4 * cos(2*q1+2*q2) - q1 * q2 * 4 * sin(2*q1+2*q2)
        d3U/dq1dq1dq1 = - q2 * 12 * sin(2*q1+2*q2) -  q1 * q2 * 8 * cos(2*q1+2*q2)
        """
        _, _, q1, q2 = self.get_pq(u)
        S = np.sin(2*q1+2*q2)
        C = np.cos(2*q1+2*q2)
        d3Udq1dq1dq1 = - q2 * 12 * S - q1 * q2 * 8 * C
        return self.epsilon * d3Udq1dq1dq1
    
    def compute_d3Hdq2dq2dq2(self, u):
        """
        d3H/dq2dq2dq2 = epsilon * d3U/dq2dq2dq2

        U(q1, q2) = q1 * q2 * sin(2*q1+2*q2)
        d2U/dq2dq2 = q1 * 4 * cos(2*q1+2*q2) - q1 * q2 * 4 * sin(2*q1+2*q2)
        d3U/dq2dq2dq2 = - q1 * 12 * sin(2*q1+2*q2) -  q1 * q2 * 8 * cos(2*q1+2*q2)
        """
        _, _, q1, q2 = self.get_pq(u)
        S = np.sin(2*q1+2*q2)
        C = np.cos(2*q1+2*q2)
        d3Udq2dq2dq2 = - q1 * 12 * S - q1 * q2 * 8 * C
        return self.epsilon * d3Udq2dq2dq2

    def compute_taylor_series(self, u0, t, order=1):
        """Compute the Taylor series expansion of the solution at time t with u(0)=u0."""
        # u0: (4,) or (batch_size, 4)
        # t: scalar or (n,) array
        # order: 0, 1, or 2
        # returns u: (n, 4) or (n, batch_size, 4)

        if isinstance(t, (int, float)):
            t = np.array([t])
        if not isinstance(t, np.ndarray):
            raise ValueError("t must be a scalar or a numpy array.")
        if t.ndim > 1:
            raise ValueError("t must be a scalar or a 1D array.")
        
        # reshape t for broadcasting
        t = t.reshape(t.shape + (1,) * u0.ndim)

        if order == 0:
            return u0[None, :].repeat(len(t), axis=0)
        elif order == 1:
            return u0 + t * self.pq_to_vx(self.compute_du(u0))
        elif order == 2:
            du = self.compute_du(u0)
            ddu = self.compute_ddu(u0)
            return u0 + t * self.pq_to_vx(du) + 0.5 * t**2 * self.pq_to_vx(ddu)
        else:
            raise ValueError("Only orders 0, 1, and 2 are supported.")
    
class NCODataset(Dataset):

    @classmethod
    def load_from_file(cls, filepath, name="none", **kwargs):
        df = pd.read_csv(filepath)
        data = States(df.values, NCO(**kwargs))
        return cls(data, name)
    
    @classmethod
    def from_vx(cls, vx, name="none", **kwargs):
        return cls(States(vx, NCO(**kwargs)), name)


class NCOTrajectory(Trajectory): 

    @classmethod
    def load_from_file(cls, filepath, dt=None, **kwargs):
        df = pd.read_csv(filepath)
        states = States(df.values, NCO(**kwargs))
        times_filepath = filepath.replace("u.csv", "t.csv")
        if os.path.exists(times_filepath):
            times = pd.read_csv(times_filepath).values.flatten()
        elif dt is not None:
            times = np.arange(0, len(states)) * dt
        else:
            raise ValueError("Either provide a valid dt or a times file.")
        return cls(times, states)

    @classmethod
    def from_vx(cls, times, vx, **kwargs):
        return cls(times, States(vx, NCO(**kwargs)))