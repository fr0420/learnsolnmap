import numpy as np 
import pandas as pd
from .utils import States, Trajectory, Dataset


OMEGA_VAL1 = 50.
OMEGA_VAL2 = 100.
OMEGA_VAL3 = 300.

REF_TRAJ_FILEPATHS = {
    OMEGA_VAL1: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/fpu/omega=50/1/202410071715/ref/k=0/u.csv",
            "dt": 0.00390625,
        },
        2: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/fpu/omega=50/2/202504070101/ref/u.csv",
            "dt": 0.0009765625,
        }
    }, 
    OMEGA_VAL2: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/fpu/omega=100/1/202504051956/ref/u.csv",
            "dt": 0.0009765625,
        },
    }, 
    OMEGA_VAL3: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/fpu/omega=300/1/202503252227/ref/u.csv",
            "dt": 0.0009765625,
        },
        2: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/fpu/omega=300/2/202504061417/ref/u.csv",
            "dt": 0.0009765625,
        }
    }
}

VV_TRAJ_FILEPATHS = {
    OMEGA_VAL1: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/fpu/omega=50/1/202503271137/ref/u.csv",
            "dt": 0.00390625,
        },
    }, 
    # OMEGA_VAL2: {
    #     1: {
    #         "filepath": "/workspace/projects_rui/learnsolnmap/out/fpu/omega=300/1/202503252227/vv/u.csv",
    #         "dt": 0.0009765625,
    #     },
    # }
}

class FPU:

    def __init__(self, omega=50.):
        self.omega = omega
        self.dof = 6
    
    def __eq__(self, other):
        if isinstance(other, FPU):
            return self.omega == other.omega
        return NotImplemented
    
    def __repr__(self):
        return f"FPU(omega={self.omega})"
    
    def get_pq(self, u):
        p, q = np.split(u, 2, axis=-1)
        return p, q
    
    def get_vx(self, u):
        p, q = np.split(u, 2, axis=-1)
        v = p
        x = q
        return v, x
    
    def compute_hamiltonian(self, u):
        p, q = np.split(u, 2, axis=-1)
        return self.compute_potential_energy(u) + self.compute_kinetic_energy(u)
    
    def compute_potential_energy(self, u):
        _, q = np.split(u, 2, axis=-1)
        dq_stiff = q[..., 1::2] - q[..., ::2]
        dq_soft = np.stack((q[..., 0], q[..., 2]-q[..., 1], q[..., 4]-q[..., 3], -q[..., 5]), axis=-1)
        return 0.25 * self.omega**2 * np.sum(dq_stiff**2, axis=-1) + np.sum(dq_soft**4, axis=-1)
    
    def compute_kinetic_energy(self, u):
        p, _ = np.split(u, 2, axis=-1)
        return 0.5 * np.sum(p**2, axis=-1)

    def get_slow_fast_coords(self, u):
        p, q = np.split(u, 2, axis=-1)
        y0 = (p[..., 1::2] + p[..., ::2]) / np.sqrt(2.)  # velocity of the stiff springs mass centers
        y1 = (p[..., 1::2] - p[..., ::2]) / np.sqrt(2.)  # velocity of the expansion (or compression) of the stiff springs
        x0 = (q[..., 1::2] + q[..., ::2]) / np.sqrt(2.)  # scaled displacement of the stiff springs mass centers
        x1 = (q[..., 1::2] - q[..., ::2]) / np.sqrt(2.)  # scaled expansion (or compression) of the stiff springs 
        return y0, y1, x0, x1
    
    def slow_fast_to_original(self, y0, y1, x0, x1):
        p_odd = (y0 + y1) / np.sqrt(2.) 
        p_even = (y0 - y1) / np.sqrt(2.)
        q_odd = (x0 + x1) / np.sqrt(2.)
        q_even = (x0 - x1) / np.sqrt(2.)
        p = np.zeros((len(y0), self.dof))
        q = np.zeros((len(y0), self.dof))
        p[:, 1::2] = p_odd
        p[:, ::2] = p_even
        q[:, 1::2] = q_odd
        q[:, ::2] = q_even
        return np.concatenate((p, q), axis=-1)
    
    def compute_stiff_spring_energies(self, u):
        p, q = np.split(u, 2, axis=-1)
        x1 = (q[..., 1::2] - q[..., ::2]) / np.sqrt(2.)
        y1 = (p[..., 1::2] - p[..., ::2]) / np.sqrt(2.)
        I = 0.5 * y1**2 + 0.5 * self.omega**2 * x1**2
        I_total = np.sum(I, axis=-1)
        return np.concatenate((I, I_total[:, None]), axis=-1)

    def compute_quadratic_energy_in_slow_fast_coords(self, u):
        y0, y1, x0, x1 = self.get_slow_fast_coords(u)
        I_fast = np.sum(0.5 * y1**2 + 0.5 * self.omega**2 * x1**2, axis=-1)
        I_slow = np.sum(0.5 * y0**2 + 0.5 * x0**2, axis=-1)
        return I_slow, I_fast, I_slow + I_fast

    def compute_T0(self, u):
        """Compute total kinetic energy of the mass center motion of stiff springs."""
        p, _ = np.split(u, 2, axis=-1)
        y0 = (p[..., 1::2] + p[..., ::2]) / np.sqrt(2.)
        return 0.5 * np.sum(y0**2, axis=-1)
    
    def compute_T1(self, u):
        """Compute total kinetic energy of the relative motion of masses joined by stiff springs."""
        p, _ = np.split(u, 2, axis=-1)
        y1 = (p[..., 1::2] - p[..., ::2]) / np.sqrt(2.)
        return 0.5 * np.sum(y1**2, axis=-1)
    
    def compute_du(self, u):
        """Compute the time derivative of u = (p, q)."""
        grad_H = self.compute_grad_H(u)
        dp = -grad_H[..., 6:]
        dq = grad_H[..., :6]
        return np.concatenate((dp, dq), axis=-1)
    
    def compute_ddu(self, u):
        """Compute the second time derivative of u = (p, q)."""
        if u.ndim == 1:
            u = u.reshape(1, -1)
        jacobian = self.compute_jacobian(u)
        du = self.compute_du(u)
        ddu = np.einsum('bij,bj->bi', jacobian, du)
        if len(u) == 1:
            ddu = ddu[0]
        return ddu
    
    def compute_grad_H(self, u): 
        """Compute dH/dp_i and dH/dq_i for i = 1, ..., 6."""
        p, q = np.split(u, 2, axis=-1)
        dHdp = p 
        dq_stiff = q[..., 1::2] - q[..., ::2]
        dq_soft = np.stack((q[..., 0], q[..., 2]-q[..., 1], q[..., 4]-q[..., 3], -q[..., 5]), axis=-1)
        dq_soft_cubic = dq_soft**3
        a_r = - 0.5 * self.omega**2 * dq_stiff + 4 * dq_soft_cubic[..., 1:]
        a_l = 0.5 * self.omega**2 * dq_stiff - 4 * dq_soft_cubic[..., :-1]
        ddq = np.stack((a_l[..., 0], a_r[..., 0], a_l[..., 1], a_r[..., 1], a_l[..., 2], a_r[..., 2]), axis=-1)
        dHdq = -ddq 
        return np.concatenate((dHdp, dHdq), axis=-1)

    def compute_hessian_H(self, u):
        """Compute the Hessian matrix of H with respect to p_i and q_i for i = 1, ..., 6."""
        batch_size = len(u)
        hessian_H = np.zeros((batch_size, 12, 12))

        # momentum-momentum block (diagonal)
        hessian_H[:, :6, :6] = np.eye(6)
        
        # position-position block (tri-diagonal)
        hessian_H[:, 6:, 6:] = np.eye(6) * 0.5 * self.omega**2
        _, q = np.split(u, 2, axis=-1)
        dq_soft = np.stack((q[..., 0], q[..., 2]-q[..., 1], q[..., 4]-q[..., 3], -q[..., 5]), axis=-1)
        dq_soft_sq = dq_soft**2
        for i in range(3):
            hessian_H[:, 6+2*i, 6+2*i] += 12 * dq_soft_sq[..., i]
            hessian_H[:, 6+2*i+1, 6+2*i+1] += 12 * dq_soft_sq[..., i+1]
            hessian_H[:, 6+2*i, 6+2*i+1] = - 0.5 * self.omega**2
            hessian_H[:, 6+2*i+1, 6+2*i] = - 0.5 * self.omega**2
            if i < 2:
                hessian_H[:, 6+2*i+1, 6+2*i+2] = -12 * dq_soft_sq[..., i+1]
                hessian_H[:, 6+2*i+2, 6+2*i+1] = -12 * dq_soft_sq[..., i+1]
        
        return hessian_H

    def compute_jacobian(self, u):
        """Compute the Jacobian of the vector field (dp/dt, dq/dt) with respect to (p, q)."""
        hessian_H = self.compute_hessian_H(u)
        J_inv = np.zeros((12, 12))
        J_inv[:6, 6:] = -np.eye(6)
        J_inv[6:, :6] = np.eye(6)
        jacobian = np.einsum('ij,bjk->bik', J_inv, hessian_H)
        return jacobian

    def compute_stiffness(self, u, zero_tol=1e-6, ratio=False):
        """Compute the stiffness measure of the vector field (dp/dt, dq/dt) at the point u."""
        jacobian = self.compute_jacobian(u)
        stiffness_values = np.zeros(len(u))
        for i in range(len(u)):
            try:
                eigenvalues = np.linalg.eigvals(jacobian[i])
                eigenvalues_magnitudes = np.abs(eigenvalues)
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
        """Compute the stiffness measure of the vector field (dp/dt, dq/dt) at the point u."""
        hessian_H = self.compute_hessian_H(u)
        matrices = hessian_H[:, :6, :6] @ hessian_H[:, 6:, 6:]
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
    
    def compute_energies(self, u):
        I = self.compute_stiff_spring_energies(u)
        return {
            "H": self.compute_hamiltonian(u),
            "U": self.compute_potential_energy(u),
            "K": self.compute_kinetic_energy(u),
            "T0": self.compute_T0(u),
            "T1": self.compute_T1(u),
            "I1": I[:, 0],
            "I2": I[:, 1],
            "I3": I[:, 2],
            "I_total": I[:, 3],
        }
    
    def compute_errors(self, u, ref_u):

        # Compute trajectory errors
        diff_squares = (u - ref_u)**2
        abs_traj_err = np.sqrt(np.sum(diff_squares, axis=-1))
        rel_traj_err = abs_traj_err / np.linalg.norm(ref_u, axis=-1)
        
        # Compute trajectory errors for slow and fast variables
        y0, y1, x0, x1 = self.get_slow_fast_coords(u)
        ref_y0, ref_y1, ref_x0, ref_x1 = self.get_slow_fast_coords(ref_u)
        abs_traj_err_y0 = np.sqrt(np.sum((y0 - ref_y0)**2, axis=-1))
        abs_traj_err_y1 = np.sqrt(np.sum((y1 - ref_y1)**2, axis=-1))
        abs_traj_err_x0 = np.sqrt(np.sum((x0 - ref_x0)**2, axis=-1))
        abs_traj_err_x1 = np.sqrt(np.sum((x1 - ref_x1)**2, axis=-1))
        rel_traj_err_y0 = abs_traj_err_y0 / np.linalg.norm(ref_y0, axis=-1)
        rel_traj_err_y1 = abs_traj_err_y1 / np.linalg.norm(ref_y1, axis=-1)
        rel_traj_err_x0 = abs_traj_err_x0 / np.linalg.norm(ref_x0, axis=-1)
        rel_traj_err_x1 = abs_traj_err_x1 / np.linalg.norm(ref_x1, axis=-1)

        # Compute Hamiltonian errors 
        H = self.compute_hamiltonian(u)
        ref_H = self.compute_hamiltonian(ref_u)
        abs_H_err = np.abs(H - ref_H)
        rel_H_err = abs_H_err / np.abs(ref_H)
        
        return {
            "abs_traj_err": abs_traj_err, 
            "rel_traj_err": rel_traj_err,
            "abs_traj_err_y0": abs_traj_err_y0,
            "abs_traj_err_y1": abs_traj_err_y1,
            "abs_traj_err_x0": abs_traj_err_x0,
            "abs_traj_err_x1": abs_traj_err_x1,
            "rel_traj_err_y0": rel_traj_err_y0,
            "rel_traj_err_y1": rel_traj_err_y1,
            "rel_traj_err_x0": rel_traj_err_x0,
            "rel_traj_err_x1": rel_traj_err_x1,
            "abs_H_err": abs_H_err, 
            "rel_H_err": rel_H_err
        }
    
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
            return u0 + t * self.compute_du(u0)
        elif order == 2:
            du = self.compute_du(u0)
            ddu = self.compute_ddu(u0)
            return u0 + t * du + 0.5 * t**2 * ddu
        else:
            raise ValueError("Only orders 0, 1, and 2 are supported.")


    

class FPUDataset(Dataset):

    @classmethod
    def load_from_file(cls, filepath, name="none", omega=50.):
        df = pd.read_csv(filepath)
        data = States(df.values, FPU(omega))
        return cls(data, name)
    
    @classmethod
    def from_vx(cls, vx, name="none", omega=50.):
        return cls(States(vx, FPU(omega)), name)
    
    

class FPUTrajectory(Trajectory): 

    @classmethod
    def load_from_file(cls, filepath, dt, omega=50.):
        df = pd.read_csv(filepath)
        states = States(df.values, FPU(omega))
        times = np.arange(0, len(states)) * dt
        return cls(times, states)

    @classmethod
    def from_vx(cls, times, vx, omega=50.):
        return cls(times, States(vx, FPU(omega)))