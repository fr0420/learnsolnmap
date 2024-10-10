import numpy as np 
import pandas as pd
from .utils import States, Trajectory, Dataset


REF_TRAJ_DT = 0.05
REF_TRAJ_FILEPATHS = {
    1: "/workspace/projects_rui/learnsolnmap/out/nco/202405091928/ref/k=0/u.csv",
    2: "/workspace/projects_rui/learnsolnmap/out/nco/202407110236_1/ref/k=0/u.csv",
    3: "/workspace/projects_rui/learnsolnmap/out/nco/202407110236_2/ref/k=0/u.csv",
    4: "/workspace/projects_rui/learnsolnmap/out/nco/202407110236_3/ref/k=0/u.csv",
    5: "/workspace/projects_rui/learnsolnmap/out/nco/202407110236_4/ref/k=0/u.csv",
    6: "/workspace/projects_rui/learnsolnmap/out/nco/202407110236_5/ref/k=0/u.csv",
    7: "/workspace/projects_rui/learnsolnmap/out/nco/202407110236_6/ref/k=0/u.csv",
    8: "/workspace/projects_rui/learnsolnmap/out/nco/202407110236_7/ref/k=0/u.csv",
    11: "/workspace/projects_rui/learnsolnmap/out/nco/202405301406/ref/k=0/u.csv",
    12: "/workspace/projects_rui/learnsolnmap/out/nco/202406151238/ref/k=0/u.csv",
    13: "/workspace/projects_rui/learnsolnmap/out/nco/202407110236_8/ref/k=0/u.csv",
    14: "/workspace/projects_rui/learnsolnmap/out/nco/202407110236_9/ref/k=0/u.csv",
    15: "/workspace/projects_rui/learnsolnmap/out/nco/202407110236_10/ref/k=0/u.csv",
    16: "/workspace/projects_rui/learnsolnmap/out/nco/202407110236_11/ref/k=0/u.csv",
}


class NCO:
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon
        self.dof = 2
    
    def __eq__(self, other):
        if isinstance(other, NCO):
            return self.epsilon == other.epsilon
        return NotImplemented
    
    def __repr__(self):
        return f"NCO(epsilon={self.epsilon})"
    
    def get_pq(self, u):
        p1, p2, q1, q2 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        return p1, p2/self.epsilon, q1, q2
    
    def get_vx(self, u):
        v1, v2, x1, x2 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        return v1, v2, x1, x2
    
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

# def compute_traj_error_elementwise(sol, ref_sol, epsilon=0.01, idx=0):
#     # Rescale the data: p2 = v2 * m2 = v2 / epsilon
#     rescaled_sol = sol * np.array([1., 1/epsilon, 1., 1.])
#     rescaled_ref_sol = ref_sol * np.array([1., 1/epsilon, 1., 1.])
#     abs_err = np.abs(rescaled_sol[..., idx] - rescaled_ref_sol[..., idx])
#     rel_err = abs_err / np.abs(rescaled_ref_sol[..., idx])
#     return abs_err, rel_err


class NCODataset(Dataset):

    @classmethod
    def load_from_file(cls, filepath, name="none", epsilon=0.01):
        df = pd.read_csv(filepath)
        data = States(df.values, NCO(epsilon))
        return cls(data, name)
    

class NCOTrajectory(Trajectory): 

    @classmethod
    def load_from_file(cls, filepath, dt=0.05, epsilon=0.01):
        df = pd.read_csv(filepath)
        states = States(df.values, NCO(epsilon))
        times = np.arange(0, len(states)) * dt
        return cls(times, states)