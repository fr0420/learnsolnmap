import numpy as np 
import pandas as pd
from ..core import States, Trajectory, Dataset
from .base import BaseProblem


REF_TRAJ_DT = 1e-2 
REF_TRAJ_FILEPATHS = {
    1: "/workspace/projects_rui/learnsolnmap/out/3body/202410302146/ref/k=0/u.csv",
    0: "/workspace/projects_rui/learnsolnmap/out/3body/202410232146/ref/k=0/u.csv",
}

class ThreeBody(BaseProblem):

    def __init__(self, m1=100., m2=1., m3=0.001, G=1.):
        super().__init__(dof=9)
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.G = G
    
    def __eq__(self, other):
        if isinstance(other, ThreeBody):
            return self.m1 == other.m1 and self.m2 == other.m2 and self.m3 == other.m3 and self.G == other.G
        return NotImplemented
    
    def __repr__(self):
        return f"ThreeBody(m1={self.m1}, m2={self.m2}, m3={self.m3}, G={self.G})"
    
    def get_pq(self, u):
        v, x = np.split(u, 2, axis=-1)
        p1 = self.m1 * v[..., :3]
        p2 = self.m2 * v[..., 3:6]
        p3 = self.m3 * v[..., 6:]
        q1 = x[..., :3]
        q2 = x[..., 3:6]
        q3 = x[..., 6:]
        return (p1, p2, p3), (q1, q2, q3)
    
    def get_vx(self, u):
        v, x = np.split(u, 2, axis=-1)
        v1 = v[..., :3]
        v2 = v[..., 3:6]
        v3 = v[..., 6:]
        x1 = x[..., :3]
        x2 = x[..., 3:6]
        x3 = x[..., 6:]
        return (v1, v2, v3), (x1, x2, x3)
    
    def convert_pq_to_vx(self, pq):
        p, q = np.split(pq, 2, axis=-1)
        v1 = p[..., :3] / self.m1
        v2 = p[..., 3:6] / self.m2
        v3 = p[..., 6:] / self.m3
        x = q 
        return np.concatenate((v1, v2, v3, x), axis=-1)
    
    def convert_vx_to_pq(self, vx):
        v, x = np.split(vx, 2, axis=-1)
        p1 = self.m1 * v[..., :3]
        p2 = self.m2 * v[..., 3:6]
        p3 = self.m3 * v[..., 6:]
        q = x
        return np.concatenate((p1, p2, p3, q), axis=-1)
    
    def compute_hamiltonian(self, u):
        return self.compute_potential_energy(u) + self.compute_kinetic_energy(u)
    
    def compute_potential_energy(self, u):
        x1, x2, x3 = self.get_vx(u)[1]
        U = -self.G * (self.m1 * self.m2 / np.linalg.norm(x1 - x2, axis=-1) \
                     + self.m1 * self.m3 / np.linalg.norm(x1 - x3, axis=-1) \
                     + self.m2 * self.m3 / np.linalg.norm(x2 - x3, axis=-1))
        return U
    
    def compute_kinetic_energy(self, u):
        v1, v2, v3 = self.get_vx(u)[0]
        K = 0.5 * (self.m1 * np.sum(v1**2, axis=-1) \
                 + self.m2 * np.sum(v2**2, axis=-1) \
                 + self.m3 * np.sum(v3**2, axis=-1))
        return K

    def compute_total_momentum(self, u):
        p1, p2, p3 = self.get_pq(u)[0]
        return p1 + p2 + p3
    
    def compute_total_angular_momentum(self, u):
        p1, p2, p3 = self.get_pq(u)[0]
        q1, q2, q3 = self.get_pq(u)[1]
        L1 = np.cross(q1, p1)
        L2 = np.cross(q2, p2)
        L3 = np.cross(q3, p3)
        return L1 + L2 + L3
    
    def compute_r12(self, u):
        x1, x2, _ = self.get_vx(u)[1]
        return np.linalg.norm(x2 - x1, axis=-1)
    
    def compute_r13(self, u):
        x1, _, x3 = self.get_vx(u)[1]
        return np.linalg.norm(x3 - x1, axis=-1)
    
    def compute_r23(self, u):
        _, x2, x3 = self.get_vx(u)[1]
        return np.linalg.norm(x3 - x2, axis=-1)
    
    def compute_energies(self, u):
        return {
            "H": self.compute_hamiltonian(u),
            "U": self.compute_potential_energy(u),
            "K": self.compute_kinetic_energy(u),
        }
    
    def compute_errors(self, u, ref_u):

        # Compute trajectory errors
        pq = self.convert_vx_to_pq(u)
        ref_pq = self.convert_vx_to_pq(ref_u)
        diff_squares = (pq - ref_pq)**2
        abs_traj_err = np.sqrt(np.sum(diff_squares, axis=-1))
        abs_traj_err_p = np.sqrt(np.sum(diff_squares[:, :9], axis=-1))
        abs_traj_err_q = np.sqrt(np.sum(diff_squares[:, 9:], axis=-1))
        rel_traj_err = abs_traj_err / np.linalg.norm(ref_pq, axis=-1)
        rel_traj_err_p = abs_traj_err_p / np.linalg.norm(ref_pq[:, :9], axis=-1)
        rel_traj_err_q = abs_traj_err_q / np.linalg.norm(ref_pq[:, 9:], axis=-1)

        # Compute Hamiltonian errors 
        H = self.compute_hamiltonian(u)
        ref_H = self.compute_hamiltonian(ref_u)
        abs_H_err = np.abs(H - ref_H)
        rel_H_err = abs_H_err / np.abs(ref_H)
        
        return {
            "abs_traj_err": abs_traj_err, 
            "abs_traj_err_p": abs_traj_err_p,
            "abs_traj_err_q": abs_traj_err_q,
            "rel_traj_err": rel_traj_err,
            "rel_traj_err_p": rel_traj_err_p,
            "rel_traj_err_q": rel_traj_err_q,
            "abs_H_err": abs_H_err, 
            "rel_H_err": rel_H_err
        }
    
    @classmethod
    def get_reference_filepaths(cls, category='default'):
        """
        Get reference trajectory filepaths for ThreeBody problem.
        
        Parameters:
        -----------
        category : str, optional
            The category of reference trajectories to retrieve. Options:
            - 'default': Standard three-body problem trajectories
            
        Returns:
        --------
        dict
            Dictionary containing reference trajectory filepaths organized by
            initial condition indices.
        """
        if category == 'default':
            # Convert simple format to standard format
            result = {}
            for ic_idx, filepath in REF_TRAJ_FILEPATHS.items():
                result[ic_idx] = {
                    "filepath": filepath,
                    "dt": REF_TRAJ_DT
                }
            return result
        else:
            raise ValueError(f"Unknown category '{category}'. Available categories: {cls.get_available_reference_categories()}")
    
    @classmethod
    def get_available_reference_categories(cls):
        """
        Get list of available reference trajectory categories for ThreeBody.
        
        Returns:
        --------
        list
            List of available category names for reference trajectories.
        """
        return ['default']


class ThreeBodyDataset(Dataset):

    @classmethod
    def load_from_file(cls, filepath, name="none", m1=100., m2=1., m3=0.001, G=1.):
        df = pd.read_csv(filepath)
        data = States(df.values, ThreeBody(m1, m2, m3, G))
        return cls(data, name)
    
    @classmethod
    def from_vx(cls, vx, name="none", m1=100., m2=1., m3=0.001, G=1.):
        return cls(States(vx, ThreeBody(m1, m2, m3, G)), name)
    

class ThreeBodyTrajectory(Trajectory): 

    @classmethod
    def load_from_file(cls, filepath, dt, m1=100., m2=1., m3=0.001, G=1.):
        df = pd.read_csv(filepath)
        states = States(df.values, ThreeBody(m1, m2, m3, G))
        times = np.arange(0, len(states)) * dt
        return cls(times, states)

    @classmethod
    def from_pq(cls, times, pq, m1=100., m2=1., m3=0.001, G=1.):
        threebody = ThreeBody(m1, m2, m3, G)
        u = threebody.convert_pq_to_vx(pq)
        return cls(times, States(u, threebody))