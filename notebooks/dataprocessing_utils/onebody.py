import os
import numpy as np 
import pandas as pd
from .utils import States, Trajectory, Dataset


ECC_VAL = 0.5

REF_TRAJ_FILEPATHS = {
    ECC_VAL: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/1body/ecc=5e-1/1/202503192208/ref/u.csv",
            "dt": 0.01,
        },
    },
}


class OneBody:

    def __init__(self, ecc=0.5):
        self.ecc = ecc
        self.dof = 2
    
    def __eq__(self, other):
        if isinstance(other, OneBody):
            return self.ecc == other.ecc
        return NotImplemented
    
    def __repr__(self):
        return f"OneBody(ecc={self.ecc})"
    
    def get_pq(self, u):
        p, q = np.split(u, 2, axis=-1)
        return p, q
    
    def get_vx(self, u):
        v, x = np.split(u, 2, axis=-1)
        return v, x

    def compute_hamiltonian(self, u):
        return self.compute_potential_energy(u) + self.compute_kinetic_energy(u)
    
    def compute_potential_energy(self, u):
        _, x = self.get_vx(u)
        return -1 / np.linalg.norm(x, axis=-1)
    
    def compute_kinetic_energy(self, u):
        v, _ = self.get_vx(u)
        return 0.5 * np.sum(v**2, axis=-1)
    
    def compute_du(self, u):
        v, x = self.get_vx(u)
        dv = self.compute_ddx(u)
        dx = v
        return np.concatenate((dv, dx), axis=-1)

    def compute_ddx(self, u):
        _, x = self.get_vx(u)
        r = np.linalg.norm(x, axis=-1, keepdims=True)
        return -x / r**3

    def compute_energies(self, u):
        return {
            "H": self.compute_hamiltonian(u),
            "U": self.compute_potential_energy(u),
            "K": self.compute_kinetic_energy(u),
        }
    
    def compute_errors(self, u, ref_u):

        # Compute trajectory errors
        diff_squares = (u - ref_u)**2
        abs_traj_err = np.sqrt(np.sum(diff_squares, axis=-1))
        abs_traj_err_p = np.sqrt(np.sum(diff_squares[:, :2], axis=-1))
        abs_traj_err_q = np.sqrt(np.sum(diff_squares[:, 2:], axis=-1))
        rel_traj_err = abs_traj_err / np.linalg.norm(ref_u, axis=-1)
        rel_traj_err_p = abs_traj_err_p / np.linalg.norm(ref_u[:, :2], axis=-1)
        rel_traj_err_q = abs_traj_err_q / np.linalg.norm(ref_u[:, 2:], axis=-1)

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
            "rel_H_err": rel_H_err,
        }
    

class OneBodyDataset(Dataset):

    @classmethod
    def load_from_file(cls, filepath, name="none", **kwargs):
        df = pd.read_csv(filepath)
        data = States(df.values, OneBody(**kwargs))
        return cls(data, name)
    
    @classmethod
    def from_vx(cls, vx, name="none", **kwargs):
        return cls(States(vx, OneBody(**kwargs)), name)
    

class OneBodyTrajectory(Trajectory): 

    @classmethod
    def load_from_file(cls, filepath, dt=None, **kwargs):
        df = pd.read_csv(filepath)
        states = States(df.values, OneBody(**kwargs))
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
        return cls(times, States(vx, OneBody(**kwargs)))
