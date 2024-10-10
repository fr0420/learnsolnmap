import numpy as np 
import pandas as pd
from .utils import States, Trajectory, Dataset


OMEGA_50_REF_TRAJ_DT = 0.00390625
OMEGA_50_REF_TRAJ_FILEPATHS = {
    1: "/workspace/projects_rui/learnsolnmap/out/fpu/omega=50/202410071715/ref/k=0/u.csv",
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

    def compute_stiff_spring_energies(self, u):
        p, q = np.split(u, 2, axis=-1)
        x1 = (q[..., 1::2] - q[..., ::2]) / np.sqrt(2.)
        y1 = (p[..., 1::2] - p[..., ::2]) / np.sqrt(2.)
        I = 0.5 * y1**2 + 0.5 * self.omega**2 * x1**2
        I_total = np.sum(I, axis=-1)
        return np.concatenate((I, I_total[:, None]), axis=-1)

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
    

class FPUDataset(Dataset):

    @classmethod
    def load_from_file(cls, filepath, name="none", omega=50.):
        df = pd.read_csv(filepath)
        data = States(df.values, FPU(omega))
        return cls(data, name)
    

class FPUTrajectory(Trajectory): 

    @classmethod
    def load_from_file(cls, filepath, dt, omega=50.):
        df = pd.read_csv(filepath)
        states = States(df.values, FPU(omega))
        times = np.arange(0, len(states)) * dt
        return cls(times, states)
