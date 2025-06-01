import os
import numpy as np 
import pandas as pd
from .utils import States, Trajectory, Dataset


REF_TRAJ_FILEPATHS = {
    1: {
        "filepath": "/workspace/projects_rui/learnsolnmap/out/lorenz/1/202501271648/u.csv",
        "dt": 1e-3,
    },
    2: {
        "filepath": "/workspace/projects_rui/learnsolnmap/out/lorenz/2/202501301137/u.csv",
        "dt": 1e-3,
    }
}


class Lorenz:

    def __init__(self, sigma=10., rho=28., beta=8/3.):
        self.dof = 3
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
    def __repr__(self):
        return f"Lorenz(sigma={self.sigma}, rho={self.rho}, beta={self.beta})"

    def __eq__(self, other):
        if isinstance(other, Lorenz):
            return self.sigma == other.sigma and self.rho == other.rho and self.beta == other.beta
        return NotImplemented

    def get_xyz(self, u):
        return u[:, 0], u[:, 1], u[:, 2]
    
    def compute_du(self, u):
        x, y, z = u[:, 0], u[:, 1], u[:, 2]
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return np.concatenate((dxdt, dydt, dzdt), axis=-1)

    def compute_jacobian(self, u):
        """Compute the Jacobian of the vector field (dp/dt, dq/dt) with respect to (p, q)."""
        return NotImplemented

    def compute_errors(self, u, ref_u):

        # Compute trajectory errors
        diff_squares = (u - ref_u)**2
        abs_traj_err = np.sqrt(np.sum(diff_squares, axis=-1))
        rel_traj_err = abs_traj_err / np.linalg.norm(ref_u, axis=-1)

        return {
            "abs_traj_err": abs_traj_err, 
            "rel_traj_err": rel_traj_err,
        }
    

class LorenzDataset(Dataset):

    @classmethod
    def load_from_file(cls, filepath, name="none", sigma=10., rho=28., beta=8/3.):
        df = pd.read_csv(filepath)
        data = States(df.values, Lorenz(sigma, rho, beta))
        return cls(data, name)
    
    @classmethod
    def from_points(cls, points, name="none", sigma=10., rho=28., beta=8/3.):
        return cls(States(points, Lorenz(sigma, rho, beta)), name)
    

class LorenzTrajectory(Trajectory): 

    @classmethod
    def load_from_file(cls, filepath, dt=None, sigma=10., rho=28., beta=8/3.):
        df = pd.read_csv(filepath)
        states = States(df.values, Lorenz(sigma, rho, beta))
        times_filepath = filepath.replace("u.csv", "t.csv")
        if os.path.exists(times_filepath):
            times = pd.read_csv(times_filepath).values.flatten()
        elif dt is not None:
            times = np.arange(0, len(states)) * dt
        else:
            raise ValueError("Either provide a valid dt or a times file.")
        return cls(times, states)

    @classmethod
    def from_u(cls, times, u, sigma=10., rho=28., beta=8/3.):
        lorenz = Lorenz(sigma, rho, beta)
        return cls(times, States(u, lorenz))
