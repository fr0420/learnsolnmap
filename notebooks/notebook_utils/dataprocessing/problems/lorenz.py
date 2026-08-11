import os
import numpy as np 
import pandas as pd
from ..core import States, Trajectory, Dataset
from .base import BaseProblem


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


class Lorenz(BaseProblem):

    def __init__(self, sigma=10., rho=28., beta=8/3.):
        super().__init__(dof=3)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
    def __repr__(self):
        return f"Lorenz(sigma={self.sigma}, rho={self.rho}, beta={self.beta})"

    def __eq__(self, other):
        if isinstance(other, Lorenz):
            return self.sigma == other.sigma and self.rho == other.rho and self.beta == other.beta
        return NotImplemented

    def get_pq(self, u):
        """Convert state vector to position-momentum coordinates."""
        return u[:, :2], u[:, 2:]  # (x,y), (z)
    
    def get_vx(self, u):
        """Convert state vector to velocity-position coordinates."""
        return u[:, :2], u[:, 2:]  # (dx/dt, dy/dt), (x, y, z)
    
    def convert_vx_to_pq(self, u):
        """Convert velocity-position to position-momentum coordinates."""
        # For Lorenz, we treat (dx/dt, dy/dt) as momenta and (x, y, z) as positions
        return u
    
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
    
    @classmethod
    def get_reference_filepaths(cls, category='default'):
        """
        Get reference trajectory filepaths for Lorenz problem.
        
        Parameters:
        -----------
        category : str, optional
            The category of reference trajectories to retrieve. Options:
            - 'default': Standard Lorenz system trajectories
            
        Returns:
        --------
        dict
            Dictionary containing reference trajectory filepaths organized by
            initial condition indices.
        """
        if category == 'default':
            return REF_TRAJ_FILEPATHS
        else:
            raise ValueError(f"Unknown category '{category}'. Available categories: {cls.get_available_reference_categories()}")
    
    @classmethod
    def get_available_reference_categories(cls):
        """
        Get list of available reference trajectory categories for Lorenz.
        
        Returns:
        --------
        list
            List of available category names for reference trajectories.
        """
        return ['default']


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
