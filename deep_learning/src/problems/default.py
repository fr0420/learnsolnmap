import numpy as np
import matplotlib.pyplot as plt
import torch

from typing import Dict, List
from torch import Tensor
from matplotlib.figure import Figure


class SeparableHamiltonianSystem:
    """Separable Hamiltonian system."""

    def __init___(self):
        self.bounds = None  # list of tuples [(low1, high1), (low2, high2), ...]
    
    def default_initial_states(self):
        """Generate initial states."""
        pass

    def compute_Hamiltonian(self, u, p):
        """Compute total energy / Hamiltonian."""
        v, x = u.chunk(2, dim=-1)
        return self.compute_U(x, p) + self.compute_K(v, p)
    
    def compute_Lagrangian(self, u, p):
        """Compute Lagrangian."""
        v, x = u.chunk(2, dim=-1)
        return self.compute_K(v, p) - self.compute_U(x, p)
    
    def compute_U(self, x, p):
        """Compute potential energy."""
        pass

    def compute_K(self, v, p):
        """Compute kinetic energy."""
        pass

    def compute_ddx(self, x, t, p):
        """Compute second derivative of x with respect to time (force/mass)."""
        pass

    def compute_du(self, u, t, p):
        """Compute time derivative of u."""
        v, x = u.chunk(2, dim=-1)
        dvdt = self.compute_ddx(x, t, p)
        dxdt = v 
        return torch.cat((dvdt, dxdt), dim=-1)
    
    def transform_to_energy_components(self, u):
        """Transform canonical variables to variables whose squared l2-norm = Hamiltonian."""
        pass
    
    def compute_quantities(self, u):
        """Compute useful quantities accessed by model trainer."""
        pass 
    
    def compute_errors(self, u, u_true):
        """Compute errors."""
        pass
    
    def plot_energy_profile(self, trajectory: Tensor) -> Figure:
        """Plot energy profile."""
        # trajectory: (traj_len, 2 * dof)

        n_grid = np.arange(len(trajectory))
        quantities = self.compute_quantities(trajectory)
        quantities = {key: quantities[key].cpu().numpy() for key in quantities.keys()}
        init_vals = [vals[0] for _, vals in quantities.items()]
        min_val = min(init_vals)
        max_val = max(init_vals)
        val_range = max_val - min_val

        fig, ax = plt.subplots()
        for key in quantities.keys():
            ax.plot(n_grid, quantities[key], lw=2, label=key)
        ax.set_xlim(n_grid[0], n_grid[-1])
        ax.set_ylim(min_val - 0.1*val_range, max_val + 0.1*val_range)
        ax.set_xlabel("n")
        ax.set_ylabel("energy")
        ax.legend()

        return fig

    def plot_trajectories(self, trajectories: Tensor) -> Dict[str, Figure]:
        """Plot trajectories."""
        # trajectories: (n_traj, traj_len, 2 * dof)

        figures = {}

        for i, traj in enumerate(trajectories):
            figures[f"traj{i+1}_energy_profile"] = self.plot_energy_profile(traj)

        return figures

    def random_states(self, n_samples: int) -> Tensor:
        """Sample states uniformly within a bounded box in the phase space."""

        sampled_dimensions = []

        for (low, high) in self.bounds:
            range = high - low
            sampled_dimensions.append(low + range * torch.rand(n_samples))

        return torch.stack(sampled_dimensions, dim=1)  # shape: (n_samples, 2*dof)

    def random_params(self, n_samples: int, param_keys: List[str]) -> Dict[str, Tensor]:
        """Sample parameters uniformly within a bounded box in the parameter space."""

        params = {}
        for key in param_keys:
            if key in self.random_params_bounds:
                low, high = self.random_params_bounds[key]
                params[key] = low + (high - low) * torch.rand((n_samples, 1))  # shape: (n_samples, 1)
            else:
                raise ValueError(f"Missing bounds for parameter: {key}.")

        return params