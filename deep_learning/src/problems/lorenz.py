import torch 
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict


class Lorenz:
    """The Lorenz system."""

    def __init__(self, sigma=10., rho=28., beta=8/3.) -> None:
        """Initialize the system."""
        self.dof = 3
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.bounds = [(-20, 20), (-20, 20), (0, 40)]  # bounds for x, y, z

    def __repr__(self) -> str:
        return f"Lorenz(sigma={self.sigma}, rho={self.rho}, beta={self.beta})"

    def default_initial_states(self) -> torch.Tensor:
        """Generate initial states."""
        states = [ 
            [1., 0., 0.],
        ]
        return torch.stack([torch.tensor(s) for s in states])  # tensor dtype is torch.float64
    
    def random_states(self, n_samples: int) -> torch.Tensor:
        """Sample states uniformly within a bounded box in the phase space."""
        sampled_dimensions = []
        for (low, high) in self.bounds:
            range = high - low
            sampled_dimensions.append(low + range * torch.rand(n_samples))
        return torch.stack(sampled_dimensions, dim=1)  # shape: (n_samples, 3)

    def compute_du(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute time derivative of u = (x,y,z)."""
        x, y, z = u.chunk(3, dim=-1)
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return torch.cat((dxdt, dydt, dzdt), dim=-1)

    def compute_quantities(self, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute useful quantities accessed by model trainer."""
        return {
            "x": u[:, 0],
            "y": u[:, 1],
            "z": u[:, 2],
        }
    
    def compute_errors(self, u: torch.Tensor, u_true: torch.Tensor, reduction: str = "none") -> Dict[str, torch.Tensor]:
        """Compute errors between predicted and true states."""

        # Compute trajectory errors
        diff_squares = (u - u_true)**2
        abs_traj_errors = diff_squares.sum(dim=-1).sqrt()
        rel_traj_errors = abs_traj_errors / torch.sum(u_true**2, dim=-1).sqrt()

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
        }
    
    def plot_energy_profile(self, trajectory: torch.Tensor) -> plt.Figure:
        """Plot energy profile."""
        # trajectory: (traj_len, 3)

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
        # ax.set_ylim(min_val - 0.1*val_range, max_val + 0.1*val_range)
        ax.set_ylim(-30, 30)
        ax.set_xlabel("n")
        ax.set_ylabel("energy")
        ax.legend()

        return fig

    def plot_trajectories(self, trajectories: torch.Tensor) -> Dict[str, plt.Figure]:
        """Plot trajectories."""

        figures = {}

        for i, traj in enumerate(trajectories):
            figures[f"traj{i+1}_energy_profile"] = self.plot_energy_profile(traj)

        # xy-plane trajectory plot for traj 1
        traj = trajectories[0]
        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
        x, y, z = x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy()
        
        fig, ax = plt.subplots()
        ax.plot(x, y, "-", lw=1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)

        fig.tight_layout()
        figures["traj1_xy"] = fig

        return figures

