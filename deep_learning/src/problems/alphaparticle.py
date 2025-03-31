import torch 
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Optional


def sample_shell_box(n_samples: int, bounds: List[List[float]]) -> torch.Tensor:
    """Sample points uniformly within a shell for vx and vy, and within a box for x and y."""
    (r_min, r_max), (x_min, x_max), (y_min, y_max) = bounds
    r = r_min + (r_max - r_min) * torch.rand(n_samples)
    theta = 2 * np.pi * torch.rand(n_samples)
    vx = r * torch.cos(theta)
    vy = r * torch.sin(theta)
    x = x_min + (x_max - x_min) * torch.rand(n_samples)
    y = y_min + (y_max - y_min) * torch.rand(n_samples)    
    return torch.stack([vx, vy, x, y], dim=1)  # shape: (n_samples, 4)


class AlphaParticle:
    """The alpha particle dynamics."""

    def __init__(self, **kwargs) -> None:
        """Initialize the system."""
        self.dof = 4
        self.default_params = {
            "epsilon": 0.05,
            "B0": 1.0,
            "a1": 0.3,
            "a2": 0.3,
            "k_x1": 3.0,
            "k_y1": 1.0,
            "k_x2": 1.0,
            "k_y2": 3.0,
        }
        for key, value in kwargs.items():
            if key in self.default_params:
                self.default_params[key] = value
            else:
                raise ValueError(f"Invalid parameter: {key}.")

        self.bounds = [(-1.5, 1.5), (-1.5, 1.5), (1, 3.5), (1, 4.5)]  # bounds for vx, vy, x, y
        self.random_params_bounds = {"epsilon": (0.05, 0.4), "a1": (0.2, 0.4), "a2": (0.2, 0.4)}

    def __repr__(self) -> str:
        params_repr = ", ".join([f"{key}={value}" for key, value in self.default_params.items()])
        return f"AlphaParticle({params_repr})"

    def default_initial_states(self) -> torch.Tensor:
        """Generate initial states."""
        states = [ 
            [np.sqrt(2), 0., 2.5, 3.0],
            [1.0, 1.0, 2.5, 3.0],
            [1.0, -1.0, 2.5, 3.0],
        ]
        return torch.stack([torch.tensor(s) for s in states])  # tensor dtype is torch.float64

    def random_states(self, n_samples: int) -> torch.Tensor:
        """Sample phase space states."""
        # sampled_dimensions = []
        # for (low, high) in self.bounds:
        #     sampled_dimensions.append(low + (high - low) * torch.rand(n_samples))
        # return torch.stack(sampled_dimensions, dim=1)  # shape: (n_samples, 4)
        return sample_shell_box(n_samples, [(np.sqrt(2)-0.3, np.sqrt(2)+0.3), (1, 4.5), (1, 4.5)])
    
    def random_params(self, n_samples: int, param_keys: List[str]) -> Dict[str, torch.Tensor]:
        """Sample parameters."""
        params = {}
        for key in param_keys:
            if key in self.random_params_bounds:
                low, high = self.random_params_bounds[key]
                params[key] = low + (high - low) * torch.rand((n_samples, 1))  # shape: (n_samples, 1)
            else:
                raise ValueError(f"Missing bounds for parameter: {key}.")
        return params

    def _get_params(self, p: Dict[str, torch.Tensor]) -> None:
        """Merge default and override parameters."""
        params = self.default_params.copy()
        if p is not None:
            for key, value in p.items():
                if key in params:
                    params[key] = value
                else:
                    raise ValueError(f"Invalid parameter: {key}.")
        return params

    def compute_B(self, x: torch.Tensor, y: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute magnetic field."""
        params = self._get_params(p)
        return params["B0"] + params["a1"] * torch.cos(params["k_x1"] * x + params["k_y1"] * y) \
            + params["a2"] * torch.cos(params["k_x2"] * x + params["k_y2"] * y)

    def compute_dBdx(self, x: torch.Tensor, y: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute derivative of magnetic field w.r.t. x."""
        params = self._get_params(p)
        return -params["a1"] * params["k_x1"] * torch.sin(params["k_x1"] * x + params["k_y1"] * y) \
            - params["a2"] * params["k_x2"] * torch.sin(params["k_x2"] * x + params["k_y2"] * y)
    
    def compute_dBdy(self, x: torch.Tensor, y: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute derivative of magnetic field w.r.t. y."""
        params = self._get_params(p)
        return -params["a1"] * params["k_y1"] * torch.sin(params["k_x1"] * x + params["k_y1"] * y) \
            - params["a2"] * params["k_y2"] * torch.sin(params["k_x2"] * x + params["k_y2"] * y)
    
    def compute_du(self, u: torch.Tensor, t: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute time derivative of u = (vx,vy,x,y)."""
        vx, vy, x, y = u.chunk(4, dim=-1)
        Bxy = self.compute_B(x, y, p)
        dvxdt = Bxy * vy
        dvydt = -Bxy * vx
        eps = p["epsilon"] if p is not None and "epsilon" in p else self.default_params["epsilon"]
        dxdt = eps * vx
        dydt = eps * vy
        return torch.cat((dvxdt, dvydt, dxdt, dydt), dim=-1)

    def compute_ddu(self, u: torch.Tensor, t: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute second time derivative of u = (vx,vy,x,y)."""
        vx, vy, x, y = u.chunk(4, dim=-1)
        Bxy = self.compute_B(x, y, p)
        Bxy_sq = Bxy**2
        dBdx = self.compute_dBdx(x, y, p)
        dBdy = self.compute_dBdy(x, y, p)
        eps = p["epsilon"] if p is not None and "epsilon" in p else self.default_params["epsilon"]
        dBdt = eps * (dBdx * vx + dBdy * vy)
        d2vxdt2 = dBdt * vy - Bxy_sq * vx
        d2vydt2 = -dBdt * vx - Bxy_sq * vy
        d2xdt2 = eps * Bxy * vy
        d2ydt2 = - eps * Bxy * vx
        return torch.cat((d2vxdt2, d2vydt2, d2xdt2, d2ydt2), dim=-1)

    def transform_to_energy_components(self, u: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Transform original variables to energy-based variables."""
        vx, vy, x, y = u.chunk(4, dim=-1)
        eps = p["epsilon"] if p is not None and "epsilon" in p else self.default_params["epsilon"]
        # sqrt_eps = torch.sqrt(torch.ones_like(vx) * eps)
        # return torch.cat((vx / sqrt_eps, vy / sqrt_eps, x * sqrt_eps, y * sqrt_eps), dim=-1)
        return torch.cat((vx, vy, x/eps, y/eps), dim=-1)

    def transform_to_original_components(self, u: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Transform energy-based variables to canonical variables."""
        vx, vy, x_scaled, y_scaled = u.chunk(4, dim=-1)
        eps = p["epsilon"] if p is not None and "epsilon" in p else self.default_params["epsilon"]
        x = x_scaled * eps
        y = y_scaled * eps
        return torch.cat((vx, vy, x, y), dim=-1)
    
    def nondim_u(self, u: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        return self.transform_to_energy_components(u, p)
    
    def dim_u(self, u_nd: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        return self.transform_to_original_components(u_nd, p)
    
    def nondim_du(self, du: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        return self.transform_to_energy_components(du, p)
    
    def dim_du(self, du_nd: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        return self.transform_to_original_components(du_nd, p)
    
    def compute_quantities(self, u: torch.Tensor, p: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Compute useful quantities accessed by model trainer."""
        return {
            "vx": u[:, 0],
            "vy": u[:, 1],
            "x": u[:, 2],
            "y": u[:, 3],
        }
    
    def compute_errors(self, u: torch.Tensor, u_true: torch.Tensor, p: Optional[Dict[str, torch.Tensor]], 
                       reduction: str = "none") -> Dict[str, torch.Tensor]:
        """Compute errors between predicted and true states."""

        # Compute trajectory errors
        diff_squares = (u - u_true)**2
        abs_traj_errors = diff_squares.sum(dim=-1).sqrt()
        rel_traj_errors = abs_traj_errors / torch.sum(u_true**2, dim=-1).sqrt()
        abs_traj_errors_vxvy = diff_squares[:, :2].sum(dim=-1).sqrt()
        rel_traj_errors_vxvy = abs_traj_errors_vxvy / torch.sum(u_true[:, :2]**2, dim=-1).sqrt()
        abs_traj_errors_xy = diff_squares[:, 2:4].sum(dim=-1).sqrt()
        rel_traj_errors_xy = abs_traj_errors_xy / torch.sum(u_true[:, 2:4]**2, dim=-1).sqrt()

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
            "abs_traj_err_vxvy": reduction_fn(abs_traj_errors_vxvy),
            "rel_traj_err_vxvy": reduction_fn(rel_traj_errors_vxvy),
            "abs_traj_err_xy": reduction_fn(abs_traj_errors_xy),
            "rel_traj_err_xy": reduction_fn(rel_traj_errors_xy),
        }
    
    def plot_energy_profile(self, trajectory: torch.Tensor) -> plt.Figure:
        """Plot energy profile."""
        # trajectory: (traj_len, 3)

        n_grid = np.arange(len(trajectory))
        quantities = self.compute_quantities(trajectory, p=None)
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
        ax.set_ylim(-2, 5)
        ax.set_xlabel("n")
        ax.set_ylabel("energy")
        ax.legend()

        return fig

    def plot_trajectories(self, trajectories: torch.Tensor) -> Dict[str, plt.Figure]:
        """Plot trajectories."""

        figures = {}

        for i, traj in enumerate(trajectories):
            figures[f"traj{i+1}_energy_profile"] = self.plot_energy_profile(traj)

        # xy-plane trajectory plots
        for i, traj in enumerate(trajectories):
            x, y = traj[:, 2], traj[:, 3]
            x, y = x.cpu().numpy(), y.cpu().numpy()

            fig, ax = plt.subplots()
            ax.plot(x, y, "-", lw=1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(1, 3.5)
            ax.set_ylim(1, 4.5)

            fig.tight_layout()
            figures[f"traj{i+1}_xy"] = fig

        return figures

