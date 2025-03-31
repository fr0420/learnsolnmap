import torch 
import numpy as np
import matplotlib.pyplot as plt

from problems.default import SeparableHamiltonianSystem
from typing import Dict


def sample_box(bounds, num_points):
    """
    Sample points uniformly within a high-dimensional bounded box.

    Parameters:
        bounds (list of tuples): Each tuple represents the (min, max) bounds of each dimension.
        num_points (int): The number of points to generate.

    Returns:
        np.ndarray: An array of shape (num_points, len(dimensions)) containing the sampled points.
    """
    # Check that all dimensions have a min and max
    assert all(len(d) == 2 for d in bounds), "Each dimension must have a min and max bound"

    # Create an empty array to store the points
    points = np.empty((num_points, len(bounds)))

    # For each dimension, generate random numbers within the given bounds
    for i, (low, high) in enumerate(bounds):
        points[:, i] = np.random.uniform(low, high, num_points)

    return points


def sample_shell(radius_bounds, num_points):
    """
    Sample points uniformly within a high-dimensional shell.

    Parameters: 
        radius_bounds (tuple): The (min, max) bounds of the shell radius.
        num_points (int): The number of points to generate.
    
    Returns:
        np.ndarray: An array of shape (num_points, len(dimensions)) containing the sampled points.
    """
    # Check that all dimensions have a min and max
    assert all(len(d) == 2 for d in radius_bounds), "Each dimension must have a min and max bound"

    # Create an empty array to store the points
    points = np.empty((num_points, len(radius_bounds)))

    # Generate random directions 
    directions = np.random.randn(num_points, len(radius_bounds))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    # Generate random radii
    radii = sample_box(radius_bounds, num_points)
    
    # Scale the directions by the radii
    points = radii * directions

    return points


def sample_microcanonical_ensemble(body1_position_bounds, body2_position_radius_bounds, 
                                   body3_local_position_bounds, body3_velocity_bounds,
                                   num_body1_points, num_body2_points_per_body1, num_body3_points_per_body2): 
    M1, M2, M3 = 100., 1., 0.001
    body1_positions = sample_box(body1_position_bounds, num_body1_points)
    body1_velocities = np.zeros_like(body1_positions)
    body1_velocities[:, 0] = -body1_positions[:, 1] / (M1+M2)
    body1_velocities[:, 1] = body1_positions[:, 0] / (M1+M2)
    body1_positions = np.repeat(body1_positions, num_body2_points_per_body1, axis=0)
    body1_velocities = np.repeat(body1_velocities, num_body2_points_per_body1, axis=0)

    body2_positions = sample_shell(body2_position_radius_bounds, num_body1_points * num_body2_points_per_body1)
    body2_velocities = np.zeros_like(body2_positions)
    body2_velocities[:, 0] = -body2_positions[:, 1] / (M1+M2)
    body2_velocities[:, 1] = body2_positions[:, 0] / (M1+M2)

    body1_positions = np.repeat(body1_positions, num_body3_points_per_body2, axis=0)
    body1_velocities = np.repeat(body1_velocities, num_body3_points_per_body2, axis=0)
    body2_positions = np.repeat(body2_positions, num_body3_points_per_body2, axis=0)
    body2_velocities = np.repeat(body2_velocities, num_body3_points_per_body2, axis=0)

    body3_positions = sample_box(body3_local_position_bounds, num_body1_points * num_body2_points_per_body1 * num_body3_points_per_body2)
    body3_positions += body2_positions
    body3_velocities = sample_box(body3_velocity_bounds, num_body1_points * num_body2_points_per_body1 * num_body3_points_per_body2)
    
    points = np.concatenate(
        (body1_velocities, body2_velocities, body3_velocities, body1_positions, body2_positions, body3_positions), 
        axis=-1)
    return points


def sample_microcanonical_ensemble2(body1_position_bounds, body2_position_radius_bounds, 
                                   body3_local_position_bounds, body3_velocity_bounds,
                                   num_points): 
    M1, M2, M3 = 100., 1., 0.001
    body1_positions = sample_box(body1_position_bounds, num_points)
    body1_velocities = np.zeros_like(body1_positions)
    body1_velocities[:, 0] = -body1_positions[:, 1] / (M1+M2)
    body1_velocities[:, 1] = body1_positions[:, 0] / (M1+M2)

    body2_positions = sample_shell(body2_position_radius_bounds, num_points)
    body2_velocities = np.zeros_like(body2_positions)
    body2_velocities[:, 0] = -body2_positions[:, 1] / (M1+M2)
    body2_velocities[:, 1] = body2_positions[:, 0] / (M1+M2)

    body3_positions = sample_box(body3_local_position_bounds, num_points)
    body3_positions += body2_positions
    body3_velocities = sample_box(body3_velocity_bounds, num_points)
    
    points = np.concatenate(
        (body1_velocities, body2_velocities, body3_velocities, body1_positions, body2_positions, body3_positions), 
        axis=-1)
    return points


def sample_microcanonical_ensemble3(body1_position_radius_bounds, body2_position_radius_bounds, 
                                   body3_local_position_bounds, body3_velocity_bounds,
                                   num_points): 
    M1, M2, M3 = 100., 1., 0.001
    body1_positions = sample_shell(body1_position_radius_bounds, num_points)
    body1_velocities = np.zeros_like(body1_positions)
    body1_velocities[:, 0] = -body1_positions[:, 1] / (M1+M2)
    body1_velocities[:, 1] = body1_positions[:, 0] / (M1+M2)

    body2_positions = sample_shell(body2_position_radius_bounds, num_points)
    body2_velocities = np.zeros_like(body2_positions)
    body2_velocities[:, 0] = -body2_positions[:, 1] / (M1+M2)
    body2_velocities[:, 1] = body2_positions[:, 0] / (M1+M2)

    body3_positions = sample_box(body3_local_position_bounds, num_points)
    body3_positions += body2_positions
    body3_velocities = sample_box(body3_velocity_bounds, num_points)
    
    points = np.concatenate(
        (body1_velocities, body2_velocities, body3_velocities, body1_positions, body2_positions, body3_positions), 
        axis=-1)
    return points

def rotate2d(p, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],[s, c]])
    R = np.transpose(R)
    return p.dot(R)
    
def sample_random_equilateral(num_points, nu=2e-1, min_radius=0.9, max_radius=1.2):
    
    q1 = sample_shell([(min_radius, max_radius), (min_radius, max_radius)], num_points)
    r = np.sqrt(np.sum((q1**2), axis=1))
    q2 = rotate2d(q1, theta=2*np.pi/3)
    q3 = rotate2d(q2, theta=2*np.pi/3)
    
    # velocity that yields a circular orbit
    v1 = rotate2d(q1, theta=np.pi/2)
    v1 = v1 / np.tile(np.expand_dims(r**1.5, axis=1), (1, 2))
    v1 = v1 * np.sqrt(np.sin(np.pi/3)/(2*np.cos(np.pi/6)**2)) # scale factor to get circular trajectories
    v2 = rotate2d(v1, theta=2*np.pi/3)
    v3 = rotate2d(v2, theta=2*np.pi/3)
    
    # make the circular orbits slightly chaotic
    v1 *= 1 + nu*(2*np.random.rand(2) - 1)
    v2 *= 1 + nu*(2*np.random.rand(2) - 1)
    v3 *= 1 + nu*(2*np.random.rand(2) - 1)
    
    q = np.zeros([num_points, 6])
    p = np.zeros([num_points, 6])
    
    q[:, :2] = q1
    q[:, 2:4] = q2
    q[:, 4:] = q3
    p[:, :2] = v1
    p[:, 2:4] = v2
    p[:, 4:] = v3
  
    return np.hstack([p, q])



class ThreeBody2D(SeparableHamiltonianSystem):
    """Three-body system in 2D."""

    def __init__(self, m1=100.0, m2=1.0, m3=0.001, G=1.0) -> None:
        """Initialize the system."""
        super().__init__()
        
        # System parameters
        self.dof = 6
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.G = G
        self.bounds = [
            (-0.01, 0.01), (-0.01, 0.01), 
            (-1.0, 1.0), (-1.0, 1.0),
            (-2.0, 2.0), (-2.0, 2.0),
            (-1.0, 1.0), (-1.0, 1.0),
            (-104, 100), (-102, 102),
            (-113, 109), (-111, 111),
        ]  # bounds for v, x
        
        # Characteristic length scales and time scales for nondimensionalization
        # char_len = (1, 100, 100), char_time = (100, 100, 100) heuristically chosen
        self.char_len1 = 100.
        self.char_len2 = 100.
        self.char_len3 = 100.
        self.char_time1 = 100.
        self.char_time2 = 100.
        self.char_time3 = 100. 
        self.char_vel1 = self.char_len1 / self.char_time1  # O(1e-2)
        self.char_vel2 = self.char_len2 / self.char_time2  # O(1)
        self.char_vel3 = self.char_len3 / self.char_time3  # O(1)
        self.char_acc1 = self.char_vel1 / self.char_time1  # O(1e-4)
        self.char_acc2 = self.char_vel2 / self.char_time2  # O(1e-2)
        self.char_acc3 = self.char_vel3 / self.char_time3  # O(1e-2)

    def __repr__(self) -> str:
        return f"ThreeBody2D(m1={self.m1}, m2={self.m2}, m3={self.m3}, G={self.G})"

    def default_initial_states(self) -> torch.Tensor:
        """Generate initial states."""
        x = np.zeros((6,))
        v = np.zeros((6,))
        # x[0:2] =   [-1.00102,      0.,  ]
        # x[2:4] =   [100.,    0.,   ]
        # x[4:6] =   [102.,    0.,   ]
        # v[0:2] =   [0.,     -0.010001,    ]
        # v[2:4] =   [0.,     1.,  ]
        # v[4:6] =   [0.,     0.1, ]
        
        # equal mass system
        x[0:2] =   [1.0,      0.,  ]
        x[2:4] =   [-0.5,    np.sqrt(3)/2,   ]
        x[4:6] =   [-0.5,    -np.sqrt(3)/2,   ]
        v[0:2] =   [0.,     1.0,    ]
        v[2:4] =   [-np.sqrt(3)/2,     -0.5,  ]
        v[4:6] =   [np.sqrt(3)/2,    -0.5, ]
        v *= 3**(-0.25)

        states = [ np.concatenate([v, x]),]

        return torch.stack([torch.tensor(s) for s in states])  # tensor dtype is torch.float64
    
    def sample_box_filtered(self, num_points, bounds=None, filter_func=None):

        if bounds is None:
            bounds = [(-1.5, 1.5)] * 12  # Default bounds for v1, v2, v3, x1, x2, x3
        
        if filter_func is None:
            H0 = -0.86
            tol = 1e-1
            filter_func = lambda u: torch.abs(self.compute_Hamiltonian(u) - H0) < tol
        
        # Sample points uniformly within the box
        points = sample_box(bounds, num_points)
        points = torch.tensor(points, dtype=torch.float64)  # shape: (num_points, 2*dof)

        # Filter the points using the given function
        mask = filter_func(points)
        points = points[mask]
        
        return points

    def sample_box_filtered2(self, num_points, bounds=None, filter_func=None, noise=0.1):

        if bounds is None:
            bounds = [(-1.5, 1.5)] * 12  # Default bounds for v1, v2, v3, x1, x2, x3
        
        if filter_func is None:
            H0 = -0.866
            L0 = 2.28
            filter_func = lambda u: (torch.abs(self.compute_Hamiltonian(u) - H0) < noise) & \
                (torch.abs(self.compute_Lz(u) - L0) < noise)
    
        # Sample points uniformly within the box
        points = sample_box(bounds, num_points)

        # Adjust body 3 momentum to ensure zero total momentum
        v1, v2, v3 = points[:, :2], points[:, 2:4], points[:, 4:6]
        v3 = -(self.m1*v1 + self.m2*v2 + noise * sample_box([(-1, 1), (-1, 1)], num_points)) / self.m3
        points[:, 4:6] = v3
        points = torch.tensor(points, dtype=torch.float64)  # shape: (num_points, 2*dof)

        # Filter the points using the given function
        mask = filter_func(points)
        points = points[mask]

        return points

    def random_states(self, n_samples: int) -> torch.Tensor:
        """Sample states uniformly from a given distribution in the phase space."""

        # points = sample_microcanonical_ensemble(
        #     body1_position_bounds=[(-1, 1), (-1, 1)],
        #     body2_position_radius_bounds=[(100-10, 100+10), (100-10, 100+10)],
        #     body3_local_position_bounds=[(-10, 10), (-10, 10)],
        #     body3_velocity_bounds=[(-2, 2), (-2, 2)],
        #     num_body1_points=n_samples//100,
        #     num_body2_points_per_body1=10,
        #     num_body3_points_per_body2=10,
        # )

        # points = sample_microcanonical_ensemble2(
        #     body1_position_bounds=[(-1, 1), (-1, 1)],
        #     body2_position_radius_bounds=[(100-10, 100+10), (100-10, 100+10)],
        #     body3_local_position_bounds=[(-10, 10), (-10, 10)],
        #     body3_velocity_bounds=[(-2, 2), (-2, 2)],
        #     num_points=n_samples,
        # )

        # points = sample_microcanonical_ensemble3(
        #     body1_position_radius_bounds=[(1-0.1, 1+0.1), (1-0.1, 1+0.1)],
        #     body2_position_radius_bounds=[(100-10, 100+10), (100-10, 100+10)],
        #     body3_local_position_bounds=[(-10, 10), (-10, 10)],
        #     body3_velocity_bounds=[(-2.5, 2.5), (-2.5, 2.5)],
        #     num_points=n_samples,
        # )
        
        # points = sample_random_equilateral(n_samples, nu=2e-1, min_radius=0.9, max_radius=1.2)

        # return torch.tensor(points, dtype=torch.float64)  # shape: (n_samples, 2*dof)

        return self.sample_box_filtered2(n_samples)
    
    def compute_U(self, x: torch.Tensor) -> torch.Tensor:
        """Compute potential energy."""
        x1, x2, x3 = x[..., 0:2], x[..., 2:4], x[..., 4:6]
        U = -self.G * (self.m1 * self.m2 / torch.norm(x1 - x2, dim=-1) \
                     + self.m1 * self.m3 / torch.norm(x1 - x3, dim=-1) \
                     + self.m2 * self.m3 / torch.norm(x2 - x3, dim=-1))
        return U
    
    def compute_K(self, v: torch.Tensor) -> torch.Tensor:
        """Compute kinetic energy."""
        v1, v2, v3 = v[..., 0:2], v[..., 2:4], v[..., 4:6]
        K = 0.5 * (self.m1 * torch.sum(v1**2, dim=-1) \
                 + self.m2 * torch.sum(v2**2, dim=-1) \
                 + self.m3 * torch.sum(v3**2, dim=-1))
        return K

    def compute_Lz(self, u: torch.Tensor) -> torch.Tensor:
        """Compute total angular momentum."""
        v, x = u.chunk(2, dim=-1)
        x1, x2, x3 = x[..., 0:2], x[..., 2:4], x[..., 4:6]
        v1, v2, v3 = v[..., 0:2], v[..., 2:4], v[..., 4:6]
        Lz = self.m1 * (x1[..., 0]*v1[..., 1] - x1[..., 1]*v1[..., 0]) \
           + self.m2 * (x2[..., 0]*v2[..., 1] - x2[..., 1]*v2[..., 0]) \
           + self.m3 * (x3[..., 0]*v3[..., 1] - x3[..., 1]*v3[..., 0])
        return Lz

    def compute_P(self, u: torch.Tensor) -> torch.Tensor:
        """Compute total linear momentum."""
        v, _ = u.chunk(2, dim=-1)
        v1, v2, v3 = v[..., 0:2], v[..., 2:4], v[..., 4:6]
        P = self.m1 * v1 + self.m2 * v2 + self.m3 * v3
        return P

    def compute_ddx(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute second derivative of x with respect to time (force/mass)."""
        x1, x2, x3 = x[..., 0:2], x[..., 2:4], x[..., 4:6]
        r12 = torch.norm(x2 - x1, dim=-1, keepdim=True)
        r13 = torch.norm(x3 - x1, dim=-1, keepdim=True)
        r23 = torch.norm(x3 - x2, dim=-1, keepdim=True)
        ddx1 = - self.G * (self.m2 * (x1 - x2) / r12**3 + self.m3 * (x1 - x3) / r13**3)
        ddx2 = - self.G * (self.m1 * (x2 - x1) / r12**3 + self.m3 * (x2 - x3) / r23**3)
        ddx3 = - self.G * (self.m1 * (x3 - x1) / r13**3 + self.m2 * (x3 - x2) / r23**3)
        return torch.cat([ddx1, ddx2, ddx3], dim=-1)

    def transform_to_energy_components(self, u_nd: torch.Tensor) -> torch.Tensor:
        """Transform canonical variables to variables whose squared l2-norm = Hamiltonian."""
        p, q = u_nd.chunk(2, dim=-1)
        p1, p2, p3 = p[..., 0:2], p[..., 2:4], p[..., 4:6]
        q1, q2, q3 = q[..., 0:2], q[..., 2:4], q[..., 4:6]
        r12 = torch.norm(q2 - q1, dim=-1, keepdim=True)
        r13 = torch.norm(q3 - q1, dim=-1, keepdim=True)
        r23 = torch.norm(q3 - q2, dim=-1, keepdim=True)
        return torch.cat((
            p1 / (2*self.m1)**0.5, p2 / (2*self.m2)**0.5, p3 / (2*self.m3)**0.5, 
            torch.sqrt(self.m1*self.m2/r12), torch.sqrt(self.m1*self.m3/r13), torch.sqrt(self.m2*self.m3/r23)), dim=-1)

    def transform_to_energy_components_anchored(self, u_nd: torch.Tensor) -> torch.Tensor:
        """Transform canonical variables to variables whose squared l2-norm = Hamiltonian."""
        p, q = u_nd.chunk(2, dim=-1)
        p1, p2, p3 = p[..., 0:2], p[..., 2:4], p[..., 4:6]
        q1, q2, q3 = q[..., 0:2], q[..., 2:4], q[..., 4:6]
        q12 = q2 - q1
        q13 = q3 - q1
        q23 = q3 - q2
        # r12 = torch.norm(q2 - q1, dim=-1, keepdim=True)
        # r13 = torch.norm(q3 - q1, dim=-1, keepdim=True)
        # r23 = torch.norm(q3 - q2, dim=-1, keepdim=True)
        return torch.cat((
            p1 / (2*self.m1)**0.5, p2 / (2*self.m2)**0.5, p3 / (2*self.m3)**0.5, 
            # torch.sqrt(self.m1*self.m2/r12), torch.sqrt(self.m1*self.m3/r13), torch.sqrt(self.m2*self.m3/r23),
            q1, q2, q3, q12, q13, q23), dim=-1)

    def compute_quantities(self, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute useful quantities accessed by model trainer."""
        v, x = u.chunk(2, dim=-1)
        return {
            "H": self.compute_Hamiltonian(u),
            "U": self.compute_U(x),
            "K": self.compute_K(v),
            "Lz": self.compute_Lz(u),
        }
    
    def vx_to_pq(self, u: torch.Tensor) -> torch.Tensor:
        """Convert v,x to p,q."""
        v, x = u.chunk(2, dim=-1)
        v1, v2, v3 = v[..., 0:2], v[..., 2:4], v[..., 4:6]
        p1 = self.m1 * v1
        p2 = self.m2 * v2
        p3 = self.m3 * v3
        return torch.cat([p1, p2, p3, x], dim=-1)
    
    def compute_errors(self, u: torch.Tensor, u_true: torch.Tensor, reduction: str = "none") -> Dict[str, torch.Tensor]:
        """Compute errors between predicted and true states."""

        # Transform (v,x) to (p,q) for traj error computation
        pq = self.vx_to_pq(u)
        pq_true = self.vx_to_pq(u_true)
        
        # Compute trajectory errors
        diff_squares = (pq - pq_true)**2
        abs_traj_errors = diff_squares.sum(dim=-1).sqrt()
        abs_traj_errors_p = diff_squares[:, :6].sum(dim=-1).sqrt()
        abs_traj_errors_q = diff_squares[:, 6:].sum(dim=-1).sqrt()
        abs_traj_errors_p1 = diff_squares[:, :2].sum(dim=-1).sqrt()
        abs_traj_errors_p2 = diff_squares[:, 2:4].sum(dim=-1).sqrt()
        abs_traj_errors_p3 = diff_squares[:, 4:6].sum(dim=-1).sqrt()
        abs_traj_errors_q1 = diff_squares[:, 6:8].sum(dim=-1).sqrt()
        abs_traj_errors_q2 = diff_squares[:, 8:10].sum(dim=-1).sqrt()
        abs_traj_errors_q3 = diff_squares[:, 10:].sum(dim=-1).sqrt()
        rel_traj_errors = abs_traj_errors / torch.sum(pq_true**2, dim=-1).sqrt()
        rel_traj_errors_p = abs_traj_errors_p / torch.sum(pq_true[:, :6]**2, dim=-1).sqrt()
        rel_traj_errors_q = abs_traj_errors_q / torch.sum(pq_true[:, 6:]**2, dim=-1).sqrt()
        rel_traj_errors_p1 = abs_traj_errors_p1 / torch.sum(pq_true[:, :2]**2, dim=-1).sqrt()
        rel_traj_errors_p2 = abs_traj_errors_p2 / torch.sum(pq_true[:, 2:4]**2, dim=-1).sqrt()
        rel_traj_errors_p3 = abs_traj_errors_p3 / torch.sum(pq_true[:, 4:6]**2, dim=-1).sqrt()
        rel_traj_errors_q1 = abs_traj_errors_q1 / torch.sum(pq_true[:, 6:8]**2, dim=-1).sqrt()
        rel_traj_errors_q2 = abs_traj_errors_q2 / torch.sum(pq_true[:, 8:10]**2, dim=-1).sqrt()
        rel_traj_errors_q3 = abs_traj_errors_q3 / torch.sum(pq_true[:, 10:]**2, dim=-1).sqrt()

        # Compute Hamiltonian errors
        H = self.compute_Hamiltonian(u)
        H_true = self.compute_Hamiltonian(u_true)
        abs_H_errors = torch.abs(H - H_true)
        rel_H_errors = abs_H_errors / torch.abs(H_true)

        # Compute angular momentum errors
        Lz = self.compute_Lz(u)
        Lz_true = self.compute_Lz(u_true)
        abs_Lz_errors = torch.abs(Lz - Lz_true)
        rel_Lz_errors = abs_Lz_errors / torch.abs(Lz_true)

        # Compute linear momentum errors 
        P = self.compute_P(u)
        P_true = self.compute_P(u_true)
        abs_P_errors = torch.sum((P - P_true)**2, dim=-1).sqrt()
        rel_P_errors = abs_P_errors / torch.sum(P_true**2, dim=-1).sqrt()

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
            "abs_traj_err_p": reduction_fn(abs_traj_errors_p),
            "rel_traj_err_p": reduction_fn(rel_traj_errors_p),
            "abs_traj_err_q": reduction_fn(abs_traj_errors_q),
            "rel_traj_err_q": reduction_fn(rel_traj_errors_q),
            "abs_traj_err_p1": reduction_fn(abs_traj_errors_p1),
            "rel_traj_err_p1": reduction_fn(rel_traj_errors_p1),
            "abs_traj_err_p2": reduction_fn(abs_traj_errors_p2),
            "rel_traj_err_p2": reduction_fn(rel_traj_errors_p2),
            "abs_traj_err_p3": reduction_fn(abs_traj_errors_p3),
            "rel_traj_err_p3": reduction_fn(rel_traj_errors_p3),
            "abs_traj_err_q1": reduction_fn(abs_traj_errors_q1),
            "rel_traj_err_q1": reduction_fn(rel_traj_errors_q1),
            "abs_traj_err_q2": reduction_fn(abs_traj_errors_q2),
            "rel_traj_err_q2": reduction_fn(rel_traj_errors_q2),
            "abs_traj_err_q3": reduction_fn(abs_traj_errors_q3),
            "rel_traj_err_q3": reduction_fn(rel_traj_errors_q3),
            "abs_H_err": reduction_fn(abs_H_errors),
            "rel_H_err": reduction_fn(rel_H_errors),
            "abs_Lz_err": reduction_fn(abs_Lz_errors),
            "rel_Lz_err": reduction_fn(rel_Lz_errors),
            "abs_P_err": reduction_fn(abs_P_errors),
            "rel_P_err": reduction_fn(rel_P_errors),
        }
    
    def plot_trajectories(self, trajectories: torch.Tensor) -> Dict[str, plt.Figure]:
        """Plot trajectories."""

        figures = {}

        for i, traj in enumerate(trajectories):
            figures[f"traj{i+1}_energy_profile"] = self.plot_energy_profile(traj)

        # xy-plane trajectory plot for traj 1
        traj = trajectories[0]
        _, x = traj.chunk(2, dim=-1)
        x1, x2, x3 = x[..., 0:2], x[..., 2:4], x[..., 4:6]
        x1, x2, x3 = x1.cpu().numpy(), x2.cpu().numpy(), x3.cpu().numpy()
        axis_labels = [('$q_{1x}$', '$q_{1y}$'), 
                       ('$q_{2x}$', '$q_{2y}$'),
                       ('$q_{3x}$', '$q_{3y}$')]
        
        fig = plt.figure(figsize=(12, 4))
        axes = fig.subplots(1, 3) 
        axes[0].plot(x1[:, 0], x1[:, 1], "-", lw=1)
        axes[1].plot(x2[:, 0], x2[:, 1], "-", lw=1)
        axes[2].plot(x3[:, 0], x3[:, 1], "-", lw=1)
        for i, ax in enumerate(axes):
            ax.set_xlabel(axis_labels[i][0])
            ax.set_ylabel(axis_labels[i][1])
            ax.set_title(f"Body {i+1}")
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
        # axes[0].set_xlim(-1.1, 1.1)
        # axes[0].set_ylim(-1.1, 1.1)
        # axes[1].set_xlim(-1.1e2, 1.1e2)
        # axes[1].set_ylim(-1.1e2, 1.1e2)
        # axes[2].set_xlim(-1.1e2, 1.1e2)
        # axes[2].set_ylim(-1.1e2, 1.1e2)

        fig.tight_layout()
        figures["traj1_xy"] = fig

        return figures

    # def nondim_u(self, u: torch.Tensor) -> torch.Tensor:
    #     """Nondimensionalize the input states."""
    #     # Convert v,x to p,q
    #     v, x = u.chunk(2, dim=-1)
    #     v1, v2, v3 = v[..., 0:2], v[..., 2:4], v[..., 4:6]
    #     p1 = self.m1 * v1
    #     p2 = self.m2 * v2
    #     p3 = self.m3 * v3
    #     u_nd = torch.cat([p1, p2, p3, x], dim=-1)
    #     return u_nd
    
    # def dim_u(self, u_nd: torch.Tensor) -> torch.Tensor:
    #     """Dimensionalize the input states."""
    #     # Convert p,q to v,x
    #     p, q = u_nd.chunk(2, dim=-1)
    #     p1, p2, p3 = p[..., 0:2], p[..., 2:4], p[..., 4:6]
    #     v1 = p1 / self.m1
    #     v2 = p2 / self.m2
    #     v3 = p3 / self.m3
    #     u = torch.cat([v1, v2, v3, q], dim=-1)
    #     return u
    
    # def nondim_du(self, du: torch.Tensor) -> torch.Tensor:
    #     """Nondimensionalize the input state derivatives."""
    #     # Convert dv, dx to dp, dq
    #     dv, dx = du.chunk(2, dim=-1)
    #     dv1, dv2, dv3 = dv[..., 0:2], dv[..., 2:4], dv[..., 4:6]
    #     dp1 = self.m1 * dv1
    #     dp2 = self.m2 * dv2
    #     dp3 = self.m3 * dv3
    #     du_nd = torch.cat([dp1, dp2, dp3, dx], dim=-1)
    #     return du_nd
    
    # def dim_du(self, du_nd: torch.Tensor) -> torch.Tensor:
    #     """Dimensionalize the input state derivatives."""
    #     # Convert dp, dq to dv, dx
    #     dp, dq = du_nd.chunk(2, dim=-1)
    #     dp1, dp2, dp3 = dp[..., 0:2], dp[..., 2:4], dp[..., 4:6]
    #     dv1 = dp1 / self.m1
    #     dv2 = dp2 / self.m2
    #     dv3 = dp3 / self.m3
    #     du = torch.cat([dv1, dv2, dv3, dq], dim=-1)
    #     return du

    def nondim_u(self, u: torch.Tensor) -> torch.Tensor:
        """Nondimensionalize the input states."""
        v, x = u.chunk(2, dim=-1)
        v1, v2, v3 = v[..., 0:2], v[..., 2:4], v[..., 4:6]
        x1, x2, x3 = x[..., 0:2], x[..., 2:4], x[..., 4:6]
        x1_nd = x1 / self.char_len1
        x2_nd = x2 / self.char_len2
        x3_nd = x3 / self.char_len3
        v1_nd = v1 / self.char_vel1
        v2_nd = v2 / self.char_vel2
        v3_nd = v3 / self.char_vel3
        u_nd = torch.cat([v1_nd, v2_nd, v3_nd, x1_nd, x2_nd, x3_nd], dim=-1)
        return u_nd
    
    def dim_u(self, u_nd: torch.Tensor) -> torch.Tensor:
        """Dimensionalize the input states."""
        v_nd, x_nd = u_nd.chunk(2, dim=-1)
        v1_nd, v2_nd, v3_nd = v_nd[..., 0:2], v_nd[..., 2:4], v_nd[..., 4:6]
        x1_nd, x2_nd, x3_nd = x_nd[..., 0:2], x_nd[..., 2:4], x_nd[..., 4:6]
        x1 = x1_nd * self.char_len1
        x2 = x2_nd * self.char_len2
        x3 = x3_nd * self.char_len3
        v1 = v1_nd * self.char_vel1
        v2 = v2_nd * self.char_vel2
        v3 = v3_nd * self.char_vel3
        u = torch.cat([v1, v2, v3, x1, x2, x3], dim=-1)
        return u
    
    def nondim_du(self, du: torch.Tensor) -> torch.Tensor:
        """Nondimensionalize the input state derivatives."""
        dv, dx = du.chunk(2, dim=-1)
        dv1, dv2, dv3 = dv[..., 0:2], dv[..., 2:4], dv[..., 4:6]
        dx1, dx2, dx3 = dx[..., 0:2], dx[..., 2:4], dx[..., 4:6]
        dv1_nd = dv1 / self.char_acc1
        dv2_nd = dv2 / self.char_acc2
        dv3_nd = dv3 / self.char_acc3
        dx1_nd = dx1 / self.char_vel1
        dx2_nd = dx2 / self.char_vel2
        dx3_nd = dx3 / self.char_vel3
        du_nd = torch.cat([dv1_nd, dv2_nd, dv3_nd, dx1_nd, dx2_nd, dx3_nd], dim=-1)
        return du_nd
    
    def dim_du(self, du_nd: torch.Tensor) -> torch.Tensor:
        """Dimensionalize the input state derivatives."""
        dv_nd, dx_nd = du_nd.chunk(2, dim=-1)
        dv1_nd, dv2_nd, dv3_nd = dv_nd[..., 0:2], dv_nd[..., 2:4], dv_nd[..., 4:6]
        dx1_nd, dx2_nd, dx3_nd = dx_nd[..., 0:2], dx_nd[..., 2:4], dx_nd[..., 4:6]
        dv1 = dv1_nd * self.char_acc1
        dv2 = dv2_nd * self.char_acc2
        dv3 = dv3_nd * self.char_acc3
        dx1 = dx1_nd * self.char_vel1
        dx2 = dx2_nd * self.char_vel2
        dx3 = dx3_nd * self.char_vel3
        du = torch.cat([dv1, dv2, dv3, dx1, dx2, dx3], dim=-1)
        return du
