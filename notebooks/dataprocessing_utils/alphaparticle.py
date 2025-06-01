import os
import numpy as np 
import pandas as pd
import torch
from .utils import States, Trajectory, Dataset

EPSILON_VAL1 = 0.05
EPSILON_VAL2 = 0.15
EPSILON_VAL3 = 0.27
EPSILON_VAL4 = 0.3
EPSILON_VAL5 = 0.4

REF_TRAJ_FILEPATHS = {
    EPSILON_VAL1: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=5e-2/1/202503070013/ref/u.csv",
            "dt": 0.01,
        },
        2: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=5e-2/2/202503071136/ref/u.csv",
            "dt": 0.01,
        },
        3: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=5e-2/3/202503081435/ref/u.csv",
            "dt": 0.01,
        }
    },
    EPSILON_VAL2: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=1.5e-1/1/202503041218/ref/u.csv",
            "dt": 0.01,
        },
        2: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=1.5e-1/2/202503071148/ref/u.csv",
            "dt": 0.01,
        },
        3: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=1.5e-1/3/202503081405/ref/u.csv",
            "dt": 0.01,
        },
    },
    EPSILON_VAL3: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=2.7e-1/1/202503070022/ref/u.csv",
            "dt": 0.01,
        },
        2: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=2.7e-1/2/202503071208/ref/u.csv",
            "dt": 0.01,
        },
        3: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=2.7e-1/3/202503081341/ref/u.csv",
            "dt": 0.01,
        }
    },
    EPSILON_VAL4: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=3e-1/1/202503070152/ref/u.csv",
            "dt": 0.01,
        },
        2: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=3e-1/2/202503071227/ref/u.csv",
            "dt": 0.01,
        },
        3: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=3e-1/3/202503081227/ref/u.csv",
            "dt": 0.01,
        }
    },
    EPSILON_VAL5: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=4e-1/1/202503070213/ref/u.csv",
            "dt": 0.01,
        },
        2: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=4e-1/2/202503071236/ref/u.csv",
            "dt": 0.01,
        },
        3: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=4e-1/3/202503071245/ref/u.csv",
            "dt": 0.01,
        },
    },
}


IM_0_1_REF_TRAJ_FILEPATHS = {
    EPSILON_VAL1: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=5e-2/1/202503152107/ref/u.csv",
            "dt": 0.1,
        },
    },
    EPSILON_VAL2: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=1.5e-1/1/202503152106/ref/u.csv",
            "dt": 0.1,
        },
    },
    EPSILON_VAL3: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=2.7e-1/1/202503152105/ref/u.csv",
            "dt": 0.1,
        },
    },
    EPSILON_VAL4: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=3e-1/1/202503152109/ref/u.csv",
            "dt": 0.1,
        },
    },
    EPSILON_VAL5: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=4e-1/1/202503152111/ref/u.csv",
            "dt": 0.1,
        },
    },
}

IM_0_01_REF_TRAJ_FILEPATHS = {
    EPSILON_VAL1: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=5e-2/1/202503162032/ref/u.csv",
            "dt": 0.01,
        },
    },
    EPSILON_VAL2: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=1.5e-1/1/202503162017/ref/u.csv",
            "dt": 0.01,
        },
    },
    EPSILON_VAL3: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=2.7e-1/1/202503162020/ref/u.csv",
            "dt": 0.01,
        },
    },
    EPSILON_VAL4: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=3e-1/1/202503162023/ref/u.csv",
            "dt": 0.01,
        },
    },
    EPSILON_VAL5: {
        1: {
            "filepath": "/workspace/projects_rui/learnsolnmap/out/alphaparticle/eps=4e-1/1/202503162030/ref/u.csv",
            "dt": 0.01,
        },
    },
}


def bisection_root(u_start, dt, Phi, h, tol=1e-6, max_iter=50):
    """
    Find the time tau in [0, dt] such that h(Phi(u_start, tau)) == 0 using the bisection method.
    
    Parameters:
        u_start : np.array
            The starting state at the beginning of the interval.
        dt : float
            The time interval length.
        Phi: function
            Function that defines the flow map Phi(u, t).
        h : function
            Function that defines the hypersurface h(u)=0.
        tol : float, optional
            Tolerance for the root-finding.
        max_iter : int, optional
            Maximum number of iterations.
    
    Returns:
        tau : float
            The estimated time (relative to the start of the interval) at which h(u) = 0.
    """
    a = 0.0
    b = dt
    fa = h(u_start)  # f(a) = h(Phi(u_start, 0))
    fb = h(Phi(u_start, dt))

    # Ensure the bisection method is applicable:
    if fa * fb > 0:
        raise ValueError("Bisection method requires a sign change in the interval.")
    
    for i in range(max_iter):
        mid = (a + b) / 2.0
        fmid = h(Phi(u_start, mid))
        if abs(fmid) < tol or (b - a) / 2.0 < tol:
            # print(f"Bisection method converged in {i} iterations.")
            # print(f"Final interval: [{a}, {b}], f(a) = {fa}, f(b) = {fb}")
            return mid
        if fa * fmid < 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid

    print(f"Warning: Bisection method did not converge in {max_iter} iterations.")
    print(f"Final interval: [{a}, {b}], f(a) = {fa}, f(b) = {fb}")

    return (a + b) / 2.0


def poincare_section(u0, Phi, h, direction='both', dt=0.01, T=100, tol=1e-6, reset_on_crossing=False):
    """
    Generate the Poincaré section for an autonomous ODE system using the bisection method
    for root-finding, saving both the crossing time and the state.
    
    Parameters:
        u0 : np.array
            Initial condition.
        Phi: function 
            A function Phi(u, t) defining the flow map.
        h : function
            A function h(u) defining the hypersurface (Poincaré section) as h(u)=0.
        direction : str, optional
            Direction of crossing to record:
              - 'increasing': record only when h(u) goes from negative to positive.
              - 'decreasing': record only when h(u) goes from positive to negative.
              - 'both': record all crossings.
        dt : float, optional
            Time step for integration.
        T : float, optional
            Total integration time.
        tol : float, optional
            Tolerance for the bisection method.
        reset_on_crossing : bool, optional
            If True, when a crossing is found, the function resets the state and time
            to the crossing point and continues from there. If False, it proceeds with
            fixed time steps.
            
    Returns:
        intersections : list of tuples
            Each tuple is (t_cross, u_cross), where t_cross is the time at the crossing and
            u_cross is the state at which the trajectory crosses the hypersurface.
    """
    
    intersections = []
    t = 0.0
    prev_u = np.array(u0)
    prev_h = h(prev_u)
    
    # Check if the initial condition is already on the hypersurface.
    if abs(prev_h) < tol:
        intersections.append((t, prev_u.copy()))
    
    while t < T:
        # Advance the solution by dt using the available solver Phi.
        next_u = Phi(prev_u, dt)
        next_h = h(next_u)
        
        # Check if the hypersurface was crossed between prev_u and next_u.
        if prev_h * next_h < 0:
            # Determine crossing direction.
            crossing_direction = None
            if prev_h < 0 and next_h >= 0:
                crossing_direction = 'increasing'
            elif prev_h > 0 and next_h <= 0:
                crossing_direction = 'decreasing'
            
            if direction == 'both' or (direction == crossing_direction):
                try:
                    # Use bisection to find the fraction tau in [0, dt] where the crossing occurs.
                    tau = bisection_root(prev_u, dt, Phi, h, tol=tol)
                    t_cross = t + tau
                    u_cross = Phi(prev_u, tau)
                    intersections.append((t_cross, u_cross.copy()))
                    
                    if reset_on_crossing:
                        # Reset the current state and time to the crossing point.
                        t = t_cross
                        prev_u[:] = u_cross[:]
                        prev_h = h(prev_u)
                        continue  # Restart loop from the crossing point.
                except ValueError:
                    # If bisection fails, simply skip this interval.
                    pass
        
        # Update state and time if no crossing was detected (or if bisection failed)
        t += dt
        prev_u[:] = next_u[:]
        prev_h = next_h
    
    return intersections


def batch_bisection_root(u_start, dt, Phi, h, tol=1e-6, max_iter=50):
    """
    Vectorized bisection root finder for a batch of initial conditions.
    
    Parameters:
        u_start : torch.Tensor
            Batch of initial states of shape (batch_size, state_dim).
        dt : float
            Time interval length.
        Phi: function
            Flow-map function that accepts a state batch and a tensor of times.
        h : function
            Function defining the hypersurface h(u)=0; accepts a state batch and returns a tensor of shape (batch_size,).
        tol : float, optional
            Tolerance for root finding.
        max_iter : int, optional
            Maximum number of iterations.
            
    Returns:
        tau : torch.Tensor
            Estimated crossing times for each sample (shape: (batch_size,)).
    """
    batch_size = u_start.shape[0]
    a = torch.zeros(batch_size, device=u_start.device, dtype=u_start.dtype)
    b = torch.full((batch_size,), dt, device=u_start.device, dtype=u_start.dtype)
    
    fa = h(u_start)
    fb = h(Phi(u_start, b))

    if torch.any(fa * fb > 0):
        raise ValueError("Bisection method requires a sign change in the interval for every sample.")
    
    for i in range(max_iter):
        mid = (a + b) / 2.0
        fmid = h(Phi(u_start, mid))

        # Convergence: either function value is small or half-interval is small.
        converged = (fmid.abs() < tol) | (((b - a) / 2.0) < tol)
        if converged.all():
            # print(f"Bisection method converged in {i} iterations.")
            # print(f"Final interval: [{a}, {b}], f(a) = {fa}, f(b) = {fb}")
            return mid
        # Update intervals based on sign change.
        update_left = (fa * fmid) >= 0  # root is in [mid, b]
        update_right = ~update_left     # root is in [a, mid]
        a[update_left] = mid[update_left]
        fa[update_left] = fmid[update_left]
        b[update_right] = mid[update_right]
        fb[update_right] = fmid[update_right]
    
    print(f"Warning: Bisection method did not converge in {max_iter} iterations.")
    print(f"Final interval: [{a}, {b}], f(a) = {fa}, f(b) = {fb}")

    return (a + b) / 2.0


def batch_poincare_section(u0, Phi, h, direction='both', dt=0.01, T=100, tol=1e-6, 
                           reset_on_crossing=False, method='bisection'):
    """
    Generate a Poincaré section for a batch of initial conditions, 
    simultaneously checking crossing directions and performing batch bisection.
    
    Parameters:
        u0 : torch.Tensor
            Batch of initial conditions of shape (batch_size, state_dim).
        Phi: function 
            Flow-map function that accepts a batch of states and times.
        h : function
            Hypersurface function h(u)=0; should accept a batch and return a tensor of shape (batch_size,).
        direction : str, optional
            Which crossings to record ('increasing', 'decreasing', or 'both').
        dt : float, optional
            Integration time step.
        T : float, optional
            Total integration time.
        tol : float, optional
            Tolerance for the bisection method.
        reset_on_crossing : bool, optional
            If True, reset state and local time for a sample when a crossing is found.
        method : str, optional
            Method for estimating the crossing point. Either 'bisection' or 'interpolation'.
            
    Returns:
        intersections : list of lists
            intersections[i] is a list of tuples (t_cross, u_cross) for the i–th sample.
    """
    batch_size = u0.shape[0]
    device = u0.device
    t = torch.zeros(batch_size, device=device, dtype=u0.dtype)  # per-sample time
    prev_u = u0.clone()  # current state for each sample
    prev_h = h(prev_u)
    
    intersections = [[] for _ in range(batch_size)]
    
    # Active mask: samples that haven't reached T
    active = t < T
    while active.any():
        # Only update for active samples.
        active_idx = active.nonzero(as_tuple=True)[0]
        
        # Compute next state for active samples.
        current_u = prev_u[active_idx]
        next_u = Phi(current_u, dt)
        next_h = h(next_u)
        
        # Identify crossings: where the sign changes.
        cross_mask = (prev_h[active_idx] * next_h) < 0

        if cross_mask.any():
            # For these samples, determine the crossing direction in batch:
            prev_h_active = prev_h[active_idx][cross_mask]
            next_h_active = next_h[cross_mask]
            
            # Create a tensor indicating the crossing direction.
            #  1 for 'increasing', -1 for 'decreasing'
            crossing_direction = torch.where(prev_h_active < 0, torch.tensor(1, device=device), torch.tensor(-1, device=device))
            
            # Determine which indices meet the desired direction:
            if direction == 'both':
                valid = torch.ones_like(crossing_direction, dtype=torch.bool)
            elif direction == 'increasing':
                valid = crossing_direction == 1
            elif direction == 'decreasing':
                valid = crossing_direction == -1
            else:
                raise ValueError("direction must be 'increasing', 'decreasing', or 'both'")

            if valid.any():
                # Map back to full active index indices.
                cross_indices = active_idx[cross_mask][valid]
                # The corresponding previous state for valid crossings.
                u_cross_batch = prev_u[cross_indices]
                
                if method == 'bisection':
                    # Use the existing bisection routine to estimate the time shift.
                    tau = batch_bisection_root(u_cross_batch, dt, Phi, h, tol=tol)
                    # Update the crossing times and states.
                    t_cross = t[cross_indices] + tau
                    u_at_cross = Phi(u_cross_batch, tau)
                elif method == 'interpolation':
                    # Directly estimate u_cross via linear interpolation in state space.
                    # Compute lambda as the fraction along dt where the crossing occurs.
                    u_next_valid = Phi(u_cross_batch, dt)
                    h_prev_valid = h(u_cross_batch)
                    h_next_valid = h(u_next_valid)
                    lambda_ = h_prev_valid / (h_prev_valid - h_next_valid)
                    t_cross = t[cross_indices] + lambda_ * dt
                    # Interpolate in state space: u_cross = u_prev + lambda*(u_next - u_prev)
                    u_at_cross = u_cross_batch + lambda_.unsqueeze(1) * (u_next_valid - u_cross_batch)
                else:
                    raise ValueError("method must be 'bisection' or 'interpolation'")

                # Record intersections.
                for idx, tc, uc in zip(cross_indices.tolist(), t_cross.tolist(), u_at_cross.tolist()):
                    intersections[idx].append((tc, uc))
                if method == 'bisection' and reset_on_crossing:
                    # Reset state and time for the crossing samples.
                    t[cross_indices] = t_cross
                    prev_u[cross_indices] = u_at_cross.clone()
                    prev_h[cross_indices] = h(u_at_cross)
                    # Skip further dt advancement for these samples.
                    # They remain active if t < T.
                    # Continue to next iteration.
        
        # For active samples, update time and state (those not reset).
        # For samples that had a valid crossing and were reset, t and prev_u are already updated.
        # For all others, simply advance dt.
        no_reset = torch.ones_like(active, dtype=torch.bool)
        if reset_on_crossing:
            # Create a boolean mask for all active samples that did NOT experience a reset:
            # For simplicity, assume that if a crossing was detected, we reset all samples in cross_indices.
            # Otherwise, they are updated here.
            reset_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            if cross_mask.any():
                reset_mask[active_idx[cross_mask]] = True
            no_reset = ~reset_mask
        
        update_idx = active_idx[no_reset[active_idx]]
        if update_idx.numel() > 0:
            t[update_idx] += dt
            prev_u[update_idx] = Phi(prev_u[update_idx], dt)
            prev_h[update_idx] = h(prev_u[update_idx])
        
        active = t < T

    return intersections


class AlphaParticle:

    def __init__(self, epsilon=0.05, a1=0.3):
        self.dof = 4
        self.epsilon = epsilon
            
        self.B0 = 1.0
        self.a1 = a1
        self.a2 = 0.3
        self.k_x1 = 3.0
        self.k_y1 = 1.0
        self.k_x2 = 1.0
        self.k_y2 = 3.0
        
    def __repr__(self):
        return f"AlphaParticle(epsilon={self.epsilon}, a1={self.a1})"

    def __eq__(self, other):
        if isinstance(other, AlphaParticle):
            return self.epsilon == other.epsilon and self.a1 == other.a1
        return NotImplemented

    def get_xy(self, u):
        return u[:, 2], u[:, 3]
    
    def compute_B(self, x, y):
        """Compute magnetic field."""
        return self.B0 + self.a1 * np.cos(self.k_x1 * x + self.k_y1 * y) + self.a2 * np.cos(self.k_x2 * x + self.k_y2 * y)

    def compute_dBdx(self, x, y):
        """Compute derivative of magnetic field w.r.t. x."""
        return -self.a1 * self.k_x1 * np.sin(self.k_x1 * x + self.k_y1 * y) - self.a2 * self.k_x2 * np.sin(self.k_x2 * x + self.k_y2 * y)
    
    def compute_dBdy(self, x, y):
        """Compute derivative of magnetic field w.r.t. y."""
        return -self.a1 * self.k_y1 * np.sin(self.k_x1 * x + self.k_y1 * y) - self.a2 * self.k_y2 * np.sin(self.k_x2 * x + self.k_y2 * y)
    
    def compute_du(self, u):
        """Compute time derivative of u = (vx,vy,x,y)."""
        vx, vy, x, y = u[..., (0,)], u[..., (1,)], u[..., (2,)], u[..., (3,)]
        Bxy = self.compute_B(x, y)
        dvxdt = Bxy * vy
        dvydt = -Bxy * vx
        dxdt = self.epsilon * vx
        dydt = self.epsilon * vy
        return np.concatenate((dvxdt, dvydt, dxdt, dydt), axis=-1)

    def compute_ddu(self, u):
        """Compute second time derivative of u = (vx,vy,x,y)."""
        vx, vy, x, y = u[..., (0,)], u[..., (1,)], u[..., (2,)], u[..., (3,)]
        Bxy = self.compute_B(x, y)
        Bxy_sq = Bxy**2
        dBdx = self.compute_dBdx(x, y)
        dBdy = self.compute_dBdy(x, y)
        eps = self.epsilon
        dBdt = eps * (dBdx * vx + dBdy * vy)
        d2vxdt2 = dBdt * vy - Bxy_sq * vx
        d2vydt2 = -dBdt * vx - Bxy_sq * vy
        d2xdt2 = eps * Bxy * vy
        d2ydt2 = - eps * Bxy * vx
        return np.concatenate((d2vxdt2, d2vydt2, d2xdt2, d2ydt2), axis=-1)

    def compute_H(self, u):
        vx, vy = u[..., 0], u[..., 1]
        return 0.5 * (vx**2 + vy**2)
    
    def compute_jacobian(self, u):
        """Compute the Jacobian of the vector field (du/dt) with respect to u."""
        hessian_H = self.compute_hessian_H(u)
        J_inv = np.zeros((12, 12))
        J_inv[:6, 6:] = -np.eye(6)
        J_inv[6:, :6] = np.eye(6)
        jacobian = np.einsum('ij,bjk->bik', J_inv, hessian_H)
        return jacobian

    def compute_stiffness(self, u, zero_tol=1e-6, ratio=False):
        """Compute the stiffness measure of the vector field (dp/dt, dq/dt) at the point u."""
        jacobian = self.compute_jacobian(u)
        stiffness_values = np.zeros(len(u))
        for i in range(len(u)):
            try:
                eigenvalues = np.linalg.eigvals(jacobian[i])
                eigenvalues_magnitudes = np.abs(eigenvalues)
                nonzero_vals = eigenvalues_magnitudes[eigenvalues_magnitudes > zero_tol]
                if ratio: 
                    stiffness_values[i] = np.max(nonzero_vals) / np.min(nonzero_vals)
                else:
                    stiffness_values[i] = np.max(nonzero_vals)
            except np.linalg.LinAlgError:
                stiffness_values[i] = np.nan
            except ValueError:
                stiffness_values[i] = np.nan
        return stiffness_values

    def compute_stiffness2(self, u, zero_tol=1e-14, ratio=False):
        """Compute the stiffness measure of the vector field (dp/dt, dq/dt) at the point u."""
        hessian_H = self.compute_hessian_H(u)
        matrices = hessian_H[:, :6, :6] @ hessian_H[:, 6:, 6:]
        stiffness_values = np.zeros(len(u))
        for i in range(len(u)):
            try:
                eigenvalues = np.linalg.eigvals(matrices[i])
                eigenvalues_magnitudes = np.abs(eigenvalues)
                nonzero_vals = eigenvalues_magnitudes[eigenvalues_magnitudes > zero_tol]
                sqrt_vals = np.sqrt(nonzero_vals)
                if ratio: 
                    stiffness_values[i] = np.max(sqrt_vals) / np.min(sqrt_vals)
                else:
                    stiffness_values[i] = np.max(sqrt_vals)
            except np.linalg.LinAlgError:
                stiffness_values[i] = np.nan
                print("LinAlgError")
            except ValueError:
                stiffness_values[i] = np.nan
                print("ValueError")
        return stiffness_values
    
    def compute_errors(self, u, ref_u):

        # Compute trajectory errors
        diff_squares = (u - ref_u)**2
        abs_traj_err = np.sqrt(np.sum(diff_squares, axis=-1))
        rel_traj_err = abs_traj_err / np.linalg.norm(ref_u, axis=-1)
        abs_traj_err_xy = np.sqrt(np.sum(diff_squares[..., 2:], axis=-1))
        rel_traj_err_xy = abs_traj_err_xy / np.linalg.norm(ref_u[..., 2:], axis=-1)
        abs_traj_err_vxvy = np.sqrt(np.sum(diff_squares[..., :2], axis=-1))
        rel_traj_err_vxvy = abs_traj_err_vxvy / np.linalg.norm(ref_u[..., :2], axis=-1)

        # Compute Hamiltonian errors
        abs_H_err = np.abs(self.compute_H(u) - self.compute_H(ref_u))

        return {
            "abs_traj_err": abs_traj_err, 
            "rel_traj_err": rel_traj_err,
            "abs_traj_err_xy": abs_traj_err_xy,
            "rel_traj_err_xy": rel_traj_err_xy,
            "abs_traj_err_vxvy": abs_traj_err_vxvy,
            "rel_traj_err_vxvy": rel_traj_err_vxvy,
            "abs_H_err": abs_H_err,
        }
    
    def compute_taylor_series(self, u0, t, order=1):
        """Compute the Taylor series expansion of the solution at time t with u(0)=u0."""
        # u0: (4,) or (batch_size, 4)
        # t: scalar or (n,) array
        # order: 0, 1, or 2
        # returns u: (n, 4) or (n, batch_size, 4)

        if isinstance(t, (int, float)):
            t = np.array([t])
        if not isinstance(t, np.ndarray):
            raise ValueError("t must be a scalar or a numpy array.")
        if t.ndim > 1:
            raise ValueError("t must be a scalar or a 1D array.")
        
        # reshape t for broadcasting
        t = t.reshape(t.shape + (1,) * u0.ndim)

        if order == 0:
            return u0
        elif order == 1:
            return u0 + t * self.compute_du(u0)
        elif order == 2:
            du = self.compute_du(u0)
            ddu = self.compute_ddu(u0)
            return u0 + t * du + 0.5 * t**2 * ddu
        else:
            raise ValueError("Only orders 0, 1, and 2 are supported.")


class AlphaParticleDataset(Dataset):

    @classmethod
    def load_from_file(cls, filepath, name="none", **kwargs):
        df = pd.read_csv(filepath)
        data = States(df.values, AlphaParticle(**kwargs))
        return cls(data, name)
    
    @classmethod
    def from_points(cls, points, name="none", **kwargs):
        return cls(States(points, AlphaParticle(**kwargs)), name)
    

class AlphaParticleTrajectory(Trajectory): 

    @classmethod
    def load_from_file(cls, filepath, dt=None, **kwargs):
        df = pd.read_csv(filepath)
        states = States(df.values, AlphaParticle(**kwargs))
        times_filepath = filepath.replace("u.csv", "t.csv")
        if os.path.exists(times_filepath):
            times = pd.read_csv(times_filepath).values.flatten()
        elif dt is not None:
            times = np.arange(0, len(states)) * dt
        else:
            raise ValueError("Either provide a valid dt or a times file.")
        return cls(times, states)

    @classmethod
    def from_u(cls, times, u, **kwargs):
        return cls(times, States(u, AlphaParticle(**kwargs)))
