import numpy as np 
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import pairwise_kernels
from wasserstein import wasserstein_distance_nd


def smooth_data(data, sigma=10):
    """
    Smooth the data using a Gaussian filter.
    Args:
        data (np.array): The input data array to smooth.
        sigma (int): The standard deviation for Gaussian kernel.
    Returns:
        np.array: Smoothed data.
    """
    return gaussian_filter1d(data, sigma=sigma)


def mmd(X, Y, kernel="rbf"):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two datasets.
    Args:
        X (np.array): The first dataset. Shape=(len(X), n_features)
        Y (np.array): The second dataset. Shape=(len(Y), n_features)
        kernel (str): The kernel function to use.
    Returns:
        float: The MMD value.
    """

    # Calculate the kernel matrix
    XX = pairwise_kernels(X, X, metric=kernel)
    YY = pairwise_kernels(Y, Y, metric=kernel)
    XY = pairwise_kernels(X, Y, metric=kernel)

    # Compute MMD statistic
    return XX.mean() + YY.mean() - 2 * XY.mean()


class States:
    """A set of phase space states in the form of a 2D array."""

    def __init__(self, u, processor):
        if not isinstance(u, np.ndarray):
            raise TypeError("u must be a numpy array")
        if len(u.shape) > 2 or len(u.shape) == 0:
            raise ValueError("u must be a 1D or 2D array")
        elif len(u.shape) == 1:  # If u is a 1D array, reshape it to a 2D array
            u = u[np.newaxis, :]
        # if u.shape[1] != processor.dof * 2:
        #     raise ValueError("Number of columns in u must be equal to 2 * dof. Got {u.shape[1]} columns and dof={processor.dof}.")

        self.u = u
        self.processor = processor
        self.dim = 2 * processor.dof

    def __len__(self):
        return len(self.u)
    
    def __getitem__(self, idx):
        return type(self)(self.u[idx], self.processor)

    def __repr__(self):
        return f"{self.__class__.__name__}(problem={self.processor}, shape={self.u.shape})"
    
    def get_pq(self):
        return self.processor.get_pq(self.u)
    
    def get_vx(self):
        return self.processor.get_vx(self.u)
    
    def convert_vx_to_pq(self):
        return self.processor.convert_vx_to_pq(self.u)

    def compare(self, other_states):
        """Compare each state of the current set with the corresponding state in another set."""
        if not isinstance(other_states, type(self)):
            raise ValueError(f"Can only compare with another {self.__class__.__name__} object")
        if self.processor != other_states.processor:
            raise ValueError("Other states must have the same processor (problem definition)")
        if len(self) != len(other_states):
            raise ValueError("Other states must have the same length")
        # Returns a dictionary of errors with shape (len(self),)
        return self.processor.compute_errors(self.u, other_states.u)  

    def pairwise_compare(self, other_states):
        """Compare each state of the current set with every state in another set."""
        if not isinstance(other_states, type(self)):
            raise ValueError(f"Can only compare with another {self.__class__.__name__} object")
        if self.processor != other_states.processor:
            raise ValueError("Other states must have the same processor (problem definition)")
        # Returns a dictionary of errors with shape (len(self), len(other_states))
        return self.processor.compute_errors(self.u[:, np.newaxis, :], other_states.u)


class Dataset:
    """A dataset with a name and a set of states."""

    def __init__(self, states: States, name: str) -> None:
        self.states = states
        self.name = name
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(states={self.states}, name={self.name})"    
        
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx):
        return type(self)(self.states[idx], self.name)

    def random_samples(self, n_samples, seed=42):
        np.random.seed(seed)
        idx = np.random.choice(len(self), n_samples, replace=False)
        return self[idx]

    def select_in_box(self, bounds):
        if not all(len(b) == 2 for b in bounds):
            raise ValueError("Each dimension must have a min and max bound")
        if self.states.dim != len(bounds):
            raise ValueError("Number of dimensions in bounds must match data dimensions")
        in_bounds = np.ones(len(self), dtype=bool)
        for dim, (low, high) in enumerate(bounds):
            in_bounds &= (self.states.u[:, dim] >= low) & (self.states.u[:, dim] <= high)
        return self[in_bounds]
    
    def closest_points_to_u0(self, u0, dist_metric="abs_traj_err"):
        """Find the closest point in the dataset to each point in the given set of points u0."""
        distances = self.states.pairwise_compare(u0)[dist_metric]  # shape (len(self), len(u0))
        closest_points = self[np.argmin(distances, axis=0)]  # shape (len(u0), dim)
        closest_dists = np.min(distances, axis=0)  # shape (len(u0),)
        return closest_points, closest_dists
    
    def neighborhood_density_at_u0(self, u0, neighborhood_size=0.1, dist_metric="abs_traj_err"):
        """Compute the fraction of points in the dataset that are within a given neighborhood of each point in u0."""
        distances = self.states.pairwise_compare(u0)[dist_metric]  # shape (len(self), len(u0))
        return np.mean(distances < neighborhood_size, axis=0)  # shape (len(u0),)

    @staticmethod
    def mmd(dataset1, dataset2, kernel="rbf"):  # TODO: check if this is correct
        return mmd(dataset1.states.u, dataset2.states.u, kernel)


def is_multiple(x, y):
    quotient = x / y
    rounded_quotient = round(quotient)
    return abs(rounded_quotient * y - x) < 1e-9  # Adjust threshold as needed

def find_quotient(x, y):
    quotient = x / y
    rounded_quotient = round(quotient)
    return rounded_quotient


class Trajectory:
    """A trajectory of states at different times."""

    def __init__(self, times, states):
        times = np.atleast_1d(times)
        if times.ndim != 1:
            raise ValueError("times must be a 1D array")
        if not isinstance(states, States):
            raise TypeError("states must be a States object")
        if len(times) != len(states):
            raise ValueError(f"Length mismatch: len(times) = {len(times)}, len(states) = {len(states)}")
        self.times = times
        self.states = states
        self.dt = times[1] - times[0] if len(times) > 1 else None

    def __repr__(self):
        if len(self) >= 1:
            return f"{self.__class__.__name__}(t_range=[{self.times[0]}, {self.times[-1]}], dt={self.dt}, u0={self.states.u[0]}, states={self.states})"
        else:
            return f"{self.__class__.__name__}(t_range=[], dt=None, u0=None, states={self.states})"

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        return type(self)(self.times[idx], self.states[idx])

    def select_between(self, t0, t1):
        idx = (self.times >= t0) & (self.times <= t1)
        return self[idx]
    
    def select_with_interval(self, Dt):
        if self.dt is None:
            return self
        if not is_multiple(Dt, self.dt):
            raise ValueError(f"Invalid interval Dt = {Dt}, dt = {self.dt}")
        idx = range(0, len(self.times), find_quotient(Dt, self.dt))
        return self[idx]

    def compare(self, ref_traj):
        matched_traj, matched_ref_traj = Trajectory.intersect(self, ref_traj)
        if matched_traj is None or matched_ref_traj is None:
            print("Trajectories could not be intersected with matched intervals.")
            return None
        print(f"After alignment: \n\ttraj = {matched_traj}\n\tref_traj = {matched_ref_traj}")
        errors = matched_traj.states.compare(matched_ref_traj.states)
        return matched_traj.times, errors
    
    def valid_prediction_time(self, ref_traj, error_name, threshold=1e-3):
        # Time before the first time the error exceeds the threshold
        times, errors = self.compare(ref_traj)
        idx = np.where(errors[error_name] > threshold)[0]
        if len(idx) == 0:
            print(f"Error {error_name} <= {threshold} for all times")
            return times[-1]
        else:
            return times[idx[0]-1] 

    @staticmethod
    def intersect(traj1, traj2, match_dt=True):
        if len(traj1) == 0 or len(traj2) == 0:
            raise ValueError("Trajectories cannot be empty")
        t0 = max(traj1.times[0], traj2.times[0])
        t1 = min(traj1.times[-1], traj2.times[-1])
        if match_dt and traj1.dt is not None and traj2.dt is not None:
            dt = max(traj1.dt, traj2.dt)
            try:
                return traj1.select_between(t0, t1).select_with_interval(dt), traj2.select_between(t0, t1).select_with_interval(dt)
            except AssertionError as e:
                print(e)
                return None, None
        else:
            return traj1.select_between(t0, t1), traj2.select_between(t0, t1)
