import numpy as np 


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
