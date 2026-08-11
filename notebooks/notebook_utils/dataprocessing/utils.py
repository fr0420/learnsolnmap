import numpy as np 
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import pairwise_kernels


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


def is_multiple(x, y):
    quotient = x / y
    rounded_quotient = round(quotient)
    return abs(rounded_quotient * y - x) < 1e-9  # Adjust threshold as needed


def find_quotient(x, y):
    quotient = x / y
    rounded_quotient = round(quotient)
    return rounded_quotient
