"""
Numerical integrator for solving ordinary differential equations.
"""

import torch


class FirstOrderODE:
    """
    First-order ODE of the form 
        dx / dt = f(x, t, p)
    """
    def __init__(self, f):
        if not callable(f):
            raise TypeError(f"Argument f should be callable, got {repr(type(f).__name__)}")
        self.f = f

    @classmethod
    def from_dynamical_ode(cls, dynamical_ode):
        """Convert a dynamical ODE to a first-order ODE."""
        def f(u, t, p):
            v, x = u.chunk(2, dim=-1)
            return torch.cat((dynamical_ode.f1(x, t, p), dynamical_ode.f2(v, p)), dim=-1)
        return cls(f)


class DynamicalODE:
    """
    Dynamical ODE of the form
        dv / dt = f1(x, t, p)
        dx / dt = f2(v, p)
    """

    def __init__(self, f1, f2):
        if not callable(f1):
            raise TypeError(f"Argument f1 should be callable, got {repr(type(f1).__name__)}")
        if not callable(f2):
            raise TypeError(f"Argument f2 should be callable, got {repr(type(f2).__name__)}")
        self.f1 = f1
        self.f2 = f2


class Integrator:
    """
    Solve an initial value problem for an ODE.
    """
    
    def __init__(self, h: float, nsteps: int):
        """
        Initialize the integrator.

        :param h: stepsize
        :param nsteps: number of time steps, default is 1
        """
        self.h = h
        self.nsteps = nsteps
        self.T = h * nsteps  # total time

    def __call__(self, x0, t0):
        """
        Integrate the ODE.

        :param x0: initial state (batch_size, dim)
        :param t0: initial time (batch_size, 1)

        :returns: final state at t = t0 + T (batch_size, dim)
        """
        raise NotImplementedError("Method '__call__' not implemented.")
   
    def compute_residual(self, x_n, x_n_plus_1, t_n, h, p):
        """
        Compute the residual for a given step.
        
        :param x_n: state at the current step (batch_size, dim)
        :param x_n_plus_1: state at the next step (batch_size, dim)
        :param t_n: current time (batch_size, 1)
        :param h: stepsize (batch_size, 1) or scalar
        :param p: parameters (a dictionary of tensors each of shape (batch_size, 1))
        
        :returns: residual (batch_size, dim)
        """
        raise NotImplementedError("Method 'compute_residual' not implemented.")
