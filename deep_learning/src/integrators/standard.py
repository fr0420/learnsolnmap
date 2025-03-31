"""
Standard integrators for solving first-order ordinary differential equations.
"""

import torch
from integrators.integrator import FirstOrderODE, Integrator


class StandardIntegrator(Integrator):
    """
    Solve an initial value problem for a first-order ODE.
        
        dx / dt = f(x, t, p)
        x(0) = x0
    """

    def __init__(self, problem: FirstOrderODE, **kwargs):
        """
        Initialize the integrator.

        :param problem: first-order ODE
        """
        super(StandardIntegrator, self).__init__(**kwargs)
        if not isinstance(problem, FirstOrderODE):
            raise TypeError(f"Argument problem should be an instance of FirstOrderODE, got {repr(type(problem).__name__)}")
        self.f = problem.f

    def __repr__(self):
        """Return a string representation of the integrator."""
        return f"{self.__class__.__name__}(interval=[0, {self.T}], h={self.h}, nsteps={self.nsteps}, f={self.f.__name__})"
    
    def __call__(self, x0, t0=None, retfull=False):
        """
        Integrate the ODE.

        :param x0: initial state (batch_size, dim)
        :param t0: initial time (batch_size, 1)
        :param retfull: whether or not to return solutions at all time points

        :returns: sequence of states (nsteps + 1, batch_size, dim) if retfull is True, otherwise 
                  only the final state (batch_size, dim) at t = t0 + T
        """
        if t0 is None:
            t0 = torch.zeros_like(x0)[:, :1]
        return self.integrate(x0, t0, self.nsteps, retfull)[0]

    def compute_residual(self, x_n, x_n_plus_1, t_n, h, p):
        """
        Compute the residual for a given step.
        
        :param x_n: state at the current step (batch_size, dim)
        :param x_n_plus_1: state at the next step (batch_size, dim)
        :param t_n: current time (batch_size, 1)
        :param h: stepsize (batch_size, 1) or scalar
        :param p: parameters (a dictionary of tensors each of shape (batch_size, 1))

        :returns: 
            true x_n_plus_1 (batch_size, dim) 
            predicted x_n_plus_1 (batch_size, dim) 
        """
        if self.is_explicit:
            return x_n_plus_1, self.step(x_n, t_n, h, p)
        else:
            raise NotImplementedError("Method 'compute_residual' not implemented.")
    
    def step(self, x, t, h, p):
        """
        Perform a single integration step.

        :param x: current state (batch_size, dim)
        :param t: current time (batch_size, 1)
        :param h: stepsize (batch_size, 1) or scalar
        :param p: parameters (a dictionary of tensors each of shape (batch_size, 1))

        :returns: next state (batch_size, dim)
        """
        raise NotImplementedError("Method 'step' not implemented.")
    
    def integrate(self, x0, t0, nsteps, retfull=False):
        """
        Integrate the ODE.

        :param x0: initial state (batch_size, dim)
        :param t0: initial time (batch_size, 1)
        :param nsteps: number of time steps
        :param retfull: whether or not to return solutions at all time points

        :returns: sequence of states (nsteps + 1, batch_size, dim), sequence of times (nsteps + 1, batch_size, 1) 
                  if retfull is True, otherwise only the final state (batch_size, dim) and time (batch_size, 1)
        """
        dt = torch.arange(nsteps+1) * self.h

        if retfull:
            batch_size, dim = x0.shape
            trajectory = torch.zeros((nsteps+1, batch_size, dim))
            trajectory[0] = x0

            x = x0
            for i in range(nsteps): 
                x = self.step(x, t0 + dt[i], self.h)  # x_{i+1} = step(x_i, t_i, h)
                trajectory[i+1] = x
            return trajectory, dt[:, None, None] + t0
        
        else:
            x = x0
            for i in range(nsteps):
                x = self.step(x, t0 + dt[i], self.h)  # x_{i+1} = step(x_i, t_i, h)
            return x, t0 + dt[-1]
    

class ExplicitStandardIntegrator(StandardIntegrator):

    def __init__(self, *args, **kwargs):
        super(ExplicitStandardIntegrator, self).__init__(*args, **kwargs)
        self.is_explicit = True


class ImplicitStandardIntegrator(StandardIntegrator):
    
    def __init__(self, *args, **kwargs):
        super(ImplicitStandardIntegrator, self).__init__(*args, **kwargs)
        self.is_explicit = False


class ForwardEuler(ExplicitStandardIntegrator):

    def step(self, x, t, h, p):
        x_next = x + h * self.f(x, t, p)
        return x_next


class BackwardEuler(ImplicitStandardIntegrator):  # TODO: implement implicit step()

    def compute_residual(self, x_n, x_n_plus_1, t_n, h, p):
        return x_n_plus_1, x_n + h * self.f(x_n_plus_1, t_n + h, p)


class ExplicitMidpoint(ExplicitStandardIntegrator):

    def step(self, x, t, h, p):
        k1 = self.f(x, t, p)
        k2 = self.f(x + h * k1 / 2, t + h / 2, p)
        x_next = x + h * k2
        return x_next


class ImplicitMidpoint(ImplicitStandardIntegrator):  # TODO: implement implicit step()

    def compute_residual(self, x_n, x_n_plus_1, t_n, h, p):
        return x_n_plus_1, x_n + h * self.f((x_n + x_n_plus_1) / 2, t_n + h / 2, p)


class RK3(ExplicitStandardIntegrator):

    def step(self, x, t, h, p):
        k1 = self.f(x, t, p)
        k2 = self.f(x + h * k1 / 2, t + h / 2, p)
        k3 = self.f(x - h * k1 + 2 * h * k2, t + h, p)
        x_next = x + h * (k1 / 6 + k2 * 2/3 + k3 / 6)
        return x_next
    

class RK4(ExplicitStandardIntegrator):

    def step(self, x, t, h, p):
        k1 = self.f(x, t, p)
        k2 = self.f(x + h * k1 / 2, t + h / 2, p)
        k3 = self.f(x + h * k2 / 2, t + h / 2, p)
        k4 = self.f(x + h * k3, t + h, p)
        x_next = x + h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
        return x_next
    