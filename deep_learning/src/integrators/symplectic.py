"""
Symplectic integrators for solving dynamical ordinary differential equations.
"""

import torch
from integrators.integrator import DynamicalODE, Integrator


class SymplecticIntegrator(Integrator):
    """
    Solve an initial value problem for dynamical ODEs of the form
        
        dv / dt = f1(x, t, p)
        dx / dt = f2(v, p)
        v(0) = v0
        x(0) = x0 
    """

    def __init__(self, problem: DynamicalODE, **kwargs):
        """
        Initialize the integrator.

        :param problem: dynamical ODE
        """
        super(SymplecticIntegrator, self).__init__(**kwargs)
        self.f1 = problem.f1
        self.f2 = problem.f2

    def __repr__(self):
        """Return a string representation of the integrator."""
        return f"{self.__class__.__name__}(interval=[0, {self.T}], h={self.h}, nsteps={self.nsteps}, f1={self.f1.__name__}, f2={self.f2.__name__})"
    
    def __call__(self, u0, t0=None, p=None, retfull=False):
        """
        Integrate the ODE given the concatenated initial state u0 = [v0, x0].
        
        :param u0: initial state u0 = [v0, x0] (batch_size, 2*dim)
        :param t0: initial time (batch_size, 1)
        :param p: parameters (a dictionary of tensors each of shape (batch_size, 1))
        :param retfull: whether or not to return solutions at all time points
        
        :returns: sequence of states (nsteps + 1, batch_size, 2*dim) if retfull is True, otherwise 
                  only the final state (batch_size, 2*dim) at t = t0 + T 
        """
        if t0 is None:
            t0 = torch.zeros_like(u0)[:, :1]
        v0, x0 = u0.chunk(2, dim=-1)
        v, x, _ = self.integrate(v0, x0, t0, p, self.nsteps, retfull)
        return torch.cat((v, x), dim=-1)
    
    def compute_residual(self, u_n, u_n_plus_1, t_n, h, p):
        """
        Compute the residual for a given step.
        
        :param u_n: state at the current step (batch_size, 2*dim)
        :param u_n_plus_1: state at the next step (batch_size, 2*dim)
        :param t_n: current time (batch_size, 1)
        :param h: stepsize (batch_size, 1) or scalar
        :param p: parameters (a dictionary of tensors each of shape (batch_size, 1))
        
        :returns: residual (batch_size, 2*dim)
        """
        if self.is_explicit:
            v_n, x_n = u_n.chunk(2, dim=-1)
            return u_n_plus_1, torch.cat(self.step(v_n, x_n, t_n, h, p), dim=-1)
        else:
            raise NotImplementedError("Method 'compute_residual' not implemented.")
    
    def step(self, v, x, t, h, p):
        """
        Perform a single integration step.

        :param v: current v (batch_size, dim)
        :param x: current x (batch_size, dim)
        :param t: current time (batch_size, 1)
        :param h: stepsize (batch_size, 1) or scalar
        :param p: parameters (a dictionary of tensors each of shape (batch_size, 1))

        :returns: next v (batch_size, dim), next x (batch_size, dim)
        """
        raise NotImplementedError("Method 'step' not implemented.")
    
    def integrate(self, v0, x0, t0, p, nsteps, retfull=False):
        """
        Integrate the ODE.

        :param v0: initial v (batch_size, dim)
        :param x0: initial x (batch_size, dim)
        :param t0: initial time (batch_size, 1)
        :param p: parameters (a dictionary of tensors each of shape (batch_size, 1))
        :param nsteps: number of time steps
        :param retfull: whether or not to return solutions at all time points

        :returns: sequence of v's (nsteps + 1, batch_size, dim), sequence of x's (nsteps + 1, batch_size, dim), 
                  sequence of times (nsteps + 1, batch_size, 1) if retfull is True, otherwise only the final 
                  v (batch_size, dim), x (batch_size, dim) and time (batch_size, 1)
        """
        dt = torch.arange(nsteps+1, dtype=v0.dtype).to(v0.device) * self.h

        if retfull:
            batch_size, dim = x0.shape
            trajectory_v = torch.zeros((nsteps+1, batch_size, dim), dtype=v0.dtype).to(v0.device)
            trajectory_x = torch.zeros((nsteps+1, batch_size, dim), dtype=x0.dtype).to(v0.device)
            trajectory_v[0] = v0
            trajectory_x[0] = x0

            v = v0
            x = x0
            for i in range(nsteps):
                v, x = self.step(v, x, t0 + dt[i], self.h, p)  # v_{i+1}, x_{i+1} = step(v_i, x_i, t_i, h)
                trajectory_v[i+1] = v
                trajectory_x[i+1] = x
            return trajectory_v, trajectory_x, dt[:, None, None] + t0

        else:
            v = v0
            x = x0
            for i in range(nsteps):
                v, x = self.step(v, x, t0 + dt[i], self.h, p)
            return v, x, t0 + dt[-1]


class ExplicitSymplecticIntegrator(SymplecticIntegrator):

    def __init__(self, *args, **kwargs):
        super(ExplicitSymplecticIntegrator, self).__init__(*args, **kwargs)
        self.is_explicit = True


class SymplecticEuler(ExplicitSymplecticIntegrator):

    def step(self, v, x, t, h, p):
        v_next = v + self.f1(x, t, p) * h
        x_next = x + self.f2(v_next, p) * h
        return v_next, x_next


class SymplecticEuler2(ExplicitSymplecticIntegrator):
    
    def step(self, v, x, t, h, p):
        x_next = x + self.f2(v, p) * h
        v_next = v + self.f1(x_next, t + h, p) * h
        return v_next, x_next


class VelocityVerlet(ExplicitSymplecticIntegrator):

    def step(self, v, x, t, h, p):
        v_half = v + 0.5 * self.f1(x, t, p) * h
        x_next = x + self.f2(v_half, p) * h
        v_next = v_half + 0.5 * self.f1(x_next, t + h, p) * h
        return v_next, x_next


class PositionVerlet(ExplicitSymplecticIntegrator):

    def step(self, v, x, t, h, p):
        x_half = x + 0.5 * self.f2(v, p) * h
        v_next = v + self.f1(x_half, t + h / 2, p) * h
        x_next = x_half + 0.5 * self.f2(v_next, p) * h
        return v_next, x_next


class Ruth3(ExplicitSymplecticIntegrator):
    """
    Reference: https://cds.cern.ch/record/143981/files/cer-000055082.pdf
    """
    def step(self, v, x, t, h, p):
        c1, c2, c3 = 7./24, 3./4, -1./24
        d1, d2, d3 = 2./3, -2./3, 1.

        v1 = v + h * c1 * self.f1(x, t, p)
        x1 = x + h * d1 * self.f2(v1, p)
        v2 = v1 + h * c2 * self.f1(x1, t + d1 * h, p)
        x2 = x1 + h * d2 * self.f2(v2, p)
        v_next = v2 + h * c3 * self.f1(x2, t + (d1+d2) * h, p)
        x_next = x2 + h * d3 * self.f2(v_next, p)
        return v_next, x_next