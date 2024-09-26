"""
Numerical integrators (pytorch implementation) for solving ordinary differential equations.
"""

import torch


def instantiate_integrator(integrator_name, h, nsteps, f, **kwargs):
    """
    Instantiate an integrator by name.
    """
    if integrator_name == "ForwardEuler":
        return ForwardEuler(f, h, nsteps, **kwargs)
    elif integrator_name == "ExplicitMidpoint":
        return ExplicitMidpoint(f, h, nsteps, **kwargs)
    elif integrator_name == "RK3":
        return RK3(f, h, nsteps, **kwargs)
    elif integrator_name == "RK4":
        return RK4(f, h, nsteps, **kwargs)
    else:
        raise ValueError(f"Unknown integrator: {integrator_name}")


def instantiate_symplectic_integrator(integrator_name, h, nsteps, f1, f2=None, **kwargs):
    """
    Instantiate a symplectic integrator by name.
    """
    if f2 is None:
        f2 = identity
    if integrator_name == "SymplecticEuler":
        return SymplecticEuler(f1, f2, h, nsteps, **kwargs)
    elif integrator_name == "SymplecticEuler2":
        return SymplecticEuler2(f1, f2, h, nsteps, **kwargs)
    elif integrator_name == "VelocityVerlet":
        return VelocityVerlet(f1, f2, h, nsteps, **kwargs)
    elif integrator_name == "PositionVerlet":
        return PositionVerlet(f1, f2, h, nsteps, **kwargs)
    else:
        raise ValueError(f"Unknown integrator: {integrator_name}")
    

def identity(v):
    return v


class Integrator:
    """
    Solve an initial value problem for a first-order ODE.
        
        dx / dt = f(x, t)
        x(0) = x0
    """

    def __init__(self, f, h, nsteps=1):
        """
        Initialize the integrator.

        :param f: function to compute the right-hand side of the ODE
        :param h: stepsize
        :param nsteps: number of time steps, default is 1
        """
        if not callable(f):
            raise TypeError(f"Argument f should be callable, got {repr(type(f).__name__)}")
        self.f = f
        self.h = h
        self.nsteps = nsteps
        self.dt = h * nsteps
    
    def __repr__(self):
        """Return a string representation of the integrator."""
        return f"{self.__class__.__name__}(h={self.h}, nsteps={self.nsteps}, f={self.f.__name__})"
    
    def __call__(self, x0, t0=0.):
        """
        Integrate the ODE.

        :param x0: initial state (batch_size, dim)
        :param t0: initial time, default is 0.

        :returns: final state (batch_size, dim)
        """
        return self.integrate(x0, t0, self.nsteps)[0]

    def step(self, x, t):
        """
        Perform a single integration step.

        :param x: current state (batch_size, dim)
        :param t: current time

        :returns: next state (batch_size, dim)
        """
        raise NotImplementedError("Method 'step' not implemented.")
    
    def integrate(self, x0, t0, nsteps, retfull=False):
        """
        Integrate the ODE.

        :param x0: initial state (batch_size, dim)
        :param t0: initial time
        :param nsteps: number of time steps
        :param retfull: whether or not to return solutions at all time points

        :returns: sequence of states (nsteps + 1, batch_size, dim), sequence of times (nsteps + 1) if retfull is True, 
                  otherwise only the final state (batch_size, dim) and time
        """
        times = t0 + torch.arange(nsteps+1) * self.h

        if retfull:
            batch_size, dim = x0.shape
            trajectory = torch.zeros((nsteps+1, batch_size, dim))
            trajectory[0] = x0

            x = x0
            for i in range(nsteps): 
                x = self.step(x, times[i])  # x_{i+1} = step(x_i, t_i)
                trajectory[i+1] = x
            return trajectory, times
        
        else:
            x = x0
            for i in range(nsteps):
                x = self.step(x, times[i])  # x_{i+1} = step(x_i, t_i)
            return x, times[-1]

    
class ForwardEuler(Integrator):
    
    def step(self, x, t):
        x_next = x + self.h * self.f(x, t)
        return x_next


class ExplicitMidpoint(Integrator):

    def step(self, x, t):
        k1 = self.f(x, t)
        k2 = self.f(x + self.h * k1 / 2, t + self.h / 2)
        x_next = x + self.h * k2
        return x_next


class RK3(Integrator):

    def step(self, x, t):
        k1 = self.f(x, t)
        k2 = self.f(x + self.h * k1 / 2, t + self.h / 2)
        k3 = self.f(x - self.h * k1 + 2 * self.h * k2, t + self.h)
        x_next = x + self.h * (k1 / 6 + k2 * 2/3 + k3 / 6)
        return x_next
    

class RK4(Integrator):
    
    def step(self, x, t):
        k1 = self.f(x, t)
        k2 = self.f(x + self.h * k1 / 2, t + self.h / 2)
        k3 = self.f(x + self.h * k2 / 2, t + self.h / 2)
        k4 = self.f(x + self.h * k3, t + self.h)
        x_next = x + self.h * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
        return x_next
    

class SymplecticIntegrator:
    """
    Solve an initial value problem for dynamical ODEs of the form
        
        dv / dt = f1(x, t)
        dx / dt = f2(v)
        v(0) = v0
        x(0) = x0 
    """

    def __init__(self, f1, f2, h, nsteps=1):
        """
        Initialize the integrator.

        :param f1: function to compute the right-hand side of the ODE (dv / dt)
        :param f2: function to compute the right-hand side of the ODE (dx / dt)
        :param h: stepsize
        :param nsteps: number of time steps, default is 1
        """
        if not callable(f1):
            raise TypeError(f"Argument f1 should be callable, got {repr(type(f1).__name__)}")
        if not callable(f2):
            raise TypeError(f"Argument f2 should be callable, got {repr(type(f2).__name__)}")
        self.f1 = f1
        self.f2 = f2
        self.h = h
        self.nsteps = nsteps
        self.dt = h * nsteps

    def __repr__(self):
        """Return a string representation of the integrator."""
        return f"{self.__class__.__name__}(h={self.h}, nsteps={self.nsteps}, f1={self.f1.__name__}, f2={self.f2.__name__})"
    
    def __call__(self, u0, t0=0.):
        """
        Integrate the ODE given the concatenated initial state u0 = [v0, x0].
        
        :param u0: initial [v0, x0] (batch_size, 2*dim)
        :param t0: initial time, default is 0.
        
        :returns: final u = [v, x] (batch_size, 2*dim)
        """
        v0, x0 = u0.chunk(2, dim=-1)
        v, x, t = self.integrate(v0, x0, t0, self.nsteps)
        return torch.cat((v, x), dim=-1)
    
    def step(self, v, x, t):
        """
        Perform a single integration step.

        :param v: current v (batch_size, dim)
        :param x: current x (batch_size, dim)
        :param t: current time

        :returns: next v (batch_size, dim), next x (batch_size, dim)
        """
        raise NotImplementedError("Method 'step' not implemented.")
    
    def integrate(self, v0, x0, t0, nsteps, retfull=False):
        """
        Integrate the ODE.

        :param v0: initial v (batch_size, dim)
        :param x0: initial x (batch_size, dim)
        :param t0: initial time
        :param nsteps: number of time steps
        :param retfull: whether or not to return solutions at all time points

        :returns: sequence of v's (nsteps + 1, batch_size, dim), sequence of x's (nsteps + 1, batch_size, dim), sequence of times (nsteps + 1) 
                  if retfull is True, otherwise only the final v (batch_size, dim), x (batch_size, dim) and time
        """
        times = t0 + torch.arange(nsteps+1) * self.h

        if retfull:
            batch_size, dim = x0.shape
            trajectory_v = torch.zeros((nsteps+1, batch_size, dim))
            trajectory_x = torch.zeros((nsteps+1, batch_size, dim))
            trajectory_v[0] = v0
            trajectory_x[0] = x0

            v = v0
            x = x0
            for i in range(nsteps):
                v, x = self.step(v, x, times[i])  # v_{i+1}, x_{i+1} = step(v_i, x_i, t_i)
                trajectory_v[i+1] = v
                trajectory_x[i+1] = x
            return trajectory_v, trajectory_x, times

        else:
            v = v0
            x = x0
            for i in range(nsteps):
                v, x = self.step(v, x, times[i])
            return v, x, times[-1]


class SymplecticEuler(SymplecticIntegrator):
    
    def step(self, v, x, t):
        v_next = v + self.f1(x, t) * self.h
        x_next = x + self.f2(v_next) * self.h
        return v_next, x_next


class SymplecticEuler2(SymplecticIntegrator):
    
    def step(self, v, x, t):
        x_next = x + self.f2(v) * self.h
        v_next = v + self.f1(x_next, t + self.h) * self.h
        return v_next, x_next


class VelocityVerlet(SymplecticIntegrator):

    def step(self, v, x, t):
        v_half = v + 0.5 * self.f1(x, t) * self.h
        x_next = x + self.f2(v_half) * self.h
        v_next = v_half + 0.5 * self.f1(x_next, t + self.h) * self.h
        return v_next, x_next


class PositionVerlet(SymplecticIntegrator):
    
    def step(self, v, x, t):
        x_half = x + 0.5 * self.f2(v) * self.h
        v_next = v + self.f1(x_half, t + self.h / 2) * self.h
        x_next = x_half + 0.5 * self.f2(v_next) * self.h
        return v_next, x_next



if __name__ == "__main__":
    
    # Define a simple ODE: dx/dt = x, with analytical solution x(t) = x0 * exp(t)
    def f_exponential(x, t):
        return x
    
    def analytical_exponential(x0, t):
        return x0 * torch.exp(t)
    
    # Define a simple Hamiltonian system: harmonic oscillator
    # dv/dt = -x, dx/dt = v
    def f1_harmonic(x, t):
        return -x

    def f2_harmonic(v):
        return v

    # Analytical solution for harmonic oscillator
    def analytical_harmonic(v0, x0, t):
        x = x0 * torch.cos(t) + v0 * torch.sin(t)
        v = -x0 * torch.sin(t) + v0 * torch.cos(t)
        return v, x
    
    # Test the integrators
    x0 = torch.tensor([[1.], [2.]], dtype=torch.float64)
    v0 = torch.tensor([[0.], [0.]], dtype=torch.float64)
    u0 = torch.cat((v0, x0), dim=-1)
    t0 = torch.tensor(0.)
    h = 0.01
    nsteps = 10

    integrator = instantiate_integrator("ForwardEuler", h, nsteps, f_exponential)
    print(integrator)
    xf = integrator(x0, t0)
    expected_tf = t0 + nsteps * h
    expected_xf = analytical_exponential(x0, expected_tf)
    print(xf, expected_xf)
    assert torch.allclose(xf, expected_xf, atol=1e-2)

    # symplectic_integrator = instantiate_symplectic_integrator("VelocityVerlet", h, f1_harmonic, f2_harmonic)
    symplectic_integrator = instantiate_symplectic_integrator("VelocityVerlet", h, nsteps, f1_harmonic)
    print(symplectic_integrator)
    uf = symplectic_integrator(u0, t0)
    vf, xf = uf.chunk(2, dim=-1)
    expected_tf = t0 + nsteps * h
    expected_vf, expected_xf = analytical_harmonic(v0, x0, expected_tf)
    expected_uf = torch.cat((expected_vf, expected_xf), dim=-1)
    print(uf, expected_uf)
    assert torch.allclose(uf, expected_uf, atol=1e-2)
