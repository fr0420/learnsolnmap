from integrators.integrator import FirstOrderODE, DynamicalODE
from integrators.standard import *
from integrators.symplectic import *


STANDARD_INTEGRATORS = {
    "ForwardEuler": ForwardEuler,
    "BackwardEuler": BackwardEuler,
    "ExplicitMidpoint": ExplicitMidpoint,
    "ImplicitMidpoint": ImplicitMidpoint,
    "RK3": RK3,
    "RK4": RK4,
}

SYMPLECTIC_INTEGRATORS = {
    "SymplecticEuler": SymplecticEuler,
    "SymplecticEuler2": SymplecticEuler2,
    "VelocityVerlet": VelocityVerlet,
    "PositionVerlet": PositionVerlet,
    "Ruth3": Ruth3,
}


def instantiate_first_order_ode_integrator(integrator_name, h, nsteps, f):
    """
    Instantiate a numerical integrator for a first-order ODE.
    """
    if integrator_name in STANDARD_INTEGRATORS:
        integrator_cls = STANDARD_INTEGRATORS[integrator_name]
        return integrator_cls(
            FirstOrderODE(f),
            h=h, 
            nsteps=nsteps
        )
    else:
        raise ValueError(f"Unknown integrator for first-order ODEs: {integrator_name}")


def identity(v):
    return v


def instantiate_dynamical_ode_integrator(integrator_name, h, nsteps, f1, f2=None):
    """
    Instantiate a numerical integrator for a dynamical ODE.
    """
    if f2 is None:
        f2 = identity

    if integrator_name in STANDARD_INTEGRATORS:
        integrator_cls = STANDARD_INTEGRATORS[integrator_name]
        return integrator_cls(
            FirstOrderODE.from_dynamical_ode(DynamicalODE(f1, f2)),
            h=h, 
            nsteps=nsteps
        )
    elif integrator_name in SYMPLECTIC_INTEGRATORS:
        integrator_cls = SYMPLECTIC_INTEGRATORS[integrator_name]
        return integrator_cls(
            DynamicalODE(f1, f2), 
            h=h, 
            nsteps=nsteps
        )
    else:
        raise ValueError(f"Unknown integrator for dynamical ODEs: {integrator_name}")
