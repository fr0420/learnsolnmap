import numpy as np 
import torch


class VelocityVerlet:
    '''
    Solve an initial value problem for a second-order ODE on [0, T] by velocity Verlet algorithm.
        
        d^2 x / dt^2 = A(x)
        x(0) = x0
        dx(0) / dt  = v0 
        
    '''
    
    def __init__(self, A, T, nsteps=1, retfull=False):
        '''
        :param A: right-hand side of the ODE
        :param T: end time 
        :param nsteps: number of time steps 
        :param retfull: whether or not to return solutions at all time points
        '''
        self.A = A
        
        self.T = T
        self.nsteps = nsteps 
        self.h = self.T / self.nsteps
        
        self.retfull = retfull
        
    def solve(self, v0, x0):
        '''
        Integrate the ODE from 0 to T given the initial state (v0, x0). 
        '''
        
        # Initialize solution 
        x = x0
        v = v0
        res = [(v0, x0)]
        
        # Integrate using velocity Verlet 
        for i in range(self.nsteps): 
            v_mid = v + 0.5 * self.h * self.A(x)
            x = x + v_mid * self.h
            v = v_mid + 0.5 * self.h * self.A(x)
            res.append((v, x))
        
        # Return solution 
        if self.retfull:
            return res
        else:
            return v, x
        