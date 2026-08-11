# Core classes
from .core import States, Dataset, Trajectory

# Utility functions
from .utils import smooth_data, mmd, is_multiple, find_quotient

# Problem classes
from .problems import (
    BaseProblem, 
    Lorenz, LorenzDataset, LorenzTrajectory,
    ThreeBody, ThreeBodyDataset, ThreeBodyTrajectory,
    ThreeBody2D, ThreeBody2DDataset, ThreeBody2DTrajectory,
    FPU, FPUDataset, FPUTrajectory,
    NCO, NCODataset, NCOTrajectory,
    AlphaParticle, AlphaParticleDataset, AlphaParticleTrajectory, poincare_section, batch_poincare_section
)

__all__ = [
    # Core classes
    'States',
    'Dataset', 
    'Trajectory',
    
    # Utility functions
    'smooth_data',
    'mmd',
    'is_multiple',
    'find_quotient',
    
    # Problem classes
    'BaseProblem',
    'Lorenz', 'LorenzDataset', 'LorenzTrajectory',
    'ThreeBody', 'ThreeBodyDataset', 'ThreeBodyTrajectory',
    'ThreeBody2D', 'ThreeBody2DDataset', 'ThreeBody2DTrajectory',
    'FPU', 'FPUDataset', 'FPUTrajectory',
    'NCO', 'NCODataset', 'NCOTrajectory',
    'AlphaParticle', 'AlphaParticleDataset', 'AlphaParticleTrajectory', 'poincare_section', 'batch_poincare_section',
]
