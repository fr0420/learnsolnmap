"""
Visualization utilities for scientific computing and analysis.

This package provides modular plotting functions organized by physical system:
- alphaparticle: α-particle dynamics in stellarators
- fpu: Fermi-Pasta-Ulam-Tsingou chain
- nco: Nonlinear coupled oscillators

Each system has its own visualizer class in the problems directory, following
the same organization as dataprocessing. MATLAB compatibility is available
for high-quality plotting when MATLAB is accessible.
"""

from .config import setup_matplotlib_style, get_color_palette, get_line_styles

# Import system-specific visualizers from problems directory
from .problems import (
    BaseVisualizer,
    AlphaParticleVisualizer,
    FPUVisualizer,
    NCOVisualizer,
)

# Import shared utilities
from . import utils

# Import MATLAB compatibility (optional)
try:
    from . import matlab_compat
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    matlab_compat = None

__all__ = [
    'setup_matplotlib_style',
    'get_color_palette', 
    'get_line_styles',
    'BaseVisualizer',
    'AlphaParticleVisualizer',
    'FPUVisualizer',
    'NCOVisualizer',
    'utils',
    'matlab_compat',
    'MATLAB_AVAILABLE'
]
