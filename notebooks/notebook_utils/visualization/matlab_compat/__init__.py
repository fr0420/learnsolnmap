"""
MATLAB compatibility layer for visualization utilities.

This module provides MATLAB engine integration for high-quality plotting
while maintaining compatibility with Python matplotlib fallbacks.
"""

from .engine import get_matlab_engine, is_matlab_available
from .plotting import setup_matlab_figure, adjust_matlab_figure, save_matlab_figure, set_matlab_axis_properties, convert_linestyle_to_matlab, convert_color_to_matlab
from .styles import get_matlab_defaults, apply_matlab_style, set_matlab_colors

__all__ = [
    'get_matlab_engine', 
    'is_matlab_available',
    'setup_matlab_figure',
    'adjust_matlab_figure',
    'save_matlab_figure', 
    'set_matlab_axis_properties',
    'convert_linestyle_to_matlab',
    'convert_color_to_matlab',
    'get_matlab_defaults',
    'apply_matlab_style',
    'set_matlab_colors'
]
