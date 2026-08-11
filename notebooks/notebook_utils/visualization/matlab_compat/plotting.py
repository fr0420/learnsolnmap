"""
Simple MATLAB plotting functions for Python visualization.
"""

import os
import numpy as np
import matplotlib.colors as mcolors
import matlab
from typing import List, Tuple, Optional, Any, Union
from .engine import get_matlab_engine


def setup_matlab_figure(engine, figsize=None):
    """
    Setup MATLAB figure with basic configuration.
    
    Args:
        engine: MATLAB engine instance
        figsize: Optional tuple (width, height). If provided, sets initial size.
                If None, lets MATLAB use default size (will be adjusted before saving).
    """
    engine.figure(nargout=0)
    
    # Set figure units to inches for consistency
    engine.set(engine.gcf(), 'Units', 'inches', nargout=0)
    
    # Only set size if explicitly provided, otherwise let MATLAB choose
    if figsize is not None:
        width, height = figsize
        engine.set(engine.gcf(), 'Position', matlab.double([1, 1, width, height]), nargout=0)


def adjust_matlab_figure(engine, target_width=3.0):
    """
    Adjust MATLAB figure to have fixed width and auto height before saving.
    
    Args:
        engine: MATLAB engine instance
        target_width: Target width in inches (default: 3.0)
    """
    current_pos = engine.get(engine.gcf(), 'Position', nargout=1)[0]        
    current_width = current_pos[2]
    current_height = current_pos[3]
        
    print(f"Current figure size: {current_width:.1f}\" x {current_height:.1f}\"")
        
    # Calculate aspect ratio to maintain
    aspect_ratio = current_height / current_width
        
    # Calculate new height to maintain aspect ratio with fixed width
    new_height = target_width * aspect_ratio
        
    # Set new figure size
    engine.set(engine.gcf(), 'Position', matlab.double([1, 1, target_width, new_height]), nargout=0)

    # Verify the size was actually set
    verify_pos = engine.get(engine.gcf(), 'Position', nargout=1)[0]
    verify_width = verify_pos[2]
    verify_height = verify_pos[3]
    print(f"New figure size: {verify_width:.1f}\" x {verify_height:.1f}\"")

    # Set paper properties to match figure size for saving
    engine.set(engine.gcf(), 'PaperUnits', 'inches', nargout=0)
    engine.set(engine.gcf(), 'PaperPosition', matlab.double([0, 0, target_width, new_height]), nargout=0)
    engine.set(engine.gcf(), 'PaperSize', matlab.double([target_width, new_height]), nargout=0)


def save_matlab_figure(engine, save_path):
    """
    Save MATLAB figure in multiple formats.
    
    Args:
        engine: MATLAB engine instance
        save_path: Base path for saving files
    """
    if save_path and save_path.strip():
        try:
            # Determine extension from save_path
            base_name, ext = os.path.splitext(save_path)
            
            # Ensure we have a valid extension
            valid_extensions = ['.fig', '.pdf', '.png', '.jpg', '.jpeg', '.eps', '.svg']
            if not ext or ext.lower() not in valid_extensions:
                # Default to .pdf if no valid extension
                save_path = f"{save_path}.pdf"
                base_name, ext = os.path.splitext(save_path)
            
            # Save the main file with the specified extension
            engine.saveas(engine.gcf(), save_path, nargout=0)
            
            # Always save a .fig file (replace extension with .fig)
            fig_path = f"{base_name}.fig"
            if fig_path != save_path:  # Only save .fig if it's different from main file
                # For .fig files, ensure we're saving the current figure state
                engine.savefig(fig_path, nargout=0)
                print(f"Saved figure as: {save_path} and {fig_path}")
            else:
                print(f"Saved figure as: {save_path}")
                
        except Exception as e:
            print(f"Warning: Could not save figure to {save_path}: {e}")


def set_matlab_axis_properties(engine, axis_lims=None, axis_labels=None, title=None):
    """
    Set common MATLAB axis properties.
    
    Args:
        engine: MATLAB engine instance
        axis_lims: Optional tuple of (x_lims, y_lims)
        axis_labels: Optional list of [xlabel, ylabel]
        title: Optional title string
    """
    # Set axis properties
    engine.axis('equal', nargout=0)
    if axis_lims:
        engine.xlim(matlab.double(list(axis_lims[0])), nargout=0)
        engine.ylim(matlab.double(list(axis_lims[1])), nargout=0)
    
    # Set labels and title with LaTeX interpreter
    if axis_labels:
        engine.xlabel(axis_labels[0], 'Interpreter', 'latex', nargout=0)
        engine.ylabel(axis_labels[1], 'Interpreter', 'latex', nargout=0)
    if title:
        engine.title(title, 'Interpreter', 'latex', nargout=0)
    
    # Add grid
    engine.grid('on', nargout=0)


def convert_linestyle_to_matlab(linestyle):
    """
    Convert matplotlib linestyle to MATLAB format.
    
    Args:
        linestyle: matplotlib linestyle string
        
    Returns:
        MATLAB linestyle string
    """
    linestyle_map = {
        '-': '-',      # solid line
        '--': '--',    # dashed line
        '-.': '-.',    # dash-dot line
        ':': ':',      # dotted line
        '': 'none',        # no line (markers only)
        ' ': 'none',       # no line (markers only)
        'None': 'none',    # no line (markers only)
        'solid': '-',
        'dashed': '--',
        'dashdot': '-.',
        'dotted': ':'
    }
    return linestyle_map.get(linestyle, '-')


def convert_color_to_matlab(color):
    """
    Convert matplotlib color to MATLAB RGB format.
    
    Args:
        color: matplotlib color (string, tuple, list, etc.)
        
    Returns:
        List of RGB values in [0, 1] range
    """
    # Convert to RGB if it's not already
    if isinstance(color, str):
        # Handle matplotlib color names and hex strings
        try:
            rgb = mcolors.to_rgb(color)
        except ValueError:
            # Fallback to a default color
            rgb = (0.0, 0.0, 0.0)  # black
    elif isinstance(color, (list, tuple, np.ndarray)):
        # Assume it's already RGB-like
        if len(color) >= 3:
            rgb = tuple(color[:3])  # Take only RGB components
        else:
            rgb = (0.0, 0.0, 0.0)  # Fallback
    else:
        rgb = (0.0, 0.0, 0.0)  # Fallback
    
    # Ensure values are in [0, 1] range
    rgb = [max(0.0, min(1.0, float(c))) for c in rgb]
    return rgb


