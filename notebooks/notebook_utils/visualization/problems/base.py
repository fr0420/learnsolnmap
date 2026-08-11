"""
Base visualization class for physical systems.

This module provides the base class that all system-specific visualizers inherit from,
ensuring consistent interface and shared functionality across different physical systems.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import matplotlib.pyplot as plt


class BaseVisualizer(ABC):
    """
    Base class for system-specific visualization utilities.
    
    This class provides a common interface for visualization functions across
    different physical systems, ensuring consistency and reusability.
    """
    
    def __init__(self, system_name: str):
        """
        Initialize the base visualizer.
        
        Args:
            system_name: Name of the physical system (e.g., 'alphaparticle', 'fpu')
        """
        self.system_name = system_name
        self._setup_default_styles()
    
    def _setup_default_styles(self):
        """Setup default plotting styles for the system."""
        self.default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.default_line_styles = ['-', '--', '-.', ':']
        self.default_marker_styles = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
    
    def plot_history(self, trajectories: List[Any], **kwargs) -> None:
        """
        Plot time series of trajectory components.
        
        Args:
            trajectories: List of trajectory objects to plot
            **kwargs: Additional plotting parameters
        """
        pass
    
    def plot_errors(self, trajectories: List[Any], reference: Any, **kwargs) -> None:
        """
        Plot error analysis comparing trajectories to reference.
        
        Args:
            trajectories: List of trajectory objects to compare
            reference: Reference trajectory object
            **kwargs: Additional plotting parameters
        """
        # Default implementation - can be overridden by subclasses
        pass
    
    def _get_plotting_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Extract and validate common plotting parameters from kwargs.
        
        Returns:
            Dictionary of validated plotting parameters
        """
        params = {
            'figsize': kwargs.get('figsize', (8, 6)),
            'colors': kwargs.get('colors', self.default_colors),
            'linestyles': kwargs.get('linestyles', self.default_line_styles),
            'legend_labels': kwargs.get('legend_labels', None),
            'title': kwargs.get('title', None),
            'show_legend': kwargs.get('show_legend', True),
            'save_path': kwargs.get('save_path', None),
            'dpi': kwargs.get('dpi', 300)
        }
        return params
    
    def _save_figure(self, fig: plt.Figure, save_path: Optional[str], dpi: int = 300) -> None:
        """
        Save figure to file if save_path is provided.
        
        Args:
            fig: Matplotlib figure object
            save_path: Path to save the figure
            dpi: Resolution for saved figure
        """
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved as '{save_path}'")
    
    def _create_legend(self, fig: plt.Figure, lines: List, labels: List[str], 
                      show_legend: bool = True) -> None:
        """
        Create legend for the figure.
        
        Args:
            fig: Matplotlib figure object
            lines: List of line objects for legend
            labels: List of labels for legend
            show_legend: Whether to show the legend
        """
        if show_legend and lines and labels:
            fig.legend(lines, labels, loc='outside center right', borderaxespad=0.1)
