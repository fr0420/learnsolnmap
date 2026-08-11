"""
α-Particle dynamics visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Any, Dict
from itertools import cycle
from .base import BaseVisualizer

# Import MATLAB compatibility
try:
    from ..matlab_compat.engine import get_matlab_engine
    from ..matlab_compat.plotting import *
    import matlab
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    get_matlab_engine = None
    setup_matlab_figure = None
    save_matlab_figure = None
    set_matlab_axis_properties = None
    convert_linestyle_to_matlab = None
    convert_color_to_matlab = None


class AlphaParticleVisualizer(BaseVisualizer):
    """
    Visualization utilities for α-particle dynamics in stellarators.
    """
    
    def __init__(self):
        """Initialize the α-particle visualizer."""
        super().__init__("alphaparticle")
        self._setup_alphaparticle_styles()
    
    def _setup_alphaparticle_styles(self):
        """Setup α-particle specific plotting styles."""
        self.magnetic_field_cmap = 'coolwarm'
        self.contour_alpha = 0.1
    
    def plot_xy(self, trajectories: List[Any], **kwargs) -> None:
        """
        Plot α-particle trajectories in (x,y) plane with magnetic field contours.
        
        Args:
            trajectories: List of trajectory objects to plot
            **kwargs: Additional plotting parameters including:
                - axis_lims: Tuple of (x_lims, y_lims) for axis ranges
                - linestyles: List of line styles for each trajectory
                - colors: List of colors for each trajectory
                - markersize: Size of trajectory markers
                - legend_labels: Labels for the legend
                - axis_labels: Labels for x and y axes
                - title: Plot title
                - figsize: Figure size tuple
                - show_legend: Whether to display legend
                - save_path: Path to save the figure
                - aspect_ratio: Aspect ratio for the plot
        """
        # Get common parameters
        params = self._get_plotting_parameters(**kwargs)
        
        # Extract method-specific parameters
        axis_lims = kwargs.get('axis_lims', [(None, None)] * 2)
        markersize = kwargs.get('markersize', 0.5)
        axis_labels = kwargs.get('axis_labels', ['$x$', '$y$'])
        aspect_ratio = kwargs.get('aspect_ratio', 'equal')
        
        # Set defaults for common parameters if not provided
        if params['legend_labels'] is None:
            params['legend_labels'] = [f'Trajectory {i+1}' for i in range(len(trajectories))]
        if params['linestyles'] == self.default_line_styles:
            params['linestyles'] = ['-'] * len(trajectories)
        
        # Handle single linestyle
        if isinstance(params['linestyles'], str):
            params['linestyles'] = [params['linestyles']] * len(trajectories)
        
        # Create figure
        fig = plt.figure(figsize=params['figsize'], layout='constrained')
        ax = fig.subplots(1, 1)
        
        # Plot trajectories
        color_cycle = cycle(params['colors'])
        lines = []
        
        for traj, linestyle, legend_label in zip(trajectories, params['linestyles'], params['legend_labels']):
            u = traj.states.u 
            x, y = u[:, 2], u[:, 3]  # Extract x, y coordinates
            
            if legend_label == 'ref':
                line, = ax.plot(x, y, linestyle, markersize=markersize, 
                              color='k', alpha=0.3, label=legend_label)
            else:
                line, = ax.plot(x, y, linestyle, markersize=markersize, 
                              color=next(color_cycle), label=legend_label)
            lines.append(line)
        
        # Add magnetic field contours
        if trajectories:
            # Use the first trajectory's processor to compute magnetic field
            traj = trajectories[0]
            xx, yy = np.meshgrid(
                np.linspace(*axis_lims[0], 100), 
                np.linspace(*axis_lims[1], 100)
            )
            B_values = traj.states.processor.compute_B(xx, yy)
            ax.contour(xx, yy, B_values, levels=20, cmap=self.magnetic_field_cmap, 
                      alpha=self.contour_alpha)
        
        # Set axis properties
        if axis_lims is not None:
            ax.set_xlim(axis_lims[0])
            ax.set_ylim(axis_lims[1])
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_aspect(aspect_ratio)
        
        if params['title']:
            ax.set_title(params['title'])
        
        # Add legend
        self._create_legend(fig, lines, params['legend_labels'], params['show_legend'])
        
        # Save figure
        self._save_figure(fig, params['save_path'], params['dpi'])
        
        if not params['save_path']:
            plt.show()
    
    def plot_xy_matlab(self, trajectories: List[Any], **kwargs) -> None:
        """
        Plot α-particle trajectories using MATLAB backend for high-quality output.
        
        Args:
            trajectories: List of trajectory objects to plot
            **kwargs: Additional plotting parameters including:
                - axis_lims: Tuple of (x_lims, y_lims) for axis ranges
                - linestyles: List of line styles for each trajectory
                - colors: List of colors for each trajectory
                - marker: Marker style (default: '+')
                - markersize: Size of trajectory markers
                - axis_labels: Labels for x and y axes
                - title: Plot title
                - figsize: Figure size tuple
                - save_path: Path to save the figure
                - show_magnetic_field: Whether to show magnetic field contours
        """
        if not MATLAB_AVAILABLE:
            print("MATLAB not available, falling back to matplotlib")
            return self.plot_xy(trajectories, **kwargs)
        
        # Get common parameters
        params = self._get_plotting_parameters(**kwargs)
        
        # Extract method-specific parameters
        axis_lims = kwargs.get('axis_lims', [(1, 4.5), (1, 4.5)])
        marker = kwargs.get('marker', '+')
        markersize = kwargs.get('markersize', 0.5)
        axis_labels = kwargs.get('axis_labels', ['$x$', '$y$'])
        show_magnetic_field = kwargs.get('show_magnetic_field', True)
        
        # Set defaults for common parameters if not provided
        if params['legend_labels'] is None:
            params['legend_labels'] = [f'Trajectory {i+1}' for i in range(len(trajectories))]
        if params['linestyles'] == self.default_line_styles:
            params['linestyles'] = ['-'] * len(trajectories)
        
        # Handle single linestyle
        if isinstance(params['linestyles'], str):
            params['linestyles'] = [params['linestyles']] * len(trajectories)
        
        # Get MATLAB engine
        engine = get_matlab_engine()
        
        # Setup figure
        setup_matlab_figure(engine, params['figsize'])
        
        # Plot magnetic field contours if requested and available
        if show_magnetic_field and trajectories and hasattr(trajectories[0], 'states') and hasattr(trajectories[0].states, 'processor'):
            try:
                xx, yy = np.meshgrid(
                    np.linspace(axis_lims[0][0], axis_lims[0][1], 100), 
                    np.linspace(axis_lims[1][0], axis_lims[1][1], 100)
                )
                B_values = trajectories[0].states.processor.compute_B(xx, yy)
                engine.contour(xx, yy, B_values, 20, 'LineWidth', 0.5, 'EdgeAlpha', 0.1, nargout=0)
                engine.colormap('nebula', nargout=0)
            except:
                pass
        
        # Hold on for multiple trajectories
        engine.hold('on', nargout=0)
        
        # Plot each trajectory
        color_cycle = cycle(params['colors'])
        
        for i, (traj, linestyle, legend_label) in enumerate(zip(trajectories, params['linestyles'], params['legend_labels'])):
            # Extract x, y coordinates
            u = traj.states.u if hasattr(traj, 'states') else traj
            x, y = u[:, 2], u[:, 3]
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)

            # Get color
            color = next(color_cycle)
            
            # Convert color to MATLAB RGB format
            matlab_color = convert_color_to_matlab(color)
            
            # Convert matplotlib linestyle to MATLAB format
            matlab_linestyle = convert_linestyle_to_matlab(linestyle)

            # Plot trajectory with linestyle and markers
            engine.plot(x, y, 'LineStyle', matlab_linestyle, 'LineWidth', 0.1, 'Marker', marker, 
            'MarkerSize', markersize, 'Color', matlab.double(matlab_color), 'MarkerFaceColor', matlab.double(matlab_color), nargout=0)
        
        # Set axis properties using helper function
        set_matlab_axis_properties(engine, axis_lims, axis_labels, params['title'])
        
        # Adjust figure size before saving
        adjust_matlab_figure(engine)

        # Save figure using helper function
        save_matlab_figure(engine, params['save_path'])

    
    def plot_history(self, trajectories: List[Any], **kwargs) -> None:
        """
        Plot time series of α-particle trajectory components.
        
        Args:
            trajectories: List of trajectory objects to plot
            **kwargs: Additional plotting parameters including:
                - func_dict: Dictionary of functions to plot (e.g., {'vx': lambda u: u[:, 0]})
                - linestyles: List of line styles for each trajectory
                - colors: List of colors for each trajectory
                - legend_labels: Labels for the legend
                - xlabel: Label for x-axis
                - ylabels: Labels for y-axes
                - ylims: List of y-axis limits for each subplot
                - figsize: Figure size tuple
                - log_yscale: Whether to use log scale for y-axis
                - log_xscale: Whether to use log scale for x-axis
                - show_legend: Whether to display legend
                - save_path: Path to save the figure
        """
        # Get common parameters
        params = self._get_plotting_parameters(**kwargs)
        
        # Extract method-specific parameters
        func_dict = kwargs.get('func_dict', {
            'vx': lambda u: u[:, 0], 
            'vy': lambda u: u[:, 1],
            'x': lambda u: u[:, 2], 
            'y': lambda u: u[:, 3]
        })
        xlabel = kwargs.get('xlabel', 't')
        ylabels = kwargs.get('ylabels', list(func_dict.keys()))
        ylims = kwargs.get('ylims', [(None, None)] * len(func_dict))
        log_yscale = kwargs.get('log_yscale', False)
        log_xscale = kwargs.get('log_xscale', False)
        
        # Set defaults for common parameters if not provided
        if params['legend_labels'] is None:
            params['legend_labels'] = [f'Trajectory {i+1}' for i in range(len(trajectories))]
        if params['linestyles'] == self.default_line_styles:
            params['linestyles'] = ['-'] * len(trajectories)
        
        # Set dynamic figsize based on number of functions
        if params['figsize'] == (8, 6):  # Default from base class
            params['figsize'] = (8, 1.5 * len(func_dict))
        
        # Handle single linestyle
        if isinstance(params['linestyles'], str):
            params['linestyles'] = [params['linestyles']] * len(trajectories)
        
        # Create figure
        fig, axes = plt.subplots(len(func_dict), 1, figsize=params['figsize'], layout='constrained')
        if len(func_dict) == 1:
            axes = [axes]
        
        # Plot trajectories
        color_cycle = cycle(params['colors'])
        lines = []
        
        for traj, linestyle, legend_label in zip(trajectories, params['linestyles'], params['legend_labels']):
            xdata = traj.times if hasattr(traj, 'times') else np.arange(len(traj))
            color = next(color_cycle)
            
            for i, (key, func) in enumerate(func_dict.items()):
                if legend_label == 'ref':
                    line, = axes[i].plot(xdata, func(traj.states.u), linestyle, 
                                       lw=1, alpha=0.2, color='k', label=legend_label)
                else:
                    line, = axes[i].plot(xdata, func(traj.states.u), linestyle, 
                                       lw=1, color=color, label=legend_label)
            lines.append(line)
        
        # Set axis properties
        for i, ax in enumerate(axes):
            ax.set_ylim(ylims[i])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabels[i])
            if log_yscale:
                ax.set_yscale('log')
            if log_xscale:
                ax.set_xscale('log')
        
        # Add legend
        self._create_legend(fig, lines, params['legend_labels'], params['show_legend'])
        
        # Save figure
        self._save_figure(fig, params['save_path'], params['dpi'])
        
        if not params['save_path']:
            plt.show()
    
    def plot_history_matlab(self, trajectories: List[Any], **kwargs) -> None:
        """
        Plot time series of α-particle trajectory components using MATLAB backend.
        
        Args:
            trajectories: List of trajectory objects to plot
            **kwargs: Additional plotting parameters including:
                - func_dict: Dictionary of functions to plot (e.g., {'vx': lambda u: u[:, 0]})
                - linestyles: List of line styles for each trajectory
                - colors: List of colors for each trajectory
                - legend_labels: Labels for the legend
                - xlabel: Label for x-axis
                - ylabels: Labels for y-axes
                - ylims: List of y-axis limits for each subplot
                - figsize: Figure size tuple
                - log_yscale: Whether to use log scale for y-axis
                - log_xscale: Whether to use log scale for x-axis
                - show_legend: Whether to display legend
                - save_path: Path to save the figure
        """
        if not MATLAB_AVAILABLE:
            print("MATLAB not available, falling back to matplotlib")
            return self.plot_history(trajectories, **kwargs)
        
        # Get common parameters
        params = self._get_plotting_parameters(**kwargs)
        
        # Extract method-specific parameters
        func_dict = kwargs.get('func_dict', {
            'vx': lambda u: u[:, 0], 
            'vy': lambda u: u[:, 1],
            'x': lambda u: u[:, 2], 
            'y': lambda u: u[:, 3]
        })
        xlabel = kwargs.get('xlabel', 't')
        ylabels = kwargs.get('ylabels', list(func_dict.keys()))
        ylims = kwargs.get('ylims', [(None, None)] * len(func_dict))
        log_yscale = kwargs.get('log_yscale', False)
        log_xscale = kwargs.get('log_xscale', False)
        
        # Set defaults for common parameters if not provided
        if params['legend_labels'] is None:
            params['legend_labels'] = [f'Trajectory {i+1}' for i in range(len(trajectories))]
        if params['linestyles'] == self.default_line_styles:
            params['linestyles'] = ['-'] * len(trajectories)
        
        # Set dynamic figsize based on number of functions
        if params['figsize'] == (8, 6):  # Default from base class
            params['figsize'] = (8, 2 * len(func_dict))
        
        # Handle single linestyle
        if isinstance(params['linestyles'], str):
            params['linestyles'] = [params['linestyles']] * len(trajectories)
        
        # Get MATLAB engine
        engine = get_matlab_engine()
        
        # Setup figure
        setup_matlab_figure(engine, params['figsize'])
        
        # Create tiledlayout (num_subplots rows, 1 column)
        num_subplots = len(func_dict)
        tcl = engine.tiledlayout(matlab.double(num_subplots), matlab.double(1), nargout=1)
        
        # Create all tiles first
        axes = []
        for i in range(num_subplots):
            ax = engine.nexttile(tcl, nargout=1)
            engine.hold(ax, 'on', nargout=0)
            axes.append(ax)
        
        # Create color cycle
        color_cycle = cycle(params['colors'])

        # Plot trajectories and collect handles
        legend_handles = []
        num_trajectories = len(trajectories)
        
        for traj_idx, (traj, linestyle, legend_label) in enumerate(zip(trajectories, params['linestyles'], params['legend_labels'])):
            xdata = traj.times if hasattr(traj, 'times') else np.arange(len(traj))
            color = next(color_cycle)
            
            # Ensure xdata is contiguous
            xdata = np.ascontiguousarray(xdata)
            
            # Convert color and linestyle to MATLAB format
            matlab_color = convert_color_to_matlab(color)
            matlab_linestyle = convert_linestyle_to_matlab(linestyle)
            
            for i, (key, func) in enumerate(func_dict.items()):
                # Use the pre-created tile
                ax = axes[i]
                
                # Get y data and ensure it's contiguous
                ydata = func(traj.states.u)
                ydata = np.ascontiguousarray(ydata)
                
                # Handle reference trajectory styling
                if legend_label == 'ref':
                    # Use black color with transparency for reference
                    ref_color = [0.0, 0.0, 0.0]  # Black
                    linewidth = 1.0
                    alpha = 0.2
                else:
                    ref_color = matlab_color
                    linewidth = 1.0
                    alpha = 1.0
                
                # Plot the line
                handle = engine.plot(ax, xdata, ydata, 'LineStyle', matlab_linestyle, 'LineWidth', linewidth, 
                                   'Color', matlab.double(ref_color), 'DisplayName', legend_label, nargout=1)
                
                # Store handles only from first subplot for legend
                if i == 0:
                    legend_handles.append(handle)
                
                # Set axis properties for this subplot (only on first trajectory)
                if traj_idx == 0:
                    if ylims[i] is not None and all(v is not None for v in ylims[i]):
                        engine.ylim(ax, matlab.double(ylims[i]), nargout=0)
                    
                    # Set labels
                    engine.xlabel(ax, xlabel, 'Interpreter', 'latex', nargout=0)
                    engine.ylabel(ax, ylabels[i], 'Interpreter', 'latex', nargout=0)
                    
                    # Set log scales
                    if log_yscale:
                        engine.set(ax, 'YScale', 'log', nargout=0)
                    if log_xscale:
                        engine.set(ax, 'XScale', 'log', nargout=0)
                    
                    # Add grid
                    engine.grid(ax, 'on', nargout=0)
        
        # Add title if provided
        if params['title']:
            engine.suptitle(params['title'], 'Interpreter', 'latex', nargout=0)
        
        print(f"legend_handles: {legend_handles}")
        print(len(legend_handles))

        # Add legend if requested
        if params['show_legend'] and legend_handles:
            # Store handles individually in MATLAB workspace
            for i, handle in enumerate(legend_handles):
                engine.workspace[f'h_{i+1}'] = handle
            
            # Create legend using eval with individual handles
            handle_list = ', '.join([f'h_{i+1}' for i in range(len(legend_handles))])
            engine.eval(f"leg = legend([{handle_list}]);", nargout=0)
            
            # Set layout using eval
            engine.eval("leg.Layout.Tile = 'south';", nargout=0)
        
        # Adjust figure size before saving
        adjust_matlab_figure(engine)

        # Save figure using helper function
        save_matlab_figure(engine, params['save_path'])
    
    def plot_errors(self, trajectories: List[Any], reference: Any, **kwargs) -> None:
        """
        Plot error analysis comparing trajectories to reference solution.
        
        Args:
            trajectories: List of trajectory objects to compare
            reference: Reference trajectory object
            **kwargs: Additional plotting parameters including:
                - error_names: List of error types to plot
                - linestyles: List of line styles for each trajectory
                - colors: List of colors for each trajectory
                - legend_labels: Labels for the legend
                - titles: List of titles for each subplot
                - xlim: X-axis limits
                - ylims: List of y-axis limits for each subplot
                - ylabels: Labels for y-axes
                - figsize: Figure size tuple
                - log_yscale: Whether to use log scale for y-axis
                - log_xscale: Whether to use log scale for x-axis
                - show_legend: Whether to display legend
                - save_path: Path to save the figure
        """
        # Get common parameters
        params = self._get_plotting_parameters(**kwargs)
        
        # Extract method-specific parameters
        error_names = kwargs.get('error_names', ["abs_traj_err", "abs_H_err"])
        xlim = kwargs.get('xlim', None)
        ylims = kwargs.get('ylims', [(1e-5, 1e1)] * len(error_names))
        ylabels = kwargs.get('ylabels', error_names)
        log_yscale = kwargs.get('log_yscale', True)
        log_xscale = kwargs.get('log_xscale', False)
        
        # Set defaults for common parameters if not provided
        if params['legend_labels'] is None:
            params['legend_labels'] = [f'Trajectory {i+1}' for i in range(len(trajectories))]
        if params['linestyles'] == self.default_line_styles:
            params['linestyles'] = ['-'] * len(trajectories)
        
        # Set dynamic figsize based on number of error types
        if params['figsize'] == (8, 6):  # Default from base class
            params['figsize'] = (8, 1.5 * len(error_names))
        
        # Handle single reference trajectory
        if not isinstance(reference, list):
            reference = [reference] * len(trajectories)
        
        # Create figure
        fig, axes = plt.subplots(len(error_names), 1, figsize=params['figsize'], layout='constrained')
        if len(error_names) == 1:
            axes = [axes]
        
        # Plot errors
        color_cycle = cycle(params['colors'])
        lines = []
        
        for traj, ref_traj, linestyle, legend_label in zip(trajectories, reference, params['linestyles'], params['legend_labels']):
            tt, errors = traj.compare(ref_traj)
            color = next(color_cycle)
            
            for i, error_name in enumerate(error_names):
                line, = axes[i].plot(tt[1:], errors[error_name][1:], linestyle, 
                                   lw=1, color=color, label=legend_label)
                print(f"{legend_label} n=1 {error_name} = {errors[error_name][1]}")
                print(f"{legend_label} n={len(tt)-1} {error_name} = {errors[error_name][-1]}")
            lines.append(line)
        
        # Set axis properties
        for i, ax in enumerate(axes):
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.set_ylim(ylims[i])
            ax.set_xlabel('$t$')
            ax.set_ylabel(ylabels[i])
            if log_yscale:
                ax.set_yscale('log')
            if log_xscale:
                ax.set_xscale('log')
        
        # Add legend
        self._create_legend(fig, lines, params['legend_labels'], params['show_legend'])
        
        # Save figure
        self._save_figure(fig, params['save_path'], params['dpi'])
        
        if not params['save_path']:
            plt.show()
