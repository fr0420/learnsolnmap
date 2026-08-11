"""
Nonlinear coupled oscillators (NCO) visualization utilities.

This module provides specialized plotting functions for NCO dynamics,
including phase space visualization and synchronization analysis.
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


class NCOVisualizer(BaseVisualizer):
    """
    Visualization utilities for nonlinear coupled oscillators.
    
    This class provides specialized plotting functions for NCO trajectories,
    including phase space visualization and synchronization analysis.
    """
    
    def __init__(self):
        """Initialize the NCO visualizer."""
        super().__init__("nco")
        self._setup_nco_styles()
    
    def _setup_nco_styles(self):
        """Setup NCO specific plotting styles."""
        self.nco_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_phase_space(self, trajectories: List[Any], **kwargs) -> None:
        """
        Plot NCO trajectories in phase space (q1,p1) and (q2,p2).
        
        Args:
            trajectories: List of trajectory objects to plot
            **kwargs: Additional plotting parameters including:
                - axis_lims: List of axis limits for each oscillator [(q1_lims, p1_lims), (q2_lims, p2_lims)]
                - linestyles: List of line styles for each trajectory
                - colors: List of colors for each trajectory
                - legend_labels: Labels for the legend
                - axis_labels: Labels for axes [(q1_label, p1_label), (q2_label, p2_label)]
                - titles: List of titles for each subplot
                - figsize: Figure size tuple
                - show_legend: Whether to display legend
                - save_path: Path to save the figure
        """
        # Get common parameters
        params = self._get_plotting_parameters(**kwargs)
        
        # Extract method-specific parameters
        axis_lims = kwargs.get('axis_lims', None)
        axis_labels = kwargs.get('axis_labels', [(r'$q_1$', r'$p_1$'), (r'$q_2$', r'$p_2$')])
        titles = kwargs.get('titles', ['Oscillator 1', 'Oscillator 2'])
        
        # Set defaults for common parameters if not provided
        if params['legend_labels'] is None:
            params['legend_labels'] = [f'Trajectory {i+1}' for i in range(len(trajectories))]
        if params['linestyles'] == self.default_line_styles:
            params['linestyles'] = ['-'] * len(trajectories)
        
        # Handle single linestyle
        if isinstance(params['linestyles'], str):
            params['linestyles'] = [params['linestyles']] * len(trajectories)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=params['figsize'], layout='constrained')
        
        # Plot trajectories
        color_cycle = cycle(params['colors'])
        lines = []
        
        for traj, linestyle, legend_label in zip(trajectories, params['linestyles'], params['legend_labels']):
            p1, p2, q1, q2 = traj.states.get_pq()
            
            if legend_label in ['ref', '$\phi$']:
                line1, = axes[0].plot(q1, p1, linestyle, color='k', alpha=0.3, label=legend_label)
                axes[1].plot(q2, p2, linestyle, color='k', alpha=0.3)
            else:
                color = next(color_cycle)
                line1, = axes[0].plot(q1, p1, linestyle, color=color, label=legend_label)
                axes[1].plot(q2, p2, linestyle, color=color)
            lines.append(line1)
        
        # Set labels and titles for each subplot
        for i, ax in enumerate(axes):
            if axis_lims is not None:
                ax.set_xlim(axis_lims[i][0])
                ax.set_ylim(axis_lims[i][1])
            ax.set_xlabel(axis_labels[i][0])
            ax.set_ylabel(axis_labels[i][1])
            ax.set_title(titles[i])
        
        # Add legend
        self._create_legend(fig, lines, params['legend_labels'], params['show_legend'])
        
        # Save figure
        self._save_figure(fig, params['save_path'], params['dpi'])
        
        if not params['save_path']:
            plt.show()
    
    def plot_phase_space_matlab(self, trajectories: List[Any], **kwargs) -> None:
        """
        Plot NCO trajectories in phase space using MATLAB backend for high-quality output.
        
        Args:
            trajectories: List of trajectory objects to plot
            **kwargs: Additional plotting parameters including:
                - axis_lims: List of axis limits for each oscillator [(q1_lims, p1_lims), (q2_lims, p2_lims)]
                - linestyles: List of line styles for each trajectory
                - colors: List of colors for each trajectory
                - legend_labels: Labels for the legend
                - axis_labels: Labels for axes [(q1_label, p1_label), (q2_label, p2_label)]
                - titles: List of titles for each subplot
                - figsize: Figure size tuple
                - save_path: Path to save the figure
        """
        if not MATLAB_AVAILABLE:
            print("MATLAB not available, falling back to matplotlib")
            return self.plot_phase_space(trajectories, **kwargs)
        
        # Get common parameters
        params = self._get_plotting_parameters(**kwargs)
        
        # Extract method-specific parameters
        axis_lims = kwargs.get('axis_lims', None)
        axis_labels = kwargs.get('axis_labels', [(r'$q_1$', r'$p_1$'), (r'$q_2$', r'$p_2$')])
        titles = kwargs.get('titles', ['Oscillator 1', 'Oscillator 2'])
        
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
        
        # Setup figure with 2 subplots (1x2 grid)
        setup_matlab_figure(engine, params['figsize'])
        
        # Create tiledlayout (1 row, 2 columns)
        tcl = engine.tiledlayout(matlab.double(1), matlab.double(2), 
                                 'TileSpacing', 'compact', 
                                 'Padding', 'compact',
                                 nargout=1)
        
        # Create subplots
        axes = []
        for i in range(2):
            ax = engine.nexttile(tcl, nargout=1)
            engine.hold(ax, 'on', nargout=0)
            axes.append(ax)
        
        # Plot trajectories
        color_cycle = cycle(params['colors'])
        legend_handles = []
        
        for traj_idx, (traj, linestyle, legend_label) in enumerate(zip(trajectories, params['linestyles'], params['legend_labels'])):
            p1, p2, q1, q2 = traj.states.get_pq()
            
            # Convert to MATLAB format
            q1 = np.ascontiguousarray(q1)
            p1 = np.ascontiguousarray(p1)
            q2 = np.ascontiguousarray(q2)
            p2 = np.ascontiguousarray(p2)
            
            # Get color and convert to MATLAB format
            if legend_label in ['ref', '$\phi$']:
                matlab_color = [0.0, 0.0, 0.0]  # Black
                alpha = 0.3
            else:
                color = next(color_cycle)
                matlab_color = convert_color_to_matlab(color)
                alpha = 1.0
            
            matlab_linestyle = convert_linestyle_to_matlab(linestyle)
            
            # Plot on both subplots
            handle1 = engine.plot(axes[0], q1, p1, 'LineStyle', matlab_linestyle, 
                                'Color', matlab.double(matlab_color), 'LineWidth', 0.5,
                                'DisplayName', legend_label, nargout=1)
            engine.plot(axes[1], q2, p2, 'LineStyle', matlab_linestyle, 
                       'Color', matlab.double(matlab_color), 'LineWidth', 0.5, nargout=0)
            
            # Store handle for legend
            legend_handles.append(handle1)
        
        # Set axis properties for each subplot
        for i, ax in enumerate(axes):
            if axis_lims is not None:
                engine.xlim(ax, matlab.double(axis_lims[i][0]), nargout=0)
                engine.ylim(ax, matlab.double(axis_lims[i][1]), nargout=0)
            
            engine.xlabel(ax, axis_labels[i][0], 'Interpreter', 'latex', nargout=0)
            engine.ylabel(ax, axis_labels[i][1], 'Interpreter', 'latex', nargout=0)
            engine.title(ax, titles[i], 'Interpreter', 'latex', nargout=0)
            engine.grid(ax, 'on', nargout=0)
        
        # Add legend if requested
        if params['show_legend'] and legend_handles:
            # Store handles individually in MATLAB workspace
            for i, handle in enumerate(legend_handles):
                engine.workspace[f'h_{i+1}'] = handle
            
            # Create legend using eval with individual handles
            handle_list = ', '.join([f'h_{i+1}' for i in range(len(legend_handles))])
            engine.eval(f"leg = legend([{handle_list}], 'Orientation', 'horizontal', 'Interpreter', 'latex');", nargout=0)
            
            # Set layout and font size using eval
            engine.eval("leg.Layout.Tile = 'south';", nargout=0)
            engine.eval("leg.FontSize = 8;", nargout=0)
        
        # Adjust figure size before saving
        adjust_matlab_figure(engine, target_width=5.0)
        
        # Save figure using helper function
        save_matlab_figure(engine, params['save_path'])
    
    def plot_history(self, trajectories: List[Any], **kwargs) -> None:
        """
        Plot time series of NCO trajectory components.
        
        Args:
            trajectories: List of trajectory objects to plot
            **kwargs: Additional plotting parameters including:
                - func_dict: Dictionary of functions to plot (e.g., {'H': lambda u: processor.compute_hamiltonian(u)})
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
                - alpha: Transparency for non-reference trajectories
        """
        # Get common parameters
        params = self._get_plotting_parameters(**kwargs)
        
        # Extract method-specific parameters
        func_dict = kwargs.get('func_dict', {
            'H': lambda u: trajectories[0].states.processor.compute_hamiltonian(u),
            'V': lambda u: trajectories[0].states.processor.compute_potential_energy(u),
            'K': lambda u: trajectories[0].states.processor.compute_kinetic_energy(u)
        })
        xlabel = kwargs.get('xlabel', 't')
        ylabels = kwargs.get('ylabels', list(func_dict.keys()))
        ylims = kwargs.get('ylims', [(None, None)] * len(func_dict))
        log_yscale = kwargs.get('log_yscale', False)
        log_xscale = kwargs.get('log_xscale', False)
        alpha = kwargs.get('alpha', 1.0)
        
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
            if legend_label == 'ref':
                color = 'k'
            else:
                color = next(color_cycle)
            
            for i, (key, func) in enumerate(func_dict.items()):
                if legend_label == 'ref':
                    line1, = axes[i].plot(xdata, func(traj.states.u), linestyle, 
                                       lw=1, alpha=0.2, color=color, label=legend_label)
                else:
                    line1, = axes[i].plot(xdata, func(traj.states.u), linestyle, 
                                       lw=1, alpha=alpha, color=color, label=legend_label)
            lines.append(line1)
        
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
        Plot time series of NCO trajectory components using MATLAB backend.
        
        Args:
            trajectories: List of trajectory objects to plot
            **kwargs: Additional plotting parameters including:
                - func_dict: Dictionary of functions to plot (e.g., {'H': lambda u: processor.compute_hamiltonian(u)})
                - linestyles: List of line styles for each trajectory
                - colors: List of colors for each trajectory
                - legend_labels: Labels for the legend
                - xlabel: Label for x-axis
                - ylabels: Labels for y-axes
                - ylims: List of y-axis limits for each subplot
                - figsize: Figure size tuple
                - log_yscale: Whether to use log scale for y-axis
                - log_xscale: Whether to use log scale for x-axis
                - save_path: Path to save the figure
                - alpha: Transparency for non-reference trajectories
        """
        if not MATLAB_AVAILABLE:
            print("MATLAB not available, falling back to matplotlib")
            return self.plot_history(trajectories, **kwargs)
        
        # Get common parameters
        params = self._get_plotting_parameters(**kwargs)
        
        # Extract method-specific parameters
        func_dict = kwargs.get('func_dict', {
            'H': lambda u: trajectories[0].states.processor.compute_hamiltonian(u),
            'V': lambda u: trajectories[0].states.processor.compute_potential_energy(u),
            'K': lambda u: trajectories[0].states.processor.compute_kinetic_energy(u)
        })
        xlabel = kwargs.get('xlabel', 't')
        ylabels = kwargs.get('ylabels', list(func_dict.keys()))
        ylims = kwargs.get('ylims', [(None, None)] * len(func_dict))
        log_yscale = kwargs.get('log_yscale', False)
        log_xscale = kwargs.get('log_xscale', False)
        alpha = kwargs.get('alpha', 1.0)
        
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
        tcl = engine.tiledlayout(matlab.double(num_subplots), matlab.double(1),
                                 'TileSpacing', 'compact',
                                 'Padding', 'compact',
                                 nargout=1)
        
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
            
            # Ensure xdata is contiguous
            xdata = np.ascontiguousarray(xdata)
            
            # Get color and convert to MATLAB format
            if legend_label == 'ref':
                matlab_color = [0.0, 0.0, 0.0]  # Black
                linewidth = 1.0
                alpha = 0.2
            else:
                color = next(color_cycle)
                matlab_color = convert_color_to_matlab(color)
                linewidth = 1.0
                alpha = alpha
            
            matlab_linestyle = convert_linestyle_to_matlab(linestyle)
            
            for i, (key, func) in enumerate(func_dict.items()):
                # Use the pre-created tile
                ax = axes[i]
                
                # Get y data and ensure it's contiguous
                ydata = func(traj.states.u)
                ydata = np.ascontiguousarray(ydata)
                
                # Plot the line
                handle = engine.plot(ax, xdata, ydata, 'LineStyle', matlab_linestyle, 'LineWidth', linewidth, 
                                   'Color', matlab.double(matlab_color), 'DisplayName', legend_label, nargout=1)
                
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
                        # Ensure at least 2 y-ticks for log scale
                        engine.eval(f"yt = get(ax, 'YTick'); if length(yt) < 2, set(ax, 'YTick', [min(yt), max(yt)]); end", nargout=0)
                    if log_xscale:
                        engine.set(ax, 'XScale', 'log', nargout=0)
                    
                    # Add grid
                    engine.grid(ax, 'on', nargout=0)
        
        # Add title if provided
        if params['title']:
            engine.suptitle(params['title'], 'Interpreter', 'latex', nargout=0)

        # Add legend if requested
        if params['show_legend'] and legend_handles:
            # Store handles individually in MATLAB workspace
            for i, handle in enumerate(legend_handles):
                engine.workspace[f'h_{i+1}'] = handle
            
            # Create legend using eval with individual handles
            handle_list = ', '.join([f'h_{i+1}' for i in range(len(legend_handles))])
            engine.eval(f"leg = legend([{handle_list}], 'Orientation', 'horizontal', 'Interpreter', 'latex');", nargout=0)
            
            # Set layout and font size using eval
            engine.eval("leg.Layout.Tile = 'south';", nargout=0)
            engine.eval("leg.FontSize = 6;", nargout=0)
        
        # Adjust figure size before saving
        adjust_matlab_figure(engine, target_width=6.0)

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
        error_names = kwargs.get('error_names', ["osc1_rel_traj_err", "osc2_rel_traj_err", "rel_traj_err", "abs_H_err"])
        titles = kwargs.get('titles', ['Oscillator 1 traj error', 'Oscillator 2 traj error', 'Traj error', 'H error'])
        ylims = kwargs.get('ylims', [(1e-4, 1e0)] * len(error_names))
        ylabels = kwargs.get('ylabels', ['rel err', 'rel err', 'rel err', 'abs err'])
        log_yscale = kwargs.get('log_yscale', True)
        log_xscale = kwargs.get('log_xscale', False)
        
        # Set defaults for common parameters if not provided
        if params['legend_labels'] is None:
            params['legend_labels'] = [f'Trajectory {i+1}' for i in range(len(trajectories))]
        if params['linestyles'] == self.default_line_styles:
            params['linestyles'] = ['-'] * len(trajectories)
        
        # Handle single reference trajectory
        if not isinstance(reference, list):
            reference = [reference] * len(trajectories)
        
        # Set dynamic figsize based on number of error types
        if params['figsize'] == (8, 6):  # Default from base class
            params['figsize'] = (9, 3)  # 2x2 grid for NCO errors
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=params['figsize'], layout='constrained')
        axes = axes.flatten()
        
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
            ax.set_ylim(ylims[i])
            ax.set_xlabel('$t$')
            ax.set_ylabel(ylabels[i])
            ax.set_title(titles[i])
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
    
    def plot_errors_matlab(self, trajectories: List[Any], reference: Any, **kwargs) -> None:
        """
        Plot error analysis comparing trajectories to reference solution using MATLAB backend.
        
        Args:
            trajectories: List of trajectory objects to compare
            reference: Reference trajectory object
            **kwargs: Additional plotting parameters including:
                - error_names: List of error types to plot
                - linestyles: List of line styles for each trajectory
                - colors: List of colors for each trajectory
                - legend_labels: Labels for the legend
                - titles: List of titles for each subplot
                - ylims: List of y-axis limits for each subplot
                - ylabels: Labels for y-axes
                - figsize: Figure size tuple
                - log_yscale: Whether to use log scale for y-axis
                - log_xscale: Whether to use log scale for x-axis
                - save_path: Path to save the figure
        """
        if not MATLAB_AVAILABLE:
            print("MATLAB not available, falling back to matplotlib")
            return self.plot_errors(trajectories, reference, **kwargs)
        
        # Get common parameters
        params = self._get_plotting_parameters(**kwargs)
        
        # Extract method-specific parameters
        error_names = kwargs.get('error_names', ["osc1_abs_traj_err", "osc2_abs_traj_err", "abs_traj_err", "abs_H_err"])
        titles = kwargs.get('titles', ['Oscillator 1 traj error', 'Oscillator 2 traj error', 'Traj error', '$H$ error'])
        ylims = kwargs.get('ylims', [(1e-4, 1e0)] * len(error_names))
        ylabels = kwargs.get('ylabels', ['abs err', 'abs err', 'abs err', 'abs err'])
        log_yscale = kwargs.get('log_yscale', True)
        log_xscale = kwargs.get('log_xscale', False)
        
        # Set defaults for common parameters if not provided
        if params['legend_labels'] is None:
            params['legend_labels'] = [f'Trajectory {i+1}' for i in range(len(trajectories))]
        if params['linestyles'] == self.default_line_styles:
            params['linestyles'] = ['-'] * len(trajectories)
        
        # Handle single reference trajectory
        if not isinstance(reference, list):
            reference = [reference] * len(trajectories)
        
        # Set dynamic figsize based on number of error types
        if params['figsize'] == (8, 6):  # Default from base class
            params['figsize'] = (9, 3)  # 2x2 grid for NCO errors
        
        # Get MATLAB engine
        engine = get_matlab_engine()
        
        # Setup figure
        setup_matlab_figure(engine, params['figsize'])
        
        # Create tiledlayout (2 rows, 2 columns)
        tcl = engine.tiledlayout(matlab.double(2), matlab.double(2), 
                                 'TileSpacing', 'compact',
                                 'Padding', 'compact',
                                 nargout=1)
        
        # Create all tiles first
        axes = []
        for i in range(len(error_names)):
            ax = engine.nexttile(tcl, nargout=1)
            engine.hold(ax, 'on', nargout=0)
            axes.append(ax)
        
        # Create color cycle
        color_cycle = cycle(params['colors'])

        # Plot errors and collect handles
        legend_handles = []
        num_trajectories = len(trajectories)
        
        for traj_idx, (traj, ref_traj, linestyle, legend_label) in enumerate(zip(trajectories, reference, params['linestyles'], params['legend_labels'])):
            tt, errors = traj.compare(ref_traj)
            
            # Ensure tt is contiguous
            tt = np.ascontiguousarray(tt)
            
            # Get color and convert to MATLAB format
            color = next(color_cycle)
            matlab_color = convert_color_to_matlab(color)
            matlab_linestyle = convert_linestyle_to_matlab(linestyle)
            
            for i, error_name in enumerate(error_names):
                # Use the pre-created tile
                ax = axes[i]
                
                # Get y data and ensure it's contiguous
                ydata = errors[error_name]
                ydata = np.ascontiguousarray(ydata)
                
                # Plot the line
                handle = engine.plot(ax, tt[1:], ydata[1:], 'LineStyle', matlab_linestyle, 'LineWidth', 0.7, 
                                   'Color', matlab.double(matlab_color), 'DisplayName', legend_label, nargout=1)
                
                # Store handles only from first subplot for legend
                if i == 0:
                    legend_handles.append(handle)
                
                # Print error values
                print(f"{legend_label} n=1 {error_name} = {errors[error_name][1]}")
                print(f"{legend_label} n={len(tt)-1} {error_name} = {errors[error_name][-1]}")
                
                # Set axis properties for this subplot (only on first trajectory)
                if traj_idx == 0:
                    if ylims[i] is not None and all(v is not None for v in ylims[i]):
                        engine.ylim(ax, matlab.double(ylims[i]), nargout=0)
                    
                    # Set labels and title
                    engine.xlabel(ax, '$t$', 'Interpreter', 'latex', nargout=0)
                    engine.ylabel(ax, ylabels[i], 'Interpreter', 'latex', nargout=0)
                    engine.title(ax, titles[i], 'Interpreter', 'latex', nargout=0)
                    
                    # Set log scales
                    if log_yscale:
                        engine.set(ax, 'YScale', 'log', nargout=0)
                    if log_xscale:
                        engine.set(ax, 'XScale', 'log', nargout=0)
                    
                    # Add grid
                    engine.grid(ax, 'on', nargout=0)

        # Add legend if requested
        if params['show_legend'] and legend_handles:
            # Store handles individually in MATLAB workspace
            for i, handle in enumerate(legend_handles):
                engine.workspace[f'h_{i+1}'] = handle
            
            # Create legend using eval with individual handles
            handle_list = ', '.join([f'h_{i+1}' for i in range(len(legend_handles))])
            engine.eval(f"leg = legend([{handle_list}], 'Orientation', 'horizontal', 'Interpreter', 'latex');", nargout=0)
            
            # Set layout and font size using eval
            engine.eval("leg.Layout.Tile = 'south';", nargout=0)
            engine.eval("leg.FontSize = 7;", nargout=0)
        
        # Adjust figure size before saving
        adjust_matlab_figure(engine, target_width=5.0)

        # Save figure using helper function
        save_matlab_figure(engine, params['save_path'])
    
    def plot_pq_vs_t(self, trajectories: List[Any], **kwargs) -> None:
        """
        Plot position and momentum variables vs time (or index) for NCO system.
        
        Args:
            trajectories: List of trajectory or dataset objects to plot
            **kwargs: Additional plotting parameters including:
                - linestyles: List of line styles for each trajectory
                - colors: List of colors for each trajectory
                - legend_labels: Labels for the legend
                - xlabel: Label for x-axis (default: 't' for trajectories, 'index' for datasets)
                - ylabels: Labels for y-axes
                - ylims: List of y-axis limits for each subplot
                - alpha: Transparency for non-reference trajectories
                - figsize: Figure size tuple
                - show_legend: Whether to display legend
                - save_path: Path to save the figure
        """
        # Get common parameters
        params = self._get_plotting_parameters(**kwargs)
        
        # Extract method-specific parameters
        ylabels = kwargs.get('ylabels', ['$q_1$', '$q_2$', '$p_1$', '$p_2$'])
        ylims = kwargs.get('ylims', [(None, None)] * 4)
        alpha = kwargs.get('alpha', 1.0)
        
        # Determine x-axis label based on data type
        has_times = hasattr(trajectories[0], 'times')
        default_xlabel = '$t$' if has_times else 'index'
        xlabel = kwargs.get('xlabel', default_xlabel)
        
        # Set defaults for common parameters if not provided
        if params['legend_labels'] is None:
            params['legend_labels'] = [f'Trajectory {i+1}' for i in range(len(trajectories))]
        if params['linestyles'] == self.default_line_styles:
            params['linestyles'] = ['-'] * len(trajectories)
        
        # Handle single linestyle
        if isinstance(params['linestyles'], str):
            params['linestyles'] = [params['linestyles']] * len(trajectories)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=params['figsize'], layout='constrained')
        axes = axes.flatten()
        
        # Plot trajectories
        color_cycle = cycle(params['colors'])
        lines = []
        
        for traj, linestyle, legend_label in zip(trajectories, params['linestyles'], params['legend_labels']):
            p1, p2, q1, q2 = traj.states.get_pq()
            
            # Get x-axis data (time or index)
            if hasattr(traj, 'times'):
                xdata = traj.times
            else:
                xdata = np.arange(len(traj))
            
            if legend_label in ['ref', '$\phi$']:
                line1, = axes[0].plot(xdata, q1, linestyle, lw=1, color='k', alpha=0.3, label=legend_label)
                axes[1].plot(xdata, q2, linestyle, lw=1, color='k', alpha=0.3)
                axes[2].plot(xdata, p1, linestyle, lw=1, color='k', alpha=0.3)
                axes[3].plot(xdata, p2, linestyle, lw=1, color='k', alpha=0.3)
            else:
                color = next(color_cycle)
                line1, = axes[0].plot(xdata, q1, linestyle, lw=1, color=color, alpha=alpha, label=legend_label)
                axes[1].plot(xdata, q2, linestyle, lw=1, color=color, alpha=alpha)
                axes[2].plot(xdata, p1, linestyle, lw=1, color=color, alpha=alpha)
                axes[3].plot(xdata, p2, linestyle, lw=1, color=color, alpha=alpha)
            lines.append(line1)
        
        # Set labels and titles for each subplot
        for i, ax in enumerate(axes):
            ax.set_ylim(ylims[i])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabels[i])
        
        # Add legend
        self._create_legend(fig, lines, params['legend_labels'], params['show_legend'])
        
        # Save figure
        self._save_figure(fig, params['save_path'], params['dpi'])
        
        if not params['save_path']:
            plt.show()
    
    def plot_pq_errors(self, trajectories: List[Any], reference: Any, **kwargs) -> None:
        """
        Plot absolute errors in position and momentum variables.
        
        Args:
            trajectories: List of trajectory objects to compare
            reference: Reference trajectory object
            **kwargs: Additional plotting parameters including:
                - linestyles: List of line styles for each trajectory
                - colors: List of colors for each trajectory
                - legend_labels: Labels for the legend
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
        ylabels = kwargs.get('ylabels', ['$q_1$ error', '$q_2$ error', '$p_1$ error', '$p_2$ error'])
        ylims = kwargs.get('ylims', [(1e-4, 1e0)] * 4)
        log_yscale = kwargs.get('log_yscale', True)
        log_xscale = kwargs.get('log_xscale', False)
        
        # Set defaults for common parameters if not provided
        if params['legend_labels'] is None:
            params['legend_labels'] = [f'Trajectory {i+1}' for i in range(len(trajectories))]
        if params['linestyles'] == self.default_line_styles:
            params['linestyles'] = ['-'] * len(trajectories)
        
        # Handle single reference trajectory
        if not isinstance(reference, list):
            reference = [reference] * len(trajectories)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=params['figsize'], layout='constrained')
        axes = axes.flatten()
        
        # Plot errors
        color_cycle = cycle(params['colors'])
        lines = []
        
        for traj, ref_traj, linestyle, legend_label in zip(trajectories, reference, params['linestyles'], params['legend_labels']):
            # Use intersect method to get aligned trajectories
            matched_traj, matched_ref_traj = traj.intersect(ref_traj)
            if matched_traj is None or matched_ref_traj is None:
                print(f"Warning: Could not align trajectory {legend_label} with reference")
                continue
                
            tt = matched_traj.times
            p1, p2, q1, q2 = matched_traj.states.get_pq()
            p1_ref, p2_ref, q1_ref, q2_ref = matched_ref_traj.states.get_pq()
            
            color = next(color_cycle)
            line1, = axes[0].plot(tt[1:], np.abs(q1-q1_ref)[1:], linestyle, color=color, label=legend_label)
            axes[1].plot(tt[1:], np.abs(q2-q2_ref)[1:], linestyle, color=color)
            axes[2].plot(tt[1:], np.abs(p1-p1_ref)[1:], linestyle, color=color)
            axes[3].plot(tt[1:], np.abs(p2-p2_ref)[1:], linestyle, color=color)
            lines.append(line1)
        
        # Set labels and titles for each subplot
        for i, ax in enumerate(axes):
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
    
    def plot_pq_density(self, trajectories: List[Any], **kwargs) -> None:
        """
        Plot probability density (histograms) of position and momentum variables.
        
        Args:
            trajectories: List of trajectory or dataset objects to plot
            **kwargs: Additional plotting parameters including:
                - colors: List of colors for each trajectory
                - legend_labels: Labels for the legend
                - bins: Number of bins for histograms (default: 30)
                - alpha: Transparency for histograms
                - density: Whether to normalize histogram (default: False)
                - ylabels: Labels for y-axes
                - xlabels: Labels for x-axes
                - figsize: Figure size tuple
                - show_legend: Whether to display legend
                - save_path: Path to save the figure
        """
        # Get common parameters
        params = self._get_plotting_parameters(**kwargs)
        
        # Extract method-specific parameters
        bins = kwargs.get('bins', 30)
        alpha = kwargs.get('alpha', 0.7)
        density = kwargs.get('density', False)
        xlabels = kwargs.get('xlabels', ['$q_1$', '$q_2$', '$p_1$', '$p_2$'])
        ylabels = kwargs.get('ylabels', ['Count', 'Count', 'Count', 'Count'])
        
        if density:
            ylabels = kwargs.get('ylabels', ['Density', 'Density', 'Density', 'Density'])
        
        # Set defaults for common parameters if not provided
        if params['legend_labels'] is None:
            params['legend_labels'] = [f'Data {i+1}' for i in range(len(trajectories))]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=params['figsize'], layout='constrained')
        axes = axes.flatten()
        
        # Plot histograms
        color_cycle = cycle(params['colors'])
        
        for traj, legend_label in zip(trajectories, params['legend_labels']):
            p1, p2, q1, q2 = traj.states.get_pq()
            color = next(color_cycle)
            
            axes[0].hist(q1, bins=bins, alpha=alpha, color=color, label=legend_label, density=density)
            axes[1].hist(q2, bins=bins, alpha=alpha, color=color, label=legend_label, density=density)
            axes[2].hist(p1, bins=bins, alpha=alpha, color=color, label=legend_label, density=density)
            axes[3].hist(p2, bins=bins, alpha=alpha, color=color, label=legend_label, density=density)
        
        # Set labels for each subplot
        for i, ax in enumerate(axes):
            ax.set_xlabel(xlabels[i])
            ax.set_ylabel(ylabels[i])
        
        # Add legend
        if params['show_legend']:
            fig.legend(params['legend_labels'], loc='outside right upper', borderaxespad=0.1)
        
        # Save figure
        self._save_figure(fig, params['save_path'], params['dpi'])
        
        if not params['save_path']:
            plt.show()
    
    def plot_density(self, trajectories: List[Any], func_dict: Dict[str, Any], **kwargs) -> None:
        """
        Plot probability density (histograms) of computed functions.
        
        Args:
            trajectories: List of trajectory or dataset objects to plot
            func_dict: Dictionary mapping function names to functions (e.g., {'H': processor.compute_hamiltonian})
            **kwargs: Additional plotting parameters including:
                - colors: List of colors for each trajectory
                - legend_labels: Labels for the legend
                - bins: Number of bins for histograms (default: 30)
                - alpha: Transparency for histograms
                - density: Whether to normalize histogram (default: False)
                - ylabels: Labels for y-axes (default: 'Count' or 'Density')
                - xlabels: Labels for x-axes (default: function names from func_dict)
                - figsize: Figure size tuple
                - show_legend: Whether to display legend
                - save_path: Path to save the figure
        """
        # Get common parameters
        params = self._get_plotting_parameters(**kwargs)
        
        # Extract method-specific parameters
        bins = kwargs.get('bins', 30)
        alpha = kwargs.get('alpha', 0.7)
        density = kwargs.get('density', False)
        xlabels = kwargs.get('xlabels', list(func_dict.keys()))
        
        # Set y-labels based on density flag
        if density:
            ylabels = kwargs.get('ylabels', ['Density'] * len(func_dict))
        else:
            ylabels = kwargs.get('ylabels', ['Count'] * len(func_dict))
        
        # Set defaults for common parameters if not provided
        if params['legend_labels'] is None:
            params['legend_labels'] = [f'Data {i+1}' for i in range(len(trajectories))]
        
        # Determine grid layout based on number of functions
        num_funcs = len(func_dict)
        if num_funcs <= 2:
            nrows, ncols = 1, num_funcs
        elif num_funcs <= 4:
            nrows, ncols = 2, 2
        elif num_funcs <= 6:
            nrows, ncols = 2, 3
        elif num_funcs <= 9:
            nrows, ncols = 3, 3
        else:
            nrows = (num_funcs + 3) // 4
            ncols = 4
        
        # Adjust figsize if default
        if params['figsize'] == (8, 6):  # Default from base class
            params['figsize'] = (4 * ncols, 3 * nrows)
        
        # Create figure
        fig, axes = plt.subplots(nrows, ncols, figsize=params['figsize'], layout='constrained')
        
        # Flatten axes for easier iteration
        if num_funcs == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else axes
        
        # Plot histograms
        color_cycle = cycle(params['colors'])
        
        for traj, legend_label in zip(trajectories, params['legend_labels']):
            color = next(color_cycle)
            
            for i, (func_name, func) in enumerate(func_dict.items()):
                # Compute function values
                values = func(traj.states.u)
                
                # Plot histogram
                axes[i].hist(values, bins=bins, alpha=alpha, color=color, 
                           label=legend_label, density=density)
        
        # Set labels for each subplot
        for i, (ax, xlabel, ylabel) in enumerate(zip(axes[:num_funcs], xlabels, ylabels)):
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        
        # Hide unused subplots
        for i in range(num_funcs, len(axes)):
            axes[i].set_visible(False)
        
        # Add legend
        if params['show_legend']:
            fig.legend(params['legend_labels'], loc='outside right upper', borderaxespad=0.1)
        
        # Save figure
        self._save_figure(fig, params['save_path'], params['dpi'])
        
        if not params['save_path']:
            plt.show()

