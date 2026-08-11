"""
FPU (Fermi-Pasta-Ulam-Tsingou) chain visualization utilities.

This module provides specialized plotting functions for FPU chain dynamics,
including position and momentum visualization, energy distribution, and
normal mode analysis.
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


class FPUVisualizer(BaseVisualizer):
    """
    Visualization utilities for FPU chain dynamics.
    
    This class provides specialized plotting functions for FPU trajectories,
    including chain visualization, energy analysis, and normal mode analysis.
    """
    
    def __init__(self):
        """Initialize the FPU visualizer."""
        super().__init__("fpu")
        self._setup_fpu_styles()
    
    def _setup_fpu_styles(self):
        """Setup FPU specific plotting styles."""
        self.fpu_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_phase_space(self, trajectories: List[Any], **kwargs) -> None:
        """
        Plot FPU chain trajectories in phase space (q_i, p_i) for all oscillators.
        
        Args:
            trajectories: List of trajectory objects to plot
            **kwargs: Additional plotting parameters including:
                - axis_lims: List of axis limits for each oscillator [(q1_lims, p1_lims), ...]
                - linestyles: List of line styles for each trajectory
                - colors: List of colors for each trajectory
                - legend_labels: Labels for the legend
                - axis_labels: Labels for axes [(q1_label, p1_label), ...]
                - titles: List of titles for each subplot
                - figsize: Figure size tuple
                - show_legend: Whether to display legend
                - save_path: Path to save the figure
        """
        # Get common parameters
        params = self._get_plotting_parameters(**kwargs)
        
        # Extract method-specific parameters
        axis_lims = kwargs.get('axis_lims', None)
        axis_labels = kwargs.get('axis_labels', 
            [('$q_1$', '$p_1$'), ('$q_2$', '$p_2$'), ('$q_3$', '$p_3$'), 
             ('$q_4$', '$p_4$'), ('$q_5$', '$p_5$'), ('$q_6$', '$p_6$')])
        titles = kwargs.get('titles', None)
        
        # Set defaults for common parameters if not provided
        if params['legend_labels'] is None:
            params['legend_labels'] = [f'Trajectory {i+1}' for i in range(len(trajectories))]
        if params['linestyles'] == self.default_line_styles:
            params['linestyles'] = ['-'] * len(trajectories)
        
        # Handle single linestyle
        if isinstance(params['linestyles'], str):
            params['linestyles'] = [params['linestyles']] * len(trajectories)
        
        # Create figure with 6 subplots (1x6 grid for 6 oscillators)
        fig, axes = plt.subplots(1, 6, figsize=params['figsize'], layout='constrained')
        
        # Plot trajectories
        color_cycle = cycle(params['colors'])
        lines = []
        
        for traj, linestyle, legend_label in zip(trajectories, params['linestyles'], params['legend_labels']):
            p, q = traj.states.get_pq()
            
            if legend_label in ['ref', '$\phi$']:
                line1, = axes[0].plot(q[:, 0], p[:, 0], linestyle, lw=1, color='k', alpha=0.3, label=legend_label)
                for i in range(1, 6):
                    axes[i].plot(q[:, i], p[:, i], linestyle, lw=1, color='k', alpha=0.3)
            else:
                color = next(color_cycle)
                line1, = axes[0].plot(q[:, 0], p[:, 0], linestyle, lw=1, color=color, label=legend_label)
                for i in range(1, 6):
                    axes[i].plot(q[:, i], p[:, i], linestyle, lw=1, color=color)
            lines.append(line1)
        
        # Set labels and titles for each subplot
        for i, ax in enumerate(axes):
            if axis_lims is not None:
                ax.set_xlim(axis_lims[i][0])
                ax.set_ylim(axis_lims[i][1])
            ax.set_xlabel(axis_labels[i][0])
            ax.set_ylabel(axis_labels[i][1])
            if titles is not None:
                ax.set_title(titles[i])
        
        # Add legend
        self._create_legend(fig, lines, params['legend_labels'], params['show_legend'])
        
        # Save figure
        self._save_figure(fig, params['save_path'], params['dpi'])
        
        if not params['save_path']:
            plt.show()
    
    def plot_history(self, trajectories: List[Any], **kwargs) -> None:
        """
        Plot time series of FPU trajectory components.
        
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
            'U': lambda u: trajectories[0].states.processor.compute_potential_energy(u),
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
            if legend_label in ['ref', '$\phi$']:
                if len(func_dict) > 1:
                    for i, (key, func) in enumerate(func_dict.items()):
                        line1, = axes[i].plot(traj.times, func(traj.states.u), linestyle, alpha=0.5, color='k', label=legend_label)
                else:
                    func = list(func_dict.values())[0]
                    line1, = axes.plot(traj.times, func(traj.states.u), linestyle, alpha=0.5, color='k', label=legend_label)
            else:
                color = next(color_cycle)
                if len(func_dict) > 1:
                    for i, (key, func) in enumerate(func_dict.items()):
                        line1, = axes[i].plot(traj.times, func(traj.states.u), linestyle, alpha=alpha, color=color, label=legend_label)
                else:
                    func = list(func_dict.values())[0]
                    line1, = axes.plot(traj.times, func(traj.states.u), linestyle, alpha=alpha, color=color, label=legend_label)
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
        error_names = kwargs.get('error_names', ["abs_traj_err", "abs_H_err"])
        ylims = kwargs.get('ylims', [(1e-4, 1e0)] * len(error_names))
        ylabels = kwargs.get('ylabels', error_names)
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
            params['figsize'] = (8, 1.5 * len(error_names))
        
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
                line1, = axes[i].plot(tt[1:], errors[error_name][1:], linestyle, color=color, label=legend_label)
                print(f"{legend_label} n=1 {error_name} = {errors[error_name][1]}")
                print(f"{legend_label} n={len(tt)-1} {error_name} = {errors[error_name][-1]}")
            lines.append(line1)
        
        # Set axis properties
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
    
    def plot_stiff_spring_energies(self, trajectory: Any, **kwargs) -> None:
        """
        Plot energy of stiff springs vs time for FPU chain.
        
        Args:
            trajectory: Single trajectory object to plot
            **kwargs: Additional plotting parameters including:
                - xlim: x-axis limits
                - ylim: y-axis limits
                - figsize: Figure size tuple
                - save_path: Path to save the figure
        """
        # Extract method-specific parameters
        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)
        figsize = kwargs.get('figsize', (4, 3))
        save_path = kwargs.get('save_path', None)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')
        
        I = trajectory.states.processor.compute_stiff_spring_energies(trajectory.states.u)
        H = trajectory.states.processor.compute_hamiltonian(trajectory.states.u)
        
        ax.plot(trajectory.times, I[:, 0], label="$I_1$")
        ax.plot(trajectory.times, I[:, 1], label="$I_2$")
        ax.plot(trajectory.times, I[:, 2], label="$I_3$")
        ax.plot(trajectory.times, I[:, 3], label="$I$")
        ax.plot(trajectory.times, H, c="k", label="$H$")
        
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel("$t$")
        
        # Add legend
        lgd = fig.legend(bbox_to_anchor=(0.55, -0.1), loc='outside lower center', 
                        borderaxespad=0., ncols=5, fontsize="small")
        
        # Save figure
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
            print(f"Figure saved as '{save_path}'")
        else:
            plt.show()
    
    def plot_kinetic_energies(self, trajectory: Any, **kwargs) -> None:
        """
        Plot kinetic energy components vs time for FPU chain.
        
        Args:
            trajectory: Single trajectory object to plot
            **kwargs: Additional plotting parameters including:
                - xlim: x-axis limits
                - ylim: y-axis limits
                - figsize: Figure size tuple
                - save_path: Path to save the figure
        """
        # Extract method-specific parameters
        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)
        figsize = kwargs.get('figsize', (4, 3))
        save_path = kwargs.get('save_path', None)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')
        
        I = trajectory.states.processor.compute_stiff_spring_energies(trajectory.states.u)
        T0 = trajectory.states.processor.compute_T0(trajectory.states.u)
        T1 = trajectory.states.processor.compute_T1(trajectory.states.u)
        H = trajectory.states.processor.compute_hamiltonian(trajectory.states.u)
        
        ax.plot(trajectory.times, T0, c="C4", label="$T_0$")
        ax.plot(trajectory.times, T1, c="C5", label="$T_1$")
        ax.plot(trajectory.times, I[:, 3], c="C3", label="$I$")
        ax.plot(trajectory.times, H, c="k", label="$H$")
        
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel("$t$")
        
        # Add legend
        lgd = fig.legend(bbox_to_anchor=(0.55, -0.1), loc='outside lower center', 
                        borderaxespad=0., ncols=5, fontsize="small")
        
        # Save figure
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
            print(f"Figure saved as '{save_path}'")
        else:
            plt.show()
    
    def plot_phase_space_matlab(self, trajectories: List[Any], **kwargs) -> None:
        """
        Plot FPU chain trajectories in phase space using MATLAB backend for high-quality output.
        
        Args:
            trajectories: List of trajectory objects to plot
            **kwargs: Additional plotting parameters including:
                - axis_lims: List of axis limits for each oscillator [(q1_lims, p1_lims), ...]
                - linestyles: List of line styles for each trajectory
                - colors: List of colors for each trajectory
                - legend_labels: Labels for the legend
                - axis_labels: Labels for axes [(q1_label, p1_label), ...]
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
        axis_labels = kwargs.get('axis_labels', 
            [('$q_1$', '$p_1$'), ('$q_2$', '$p_2$'), ('$q_3$', '$p_3$'), 
             ('$q_4$', '$p_4$'), ('$q_5$', '$p_5$'), ('$q_6$', '$p_6$')])
        titles = kwargs.get('titles', None)
        
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
        
        # Setup figure with 6 subplots (1x6 grid)
        setup_matlab_figure(engine, params['figsize'])
        
        # Create tiledlayout (1 row, 6 columns)
        tcl = engine.tiledlayout(matlab.double(1), matlab.double(6), 
                                 'TileSpacing', 'compact', 
                                 'Padding', 'compact',
                                 nargout=1)
        
        # Create subplots
        axes = []
        for i in range(6):
            ax = engine.nexttile(tcl, nargout=1)
            engine.hold(ax, 'on', nargout=0)
            axes.append(ax)
        
        # Plot trajectories
        color_cycle = cycle(params['colors'])
        legend_handles = []
        
        for traj_idx, (traj, linestyle, legend_label) in enumerate(zip(trajectories, params['linestyles'], params['legend_labels'])):
            p, q = traj.states.get_pq()
            
            # Convert to MATLAB format
            p = np.ascontiguousarray(p)
            q = np.ascontiguousarray(q)
            
            # Get color and convert to MATLAB format
            if legend_label in ['ref', '$\phi$']:
                matlab_color = [0.0, 0.0, 0.0]  # Black
                alpha = 0.3
            else:
                color = next(color_cycle)
                matlab_color = convert_color_to_matlab(color)
                alpha = 1.0
            
            matlab_linestyle = convert_linestyle_to_matlab(linestyle)
            
            # Plot on all subplots
            handle1 = engine.plot(axes[0], q[:, 0], p[:, 0], 'LineStyle', matlab_linestyle, 
                                'Color', matlab.double(matlab_color), 'LineWidth', 0.5,
                                'DisplayName', legend_label, nargout=1)
            for i in range(1, 6):
                engine.plot(axes[i], q[:, i], p[:, i], 'LineStyle', matlab_linestyle, 
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
            if titles is not None:
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
        adjust_matlab_figure(engine, target_width=8.0)
        
        # Save figure using helper function
        save_matlab_figure(engine, params['save_path'])
    
    def plot_history_matlab(self, trajectories: List[Any], **kwargs) -> None:
        """
        Plot time series of FPU trajectory components using MATLAB backend.
        
        Args:
            trajectories: List of trajectory objects to plot
            **kwargs: Additional plotting parameters including:
                - func_dict: Dictionary of functions to plot
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
            'U': lambda u: trajectories[0].states.processor.compute_potential_energy(u),
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
        
        for traj_idx, (traj, linestyle, legend_label) in enumerate(zip(trajectories, params['linestyles'], params['legend_labels'])):
            xdata = traj.times if hasattr(traj, 'times') else np.arange(len(traj))
            
            # Ensure xdata is contiguous
            xdata = np.ascontiguousarray(xdata)
            
            # Get color and convert to MATLAB format
            if legend_label in ['ref', '$\phi$']:
                matlab_color = [0.0, 0.0, 0.0]  # Black
                linewidth = 1.0
                alpha = 0.3
            else:
                color = next(color_cycle)
                matlab_color = convert_color_to_matlab(color)
                linewidth = 1.0
                alpha = alpha
            
            # Add alpha channel to color for transparency
            matlab_color_with_alpha = matlab_color + [alpha]
            matlab_linestyle = convert_linestyle_to_matlab(linestyle)
            
            for i, (key, func) in enumerate(func_dict.items()):
                # Use the pre-created tile
                ax = axes[i]
                
                # Get y data and ensure it's contiguous
                ydata = func(traj.states.u)
                ydata = np.ascontiguousarray(ydata)
                
                # Plot the line with transparency
                handle = engine.plot(ax, xdata, ydata, 'LineStyle', matlab_linestyle, 'LineWidth', linewidth, 
                                   'Color', matlab.double(matlab_color_with_alpha), 'DisplayName', legend_label, nargout=1)
                
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
        
        # Add legend if requested
        if params['show_legend'] and legend_handles:
            # Store handles individually in MATLAB workspace
            for i, handle in enumerate(legend_handles):
                engine.workspace[f'h_{i+1}'] = handle
            
            # Create legend using eval with individual handles
            handle_list = ', '.join([f'h_{i+1}' for i in range(len(legend_handles))])
            engine.eval(f"leg = legend([{handle_list}], 'Orientation', 'horizontal', 'Interpreter', 'latex', 'NumColumns', 2);", nargout=0)
            
            # Set layout and font size using eval
            engine.eval("leg.Layout.Tile = 'south';", nargout=0)
            engine.eval("leg.FontSize = 7;", nargout=0)
        
        # Adjust figure size before saving
        adjust_matlab_figure(engine, target_width=3.0)

        # Save figure using helper function
        save_matlab_figure(engine, params['save_path'])
    
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
        error_names = kwargs.get('error_names', ["abs_traj_err", "abs_H_err"])
        ylims = kwargs.get('ylims', [(1e-4, 1e0)] * len(error_names))
        ylabels = kwargs.get('ylabels', error_names)
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
            params['figsize'] = (8, 1.5 * len(error_names))
        
        # Get MATLAB engine
        engine = get_matlab_engine()
        
        # Setup figure
        setup_matlab_figure(engine, params['figsize'])
        
        # Create tiledlayout (num_subplots rows, 1 column)
        num_subplots = len(error_names)
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

        # Collect all time data to determine consistent xlim
        all_times = []
        for traj, ref_traj in zip(trajectories, reference):
            tt, _ = traj.compare(ref_traj)
            all_times.extend(tt[1:])  # Skip first point as in plotting
        
        # Determine consistent xlim from all data
        xlim_min = min(all_times)
        xlim_max = max(all_times)
        
        # Plot errors and collect handles
        legend_handles = []
        
        for traj_idx, (traj, ref_traj, linestyle, legend_label) in enumerate(zip(trajectories, reference, params['linestyles'], params['legend_labels'])):
            tt, errors = traj.compare(ref_traj)
            
            # Ensure time data is contiguous
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
                handle = engine.plot(ax, tt[1:], ydata[1:], 'LineStyle', matlab_linestyle, 'LineWidth', 1, 
                                   'Color', matlab.double(matlab_color), 'DisplayName', legend_label, nargout=1)
                
                # Store handles only from first subplot for legend
                if i == 0:
                    legend_handles.append(handle)
                
                # Print error information
                print(f"{legend_label} n=1 {error_name} = {errors[error_name][1]}")
                print(f"{legend_label} n={len(tt)-1} {error_name} = {errors[error_name][-1]}")
                
                # Set axis properties for this subplot (only on first trajectory)
                if traj_idx == 0:
                    # Set consistent xlim for all subplots
                    engine.xlim(ax, matlab.double([xlim_min, xlim_max]), nargout=0)
                    
                    if ylims[i] is not None and all(v is not None for v in ylims[i]):
                        engine.ylim(ax, matlab.double(ylims[i]), nargout=0)
                    
                    # Set labels
                    engine.xlabel(ax, '$t$', 'Interpreter', 'latex', nargout=0)
                    engine.ylabel(ax, ylabels[i], 'Interpreter', 'latex', nargout=0)
                    
                    # Set log scales
                    if log_yscale:
                        engine.set(ax, 'YScale', 'log', nargout=0)
                        # Ensure tick labels are visible for log scale
                        if ylims[i] is not None and all(v is not None for v in ylims[i]):
                            y_min, y_max = ylims[i]
                            if y_min > 0 and y_max > 0:  # Ensure positive values for log scale
                                tick_values = [y_min, y_max]
                                engine.set(ax, 'YTick', matlab.double(tick_values), nargout=0)
                        else:
                            # Auto-generate ticks for log scale
                            engine.eval("ax.YTickMode = 'auto';", nargout=0)
                            engine.eval("ax.YTickLabelMode = 'auto';", nargout=0)
                    if log_xscale:
                        engine.set(ax, 'XScale', 'log', nargout=0)
                        # Ensure tick labels are visible for log scale
                        engine.eval("ax.XTickMode = 'auto';", nargout=0)
                        engine.eval("ax.XTickLabelMode = 'auto';", nargout=0)
                    
                    # Add grid
                    engine.grid(ax, 'on', nargout=0)
        
        # Add legend if requested
        if params['show_legend'] and legend_handles:
            # Store handles individually in MATLAB workspace
            for i, handle in enumerate(legend_handles):
                engine.workspace[f'h_{i+1}'] = handle
            
            # Create legend using eval with individual handles
            handle_list = ', '.join([f'h_{i+1}' for i in range(len(legend_handles))])
            engine.eval(f"leg = legend([{handle_list}], 'Orientation', 'horizontal', 'Interpreter', 'latex', 'NumColumns', 2);", nargout=0)
            
            # Set layout and font size using eval
            engine.eval("leg.Layout.Tile = 'south';", nargout=0)
            engine.eval("leg.FontSize = 7;", nargout=0)
        
        # Adjust figure size before saving
        adjust_matlab_figure(engine, target_width=3.0)

        # Save figure using helper function
        save_matlab_figure(engine, params['save_path'])
    
    def plot_stiff_spring_energies_matlab(self, trajectory: Any, **kwargs) -> None:
        """
        Plot energy of stiff springs vs time for FPU chain using MATLAB backend.
        
        Args:
            trajectory: Single trajectory object to plot
            **kwargs: Additional plotting parameters including:
                - xlim: x-axis limits
                - ylim: y-axis limits
                - figsize: Figure size tuple
                - save_path: Path to save the figure
        """
        if not MATLAB_AVAILABLE:
            print("MATLAB not available, falling back to matplotlib")
            return self.plot_stiff_spring_energies(trajectory, **kwargs)
        
        # Extract method-specific parameters
        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)
        figsize = kwargs.get('figsize', (4, 3))
        save_path = kwargs.get('save_path', None)
        
        # Get MATLAB engine
        engine = get_matlab_engine()
        
        # Setup figure
        setup_matlab_figure(engine, figsize)
        
        # Compute energy components
        I = trajectory.states.processor.compute_stiff_spring_energies(trajectory.states.u)
        H = trajectory.states.processor.compute_hamiltonian(trajectory.states.u)
        
        # Convert to MATLAB format
        times = np.ascontiguousarray(trajectory.times)
        I1 = np.ascontiguousarray(I[:, 0])
        I2 = np.ascontiguousarray(I[:, 1])
        I3 = np.ascontiguousarray(I[:, 2])
        I_total = np.ascontiguousarray(I[:, 3])
        H_data = np.ascontiguousarray(H)
        
        # Create single subplot
        ax = engine.gca(nargout=1)
        engine.hold(ax, 'on', nargout=0)
        
        # Plot energy components using default matplotlib color sequence
        engine.plot(ax, times, I1, 'Color', matlab.double(convert_color_to_matlab('C0')), 'DisplayName', '$I_1$', 'LineWidth', 1, nargout=1)
        engine.plot(ax, times, I2, 'Color', matlab.double(convert_color_to_matlab('C1')), 'DisplayName', '$I_2$', 'LineWidth', 1, nargout=1)
        engine.plot(ax, times, I3, 'Color', matlab.double(convert_color_to_matlab('C2')), 'DisplayName', '$I_3$', 'LineWidth', 1, nargout=1)
        engine.plot(ax, times, I_total, 'Color', matlab.double(convert_color_to_matlab('C3')), 'DisplayName', '$I$', 'LineWidth', 1, nargout=1)
        engine.plot(ax, times, H_data, 'Color', matlab.double(convert_color_to_matlab('black')), 'DisplayName', '$H$', 'LineWidth', 1, nargout=1)
        
        # Set axis properties
        engine.xlabel(ax, '$t$', 'Interpreter', 'latex', nargout=0)
        if xlim is not None:
            engine.xlim(ax, matlab.double(xlim), nargout=0)
        else:
            # Set xlim to match the data range
            engine.xlim(ax, matlab.double([times[0], times[-1]]), nargout=0)
        if ylim is not None:
            engine.ylim(ax, matlab.double(ylim), nargout=0)
        
        # Add grid
        engine.grid(ax, 'on', nargout=0)
        
        # Add legend positioned to the south
        try:
            leg = engine.legend(ax, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Location', 'southoutside', nargout=1)
            if leg is not None:
                engine.set(leg, 'FontSize', 7, nargout=0)
        except:
            # Fallback: create legend without advanced positioning
            engine.legend(ax, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Location', 'southoutside', nargout=0)
        
        # Adjust figure size before saving
        adjust_matlab_figure(engine, target_width=4.0)
        
        # Save figure using helper function
        save_matlab_figure(engine, save_path)
    
    def plot_kinetic_energies_matlab(self, trajectory: Any, **kwargs) -> None:
        """
        Plot kinetic energy components vs time for FPU chain using MATLAB backend.
        
        Args:
            trajectory: Single trajectory object to plot
            **kwargs: Additional plotting parameters including:
                - xlim: x-axis limits
                - ylim: y-axis limits
                - figsize: Figure size tuple
                - save_path: Path to save the figure
        """
        if not MATLAB_AVAILABLE:
            print("MATLAB not available, falling back to matplotlib")
            return self.plot_kinetic_energies(trajectory, **kwargs)
        
        # Extract method-specific parameters
        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)
        figsize = kwargs.get('figsize', (4, 3))
        save_path = kwargs.get('save_path', None)
        
        # Get MATLAB engine
        engine = get_matlab_engine()
        
        # Setup figure
        setup_matlab_figure(engine, figsize)
        
        # Compute energy components
        I = trajectory.states.processor.compute_stiff_spring_energies(trajectory.states.u)
        T0 = trajectory.states.processor.compute_T0(trajectory.states.u)
        T1 = trajectory.states.processor.compute_T1(trajectory.states.u)
        H = trajectory.states.processor.compute_hamiltonian(trajectory.states.u)
        
        # Convert to MATLAB format
        times = np.ascontiguousarray(trajectory.times)
        T0_data = np.ascontiguousarray(T0)
        T1_data = np.ascontiguousarray(T1)
        I_total = np.ascontiguousarray(I[:, 3])
        H_data = np.ascontiguousarray(H)
        
        # Create single subplot
        ax = engine.gca(nargout=1)
        engine.hold(ax, 'on', nargout=0)
        
        # Plot energy components using default matplotlib color sequence
        engine.plot(ax, times, T0_data, 'Color', matlab.double(convert_color_to_matlab('C4')), 'DisplayName', '$T_0$', 'LineWidth', 1, nargout=1)
        engine.plot(ax, times, T1_data, 'Color', matlab.double(convert_color_to_matlab('C5')), 'DisplayName', '$T_1$', 'LineWidth', 1, nargout=1)
        engine.plot(ax, times, I_total, 'Color', matlab.double(convert_color_to_matlab('C3')), 'DisplayName', '$I$', 'LineWidth', 1, nargout=1)
        engine.plot(ax, times, H_data, 'Color', matlab.double(convert_color_to_matlab('black')), 'DisplayName', '$H$', 'LineWidth', 1, nargout=1)
        
        # Set axis properties
        engine.xlabel(ax, '$t$', 'Interpreter', 'latex', nargout=0)
        if xlim is not None:
            engine.xlim(ax, matlab.double(xlim), nargout=0)
        else:
            # Set xlim to match the data range
            engine.xlim(ax, matlab.double([times[0], times[-1]]), nargout=0)
        if ylim is not None:
            engine.ylim(ax, matlab.double(ylim), nargout=0)
        
        # Add grid
        engine.grid(ax, 'on', nargout=0)
        
        # Add legend positioned to the south
        try:
            leg = engine.legend(ax, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Location', 'southoutside', nargout=1)
            if leg is not None:
                engine.set(leg, 'FontSize', 7, nargout=0)
        except:
            # Fallback: create legend without advanced positioning
            engine.legend(ax, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Location', 'southoutside', nargout=0)
        
        # Adjust figure size before saving
        adjust_matlab_figure(engine, target_width=4.0)
        
        # Save figure using helper function
        save_matlab_figure(engine, save_path)
