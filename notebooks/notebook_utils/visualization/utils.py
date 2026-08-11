"""
Shared visualization utilities for common plotting tasks.

This module provides generic plotting functions that can be used across
different physical systems, along with common data analysis utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Callable, Any

from .config import get_color_palette, get_line_styles


def plot_trajectory_history(
    trajectories: List[Any], 
    quantity_functions: Dict[str, Callable], 
    line_styles: Optional[List[str]] = None, 
    alpha: float = 1.0, 
    colors: Optional[List[str]] = None, 
    legend_labels: Optional[List[str]] = None, 
    x_label: str = "t", 
    y_labels: Optional[List[str]] = None, 
    y_limits: Optional[List[Tuple[Optional[float], Optional[float]]]] = None, 
    figure_size: Optional[Tuple[float, float]] = None, 
    log_y_scale: bool = False, 
    log_x_scale: bool = False, 
    show_legend: bool = True, 
    save_path: Optional[str] = None
) -> None:
    """
    Plot time evolution of various quantities for multiple trajectories.
    
    Args:
        trajectories: List of trajectory objects to plot
        quantity_functions: Dictionary mapping quantity names to functions that extract values
        line_styles: List of line styles for each trajectory
        alpha: Transparency level for non-reference trajectories
        colors: List of colors for each trajectory
        legend_labels: Labels for the legend
        x_label: Label for x-axis
        y_labels: Labels for y-axes
        y_limits: List of (min, max) tuples for y-axes
        figure_size: Figure dimensions
        log_y_scale: Whether to use logarithmic y-scale
        log_x_scale: Whether to use logarithmic x-scale
        show_legend: Whether to display the legend
        save_path: Path to save the figure
    """
    # Set default values
    if line_styles is None:
        line_styles = get_line_styles("default")[:len(trajectories)]
    elif isinstance(line_styles, str):
        line_styles = [line_styles] * len(trajectories)

    if legend_labels is None:
        legend_labels = [f'Trajectory {i+1}' for i in range(len(trajectories))]
    
    if colors is None:
        colors = get_color_palette("default")
    color_cycle = cycle(colors)
    
    if y_labels is None:
        y_labels = list(quantity_functions.keys())

    if y_limits is None:
        y_limits = [(None, None)] * len(quantity_functions)
    
    if figure_size is None:
        figure_size = (8, 1.5 * len(quantity_functions))

    # Create subplots
    fig, axes = plt.subplots(len(quantity_functions), 1, figsize=figure_size, layout='constrained')
    if len(quantity_functions) == 1:
        axes = [axes]

    # Plot each trajectory
    plotted_lines = []
    for trajectory, line_style, legend_label in zip(trajectories, line_styles, legend_labels):
        time_data = trajectory.times if hasattr(trajectory, 'times') else np.arange(len(trajectory))
        color = next(color_cycle)
        
        for i, (quantity_name, extract_function) in enumerate(quantity_functions.items()):
            quantity_values = extract_function(trajectory.states.u)
            
            if legend_label == 'ref':
                line, = axes[i].plot(time_data, quantity_values, line_style, lw=1, 
                                    alpha=0.2, color='k', label=legend_label)
            else:
                line, = axes[i].plot(time_data, quantity_values, line_style, lw=1, 
                                    alpha=alpha, color=color, label=legend_label)
        plotted_lines.append(line)

    # Configure subplot appearance
    for i, ax in enumerate(axes):
        ax.set_ylim(y_limits[i])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_labels[i])
        if log_y_scale:
            ax.set_yscale('log')
        if log_x_scale:
            ax.set_xscale('log')

    if show_legend:
        fig.legend(plotted_lines, legend_labels, loc='outside center right', borderaxespad=0.1)

    # Save or display
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Figure saved as '{save_path}'")
    else:
        plt.show()

    return 


def plot_error_evolution(
    trajectories: List[Any], 
    reference_trajectories: Union[Any, List[Any]], 
    line_styles: Optional[List[str]] = None, 
    colors: Optional[List[str]] = None, 
    legend_labels: Optional[List[str]] = None, 
    titles: Optional[List[str]] = None, 
    error_types: Optional[List[str]] = None, 
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[List[Tuple[float, float]]] = None, 
    y_labels: Optional[List[str]] = None, 
    figure_size: Optional[Tuple[float, float]] = None, 
    log_y_scale: bool = True, 
    log_x_scale: bool = False, 
    show_legend: bool = False, 
    save_path: Optional[str] = None
) -> None:
    """
    Plot error evolution over time for multiple trajectories compared to reference.
    
    Args:
        trajectories: List of trajectories to analyze
        reference_trajectories: Reference trajectory or list of reference trajectories
        line_styles: List of line styles for each trajectory
        colors: List of colors for each trajectory
        legend_labels: Labels for the legend
        titles: Titles for subplots
        error_types: Types of errors to plot
        x_limits: Limits for x-axis
        y_limits: List of (min, max) tuples for y-axes
        y_labels: Labels for y-axes
        figure_size: Figure dimensions
        log_y_scale: Whether to use logarithmic y-scale
        log_x_scale: Whether to use logarithmic x-scale
        show_legend: Whether to display the legend
        save_path: Path to save the figure
    """
    # Handle reference trajectory input
    if not isinstance(reference_trajectories, list):
        reference_trajectories = [reference_trajectories] * len(trajectories)
    elif len(trajectories) != len(reference_trajectories):
        raise ValueError("Number of trajectories and reference trajectories must match")

    # Set default values
    if line_styles is None:
        line_styles = get_line_styles("default")[:len(trajectories)]
    
    if legend_labels is None:
        legend_labels = [f'Trajectory {i+1}' for i in range(len(trajectories))]
    
    if colors is None:
        colors = get_color_palette("default")
    color_cycle = cycle(colors)
    
    if error_types is None:
        error_types = ["abs_traj_err", "abs_H_err"]
    
    if y_limits is None:
        y_limits = [(1e-5, 1e1)] * len(error_types)
    
    if y_labels is None:
        y_labels = error_types
    
    if figure_size is None:
        figure_size = (8, 1.5 * len(error_types))

    # Create subplots
    fig, axes = plt.subplots(len(error_types), 1, figsize=figure_size, layout='constrained')
    if len(error_types) == 1:
        axes = [axes]

    # Plot error evolution for each trajectory
    plotted_lines = []
    for trajectory, ref_trajectory, line_style, legend_label in zip(trajectories, reference_trajectories, line_styles, legend_labels):
        time_points, error_data = trajectory.compare(ref_trajectory)
        color = next(color_cycle)
        
        for i, error_type in enumerate(error_types):
            line, = axes[i].plot(time_points[1:], error_data[error_type][1:], line_style, 
                                lw=1, color=color, label=legend_label)
            print(f"{legend_label} n=1 {error_type} = {error_data[error_type][1]}")
            print(f"{legend_label} n={len(time_points)-1} {error_type} = {error_data[error_type][-1]}")
        plotted_lines.append(line)

    # Configure subplot appearance
    for i, ax in enumerate(axes):
        if x_limits is not None:
            ax.set_xlim(x_limits)
        ax.set_ylim(y_limits[i])
        ax.set_xlabel('$t$')
        ax.set_ylabel(y_labels[i])
        if log_y_scale:
            ax.set_yscale('log')
        if log_x_scale:
            ax.set_xscale('log')

    if show_legend:
        fig.legend(plotted_lines, legend_labels, loc='outside center right', borderaxespad=0.1)

    # Save or display
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Figure saved as '{save_path}'")
    else:
        plt.show()

    return


def plot_convergence_vs_iterations(
    algorithm_groups: Dict[str, List[Any]], 
    reference_trajectory: Any, 
    final_time: float, 
    error_types: Optional[List[str]] = None, 
    figure_size: Optional[Tuple[float, float]] = None, 
    save_path: Optional[str] = None, 
    log_y_scale: bool = True, 
    colors: Optional[List[str]] = None, 
    line_styles: Optional[List[str]] = None, 
    y_limits: Optional[List[Tuple[float, float]]] = None
) -> None:
    """
    Plot error convergence against Parareal iterations for different algorithms.
    
    Args:
        algorithm_groups: Dictionary mapping algorithm names to trajectory lists
        reference_trajectory: Reference trajectory for error computation
        final_time: Time point at which to evaluate errors
        error_types: Types of errors to plot
        figure_size: Figure dimensions
        save_path: Path to save the figure
        log_y_scale: Whether to use logarithmic y-scale
        colors: List of colors for each algorithm group
        line_styles: List of line styles for each algorithm group
        y_limits: List of (min, max) tuples for y-axes
    """
    if error_types is None:
        error_types = ["abs_traj_err", "abs_H_err"]
    
    if figure_size is None:
        figure_size = (5, 1.5 * len(error_types))

    if colors is None:
        colors = get_color_palette("parareal")
    color_cycle = cycle(colors)
    
    if line_styles is None:
        line_styles = get_line_styles("parareal")[:len(algorithm_groups)]
    
    if y_limits is None:
        y_limits = [(1e-14, 1e1)] * len(error_types)

    # Create subplots
    fig, axes = plt.subplots(len(error_types), 1, figsize=figure_size, layout='constrained')
    if len(error_types) == 1:
        axes = [axes]
    
    # Plot convergence for each algorithm group
    for group_idx, (algorithm_name, trajectory_list) in enumerate(algorithm_groups.items()):
        color = next(color_cycle)
        errors_at_final_time = defaultdict(list)
        
        # Compute errors at final time for each iteration
        for trajectory in trajectory_list:
            time_points, error_data = trajectory.compare(reference_trajectory.select_between(final_time, final_time))
            for error_type in error_types:
                errors_at_final_time[error_type].append(error_data[error_type][0])

        # Plot convergence curves
        for error_idx, (error_type, error_values) in enumerate(errors_at_final_time.items()):
            axes[error_idx].plot(error_values, line_styles[group_idx], color=color, label=algorithm_name)
                  
    # Configure subplot appearance
    for i, ax in enumerate(axes):
        ax.set_xlabel("Iteration")
        ax.set_ylabel(error_types[i])
        if log_y_scale:
            ax.set_yscale('log')
        ax.set_ylim(y_limits[i])
        
    # Add legend
    fig.legend(algorithm_groups.keys(), loc='outside center right', borderaxespad=0.1)

    # Save or display
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"Figure saved as '{save_path}'")
    else:
        plt.show()

    return


def plot_error_heatmaps(
    trajectories: List[Any], 
    reference_trajectory: Any, 
    error_types: Optional[List[str]] = None, 
    color_limits: Optional[List[Tuple[float, float]]] = None, 
    figure_size: Optional[Tuple[float, float]] = None, 
    save_path: Optional[str] = None
) -> None:
    """
    Create heatmaps showing error evolution over time and iterations.
    
    Args:
        trajectories: List of trajectories for different iterations
        reference_trajectory: Reference trajectory for error computation
        error_types: Types of errors to visualize
        color_limits: List of (min, max) tuples for color scales
        figure_size: Figure dimensions
        save_path: Path to save the figure
    """
    if error_types is None:
        error_types = ["abs_traj_err", "abs_H_err"]
    
    if color_limits is None:
        color_limits = [(-8, 0)] * len(error_types)

    if figure_size is None:
        figure_size = (5, 4 * len(error_types))

    # Create subplots
    fig, axes = plt.subplots(len(error_types), 1, figsize=figure_size, layout='constrained')
    if len(error_types) == 1:
        axes = [axes]
    
    # Compute error matrices
    error_matrices = defaultdict(list)
    for trajectory in trajectories:
        time_points, error_data = trajectory.compare(reference_trajectory)
        for error_type in error_types:
            error_matrices[error_type].append(error_data[error_type][1:])

    # Create heatmaps
    for i, (error_type, error_matrix_list) in enumerate(error_matrices.items()):
        error_matrix = np.array(error_matrix_list)
        im = axes[i].imshow(
            np.log10(error_matrix), 
            cmap='viridis', 
            interpolation='none', 
            aspect='auto', 
            clim=color_limits[i], 
            extent=[time_points[1], time_points[-1], len(trajectories), 0]
        )
        axes[i].set_title(f"{error_type}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Iteration")
        fig.colorbar(im, ax=axes[i])
    
    # Save or display
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Figure saved as '{save_path}'")
    else:
        plt.show()

    return


def plot_error_difference_heatmaps(
    trajectories_group1: List[Any], 
    trajectories_group2: List[Any], 
    reference_trajectory: Any, 
    error_types: Optional[List[str]] = None, 
    color_limits: Optional[List[Tuple[float, float]]] = None, 
    figure_size: Optional[Tuple[float, float]] = None, 
    save_path: Optional[str] = None, 
    min_log_error: float = -8, 
    max_log_error: float = 2
) -> None:
    """
    Create heatmaps showing differences in error between two algorithm groups.
    
    Args:
        trajectories_group1: First group of trajectories
        trajectories_group2: Second group of trajectories
        reference_trajectory: Reference trajectory for error computation
        error_types: Types of errors to compare
        color_limits: List of (min, max) tuples for color scales
        figure_size: Figure dimensions
        save_path: Path to save the figure
        min_log_error: Minimum log10 error value for clipping
        max_log_error: Maximum log10 error value for clipping
    """
    if error_types is None:
        error_types = ["abs_traj_err", "abs_H_err"]
    
    if color_limits is None:
        color_limits = [(-4, 4)] * len(error_types)

    if figure_size is None:
        figure_size = (5, 4 * len(error_types))

    # Create subplots
    fig, axes = plt.subplots(len(error_types), 1, figsize=figure_size, layout='constrained')
    if len(error_types) == 1:
        axes = [axes]
    
    # Compute error matrices for both groups
    error_matrices_group1 = defaultdict(list)
    for trajectory in trajectories_group1:
        time_points, error_data = trajectory.compare(reference_trajectory)
        for error_type in error_types:
            error_matrices_group1[error_type].append(error_data[error_type][1:])

    error_matrices_group2 = defaultdict(list)
    for trajectory in trajectories_group2:
        time_points, error_data = trajectory.compare(reference_trajectory)
        for error_type in error_types:
            error_matrices_group2[error_type].append(error_data[error_type][1:])

    # Create difference heatmaps
    for i, (error_type, error_matrix_list1) in enumerate(error_matrices_group1.items()):
        error_matrix1 = np.array(error_matrix_list1)
        error_matrix2 = np.array(error_matrices_group2[error_type])
        
        # Clip error values to specified range
        error_matrix1[error_matrix1 < 10**min_log_error] = 10**min_log_error
        error_matrix2[error_matrix2 < 10**min_log_error] = 10**min_log_error
        error_matrix1[error_matrix1 > 10**max_log_error] = 10**max_log_error
        error_matrix2[error_matrix2 > 10**max_log_error] = 10**max_log_error
        
        # Compute log difference
        log_error_difference = np.log10(error_matrix1) - np.log10(error_matrix2)
        
        im = axes[i].imshow(
            log_error_difference, 
            cmap='coolwarm', 
            interpolation='none', 
            aspect='auto', 
            clim=color_limits[i], 
            extent=[time_points[1], time_points[-1], len(trajectories_group1), 0]
        )
        axes[i].set_title(f"Difference in {error_type}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Iteration")
        fig.colorbar(im, ax=axes[i])
    
    # Save or display
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Figure saved as '{save_path}'")
    else:
        plt.show()

    return


def create_phase_space_plot(
    trajectories: List[Any],
    x_indices: List[int],
    y_indices: List[int],
    axis_limits: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    line_styles: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    marker_size: float = 0.5,
    legend_labels: Optional[List[str]] = None,
    axis_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    figure_size: Tuple[float, float] = (6, 4),
    show_legend: bool = False,
    save_path: Optional[str] = None,
    aspect_ratio: str = 'equal'
) -> None:
    """
    Generic phase space plotting function for any coordinate system.
    
    Args:
        trajectories: List of trajectory objects to plot
        x_indices: List of indices for x-coordinates in state vector
        y_indices: List of indices for y-coordinates in state vector
        axis_limits: List of (min, max) tuples for x and y axes
        line_styles: List of line styles for each trajectory
        colors: List of colors for each trajectory
        marker_size: Size of markers on the plot
        legend_labels: Labels for the legend
        axis_labels: Labels for x and y axes
        title: Plot title
        figure_size: Figure dimensions (width, height)
        show_legend: Whether to display the legend
        save_path: Path to save the figure
        aspect_ratio: Aspect ratio setting for the plot
    """
    # Set default values
    if axis_limits is None:
        axis_limits = [(None, None)] * 2

    if line_styles is None:
        line_styles = get_line_styles("default")[:len(trajectories)]
    elif isinstance(line_styles, str):
        line_styles = [line_styles] * len(trajectories)

    if legend_labels is None:
        legend_labels = [f'Trajectory {i+1}' for i in range(len(trajectories))]

    if colors is None:
        colors = get_color_palette("default")
    color_cycle = cycle(colors)
    
    if axis_labels is None:
        axis_labels = [f'$x_{{{idx}}}$' for idx in x_indices] + [f'$y_{{{idx}}}$' for idx in y_indices]

    # Create figure and axis
    fig = plt.figure(figsize=figure_size, layout='constrained')
    ax = fig.subplots(1, 1)

    # Plot each trajectory
    plotted_lines = []
    for trajectory, line_style, legend_label in zip(trajectories, line_styles, legend_labels):
        states = trajectory.states.u 
        
        # Extract coordinates based on indices
        x_coords = states[:, x_indices[0]] if len(x_indices) == 1 else states[:, x_indices]
        y_coords = states[:, y_indices[0]] if len(y_indices) == 1 else states[:, y_indices]
        
        if legend_label == 'ref':
            line, = ax.plot(x_coords, y_coords, line_style, markersize=marker_size, 
                           color='k', alpha=0.3, label=legend_label)
        else:
            line, = ax.plot(x_coords, y_coords, line_style, markersize=marker_size, 
                           color=next(color_cycle), label=legend_label)
        plotted_lines.append(line)

    # Configure plot appearance
    if axis_limits is not None:
        ax.set_xlim(axis_limits[0])
        ax.set_ylim(axis_limits[1])
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_aspect(aspect_ratio)

    if title is not None:
        ax.set_title(title)

    if show_legend:
        fig.legend(plotted_lines, legend_labels, loc='outside center right', borderaxespad=0.1)

    # Save or display
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"Figure saved as '{save_path}'")
    else:
        plt.show()

    return
