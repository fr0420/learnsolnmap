"""
MATLAB-style plotting configurations.
"""

from typing import Dict, Any


def get_matlab_defaults() -> Dict[str, Any]:
    """Get MATLAB default plotting parameters."""
    return {
        'LineWidth': 0.5,
        'MarkerSize': 6,
        'FontSize': 10,
        'FontName': 'Helvetica',
        'Color': [0, 0.4470, 0.7410],  # MATLAB blue
        'Grid': True,
        'Box': 'on',
        'TickDir': 'out',
        'TickLength': [0.01, 0.025],
        'XMinorTick': 'on',
        'YMinorTick': 'on'
    }


def apply_matlab_style(engine):
    """Apply MATLAB default style to current figure."""
    defaults = get_matlab_defaults()
    
    # Set default line properties
    engine.set(engine.gca(), 'LineWidth', defaults['LineWidth'])
    engine.set(engine.gca(), 'FontSize', defaults['FontSize'])
    engine.set(engine.gca(), 'FontName', defaults['FontName'])
    
    # Enable grid
    if defaults['Grid']:
        engine.grid('on')
    
    # Set box
    engine.set(engine.gca(), 'Box', defaults['Box'])
    
    # Set tick properties
    engine.set(engine.gca(), 'TickDir', defaults['TickDir'])
    engine.set(engine.gca(), 'TickLength', matlab.double(defaults['TickLength']))
    engine.set(engine.gca(), 'XMinorTick', defaults['XMinorTick'])
    engine.set(engine.gca(), 'YMinorTick', defaults['YMinorTick'])


def set_matlab_colors(engine):
    """Set MATLAB default color scheme."""
    colors = [
        [0, 0.4470, 0.7410],      # Blue
        [0.8500, 0.3250, 0.0980], # Orange
        [0.9290, 0.6940, 0.1250], # Yellow
        [0.4940, 0.1840, 0.5560], # Purple
        [0.4660, 0.6740, 0.1880], # Green
        [0.3010, 0.7450, 0.9330], # Light blue
        [0.6350, 0.0780, 0.1840]  # Red
    ]
    
    # Set color order
    engine.set(engine.gca(), 'ColorOrder', matlab.double(colors))


def set_publication_style(engine):
    """Set publication-quality MATLAB style."""
    # Set figure properties
    engine.set(engine.gcf(), 'Color', 'white')
    engine.set(engine.gcf(), 'PaperPositionMode', 'auto')
    
    # Set axes properties
    ax = engine.gca()
    engine.set(ax, 'FontSize', 12)
    engine.set(ax, 'FontName', 'Times New Roman')
    engine.set(ax, 'LineWidth', 1.5)
    engine.set(ax, 'TickDir', 'out')
    engine.set(ax, 'TickLength', matlab.double([0.02, 0.04]))
    engine.set(ax, 'Box', 'on')
    
    # Enable grid
    engine.grid('on')
    engine.set(ax, 'GridLineStyle', ':')
    engine.set(ax, 'GridAlpha', 0.3)


def set_alphaparticle_style(engine):
    """Set specialized style for α-particle plots."""
    # Apply base publication style
    set_publication_style(engine)
    
    # Set specific properties for α-particle visualization
    ax = engine.gca()
    engine.set(ax, 'FontSize', 14)
    engine.set(ax, 'LineWidth', 2.0)
    
    # Set aspect ratio for phase space plots
    engine.axis('equal')
    
    # Set color scheme suitable for magnetic field visualization
    engine.colormap('coolwarm')
    
    # Enable minor grid for better readability
    engine.set(ax, 'XMinorGrid', 'on')
    engine.set(ax, 'YMinorGrid', 'on')
    engine.set(ax, 'MinorGridLineStyle', ':')
    engine.set(ax, 'MinorGridAlpha', 0.2)


def set_figure_size(engine, width: float, height: float, units: str = 'inches'):
    """Set figure size in specified units."""
    if units == 'inches':
        engine.set(engine.gcf(), 'Units', 'inches')
        engine.set(engine.gcf(), 'Position', matlab.double([1, 1, width, height]))
    elif units == 'centimeters':
        engine.set(engine.gcf(), 'Units', 'centimeters')
        engine.set(engine.gcf(), 'Position', matlab.double([2.54, 2.54, width, height]))
    elif units == 'pixels':
        engine.set(engine.gcf(), 'Units', 'pixels')
        engine.set(engine.gcf(), 'Position', matlab.double([100, 100, width, height]))


def set_line_properties(engine, line_handle, **kwargs):
    """Set properties for a specific line."""
    if 'linewidth' in kwargs:
        engine.set(line_handle, 'LineWidth', kwargs['linewidth'])
    if 'markersize' in kwargs:
        engine.set(line_handle, 'MarkerSize', kwargs['markersize'])
    if 'color' in kwargs:
        engine.set(line_handle, 'Color', kwargs['color'])
    if 'linestyle' in kwargs:
        engine.set(line_handle, 'LineStyle', kwargs['linestyle'])
    if 'marker' in kwargs:
        engine.set(line_handle, 'Marker', kwargs['marker'])


def set_contour_properties(engine, contour_handle, **kwargs):
    """Set properties for contour plots."""
    if 'linewidth' in kwargs:
        engine.set(contour_handle, 'LineWidth', kwargs['linewidth'])
    if 'color' in kwargs:
        engine.set(contour_handle, 'Color', kwargs['color'])
    if 'linestyle' in kwargs:
        engine.set(contour_handle, 'LineStyle', kwargs['linestyle'])
    if 'alpha' in kwargs:
        engine.set(contour_handle, 'FaceAlpha', kwargs['alpha'])
        engine.set(contour_handle, 'EdgeAlpha', kwargs['alpha'])


def set_poincare_style(engine):
    """Set specialized style for α-particle Poincaré sections."""
    # Apply base α-particle style
    set_alphaparticle_style(engine)
    
    # Additional Poincaré-specific settings
    ax = engine.gca()
    
    # Set marker properties for Poincaré points
    engine.set(ax, 'MarkerSize', 0.5)
    engine.set(ax, 'MarkerEdgeColor', 'none')
    
    # Set grid properties for better point visibility
    engine.set(ax, 'GridLineStyle', ':')
    engine.set(ax, 'GridAlpha', 0.3)
    engine.set(ax, 'MinorGridLineStyle', ':')
    engine.set(ax, 'MinorGridAlpha', 0.1)
    
    # Set axis properties for phase space
    engine.set(ax, 'TickDir', 'out')
    engine.set(ax, 'TickLength', matlab.double([0.015, 0.03]))
    
    # Set color scheme for magnetic field
    engine.colormap('coolwarm')
    
    # Set aspect ratio for equal scaling
    engine.axis('equal')


def set_publication_poincare_style(engine):
    """Set publication-quality style for α-particle Poincaré sections."""
    # Apply base Poincaré style
    set_poincare_style(engine)
    
    # Publication-specific enhancements
    ax = engine.gca()
    
    # Larger fonts for publication
    engine.set(ax, 'FontSize', 16)
    engine.set(ax, 'FontName', 'Times New Roman')
    
    # Thicker lines
    engine.set(ax, 'LineWidth', 2.0)
    
    # Enhanced grid
    engine.set(ax, 'GridLineStyle', '-')
    engine.set(ax, 'GridAlpha', 0.2)
    engine.set(ax, 'MinorGridLineStyle', ':')
    engine.set(ax, 'MinorGridAlpha', 0.1)
    
    # Set high-quality rendering
    engine.set(engine.gcf(), 'Renderer', 'painters')
    engine.set(engine.gcf(), 'PaperPositionMode', 'auto')
    engine.set(engine.gcf(), 'Color', 'white')


def set_poincare_colors(engine, n_trajectories: int):
    """Set color scheme for multiple Poincaré trajectories."""
    # Use MATLAB's jet colormap for trajectory colors
    colors = []
    for i in range(n_trajectories):
        t = i / max(1, n_trajectories - 1)
        if t < 0.125:
            r, g, b = 0, 0, 0.5 + 4 * t
        elif t < 0.375:
            r, g, b = 0, 4 * (t - 0.125), 1
        elif t < 0.625:
            r, g, b = 4 * (t - 0.375), 1, 1 - 4 * (t - 0.375)
        elif t < 0.875:
            r, g, b = 1, 1 - 4 * (t - 0.625), 0
        else:
            r, g, b = 1 - 4 * (t - 0.875), 0, 0
        colors.append([r, g, b])
    
    # Set color order
    engine.set(engine.gca(), 'ColorOrder', matlab.double(colors))


def set_magnetic_field_style(engine, alpha: float = 0.1):
    """Set style for magnetic field contour overlay."""
    # Set colormap for magnetic field
    engine.colormap('coolwarm')
    
    # Set transparency
    ax = engine.gca()
    engine.set(ax, 'Colorbar', 'off')  # Hide colorbar for cleaner look
    
    # Set contour line properties
    engine.set(ax, 'ContourLineWidth', 0.5)
    engine.set(ax, 'ContourAlpha', alpha)
