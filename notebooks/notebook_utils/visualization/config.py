"""
Matplotlib configuration and styling utilities.

This module provides functions to set up consistent plotting styles
for scientific publications and analysis.
"""

import matplotlib.pyplot as plt


def setup_matplotlib_style(style_name: str = "publication") -> None:
    """
    Configure matplotlib with predefined styles for different use cases.
    
    Args:
        style_name: Name of the style to apply. Options: "publication", "default"
    """
    if style_name == "publication":
        # Publication-quality style settings
        font = {
            'family': 'sans-serif',
            'weight': 'normal',
            'size': 10
        }
        axes = {'linewidth': 0.4}
        lines = {
            'linewidth': 0.7,
            'markersize': 3.0,
            'markeredgewidth': 0.4
        }
        mathtext = {'fontset': 'cm'}
        xtick = {
            'labelsize': 'small',
            'major.size': 2,
            'minor.size': 1
        }
        ytick = {
            'labelsize': 'small',
            'major.size': 2,
            'minor.size': 1
        }
        
        plt.rc('font', **font)
        plt.rc('axes', **axes)
        plt.rc('lines', **lines)
        plt.rc('mathtext', **mathtext)
        plt.rc('xtick', **xtick)
        plt.rc('ytick', **ytick)
        
    elif style_name == "default":
        # Reset to matplotlib defaults
        plt.rcdefaults()
    
    else:
        raise ValueError(f"Unknown style name: {style_name}")


def get_color_palette(palette_name: str = "default") -> list:
    """
    Get predefined color palettes for consistent plotting.
    
    Args:
        palette_name: Name of the color palette
        
    Returns:
        List of color codes
    """
    palettes = {
        "default": plt.rcParams['axes.prop_cycle'].by_key()['color'],
        "parareal": ["C0", "C2", "C1", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
        "convergence": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    }
    
    return palettes.get(palette_name, palettes["default"])


def get_line_styles(style_name: str = "default") -> list:
    """
    Get predefined line style patterns.
    
    Args:
        style_name: Name of the line style pattern
        
    Returns:
        List of line style strings
    """
    styles = {
        "default": ["-"] * 10,
        "parareal": ["o-", "o--", "x-", "x--", "s-", "s--", "^-", "^--", "v-", "v--"],
        "convergence": ["o-", "s-", "^-", "v-", "D-"]
    }
    
    return styles.get(style_name, styles["default"])
