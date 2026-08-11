"""
Simple MATLAB engine management for visualization compatibility.
"""

try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False


def is_matlab_available() -> bool:
    """Check if MATLAB is available on the system."""
    return MATLAB_AVAILABLE


def get_matlab_engine():
    """Get MATLAB engine instance."""
    if not MATLAB_AVAILABLE:
        raise ImportError("MATLAB Engine for Python not available")
    try:
        return matlab.engine.start_matlab()
    except Exception as e:
        raise RuntimeError(f"Failed to start MATLAB engine: {e}")
