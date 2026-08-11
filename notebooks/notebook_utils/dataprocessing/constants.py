"""
Global constants for the dataprocessing package.
"""

import os

# Base output directory for all trajectory files (dynamically resolved relative to project root)
BASE_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "out"))
