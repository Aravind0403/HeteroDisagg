"""Pytest configuration and environment initialization."""

import os

# Prevent OpenMP runtime abort on macOS when NumPy and PyTorch both link OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
