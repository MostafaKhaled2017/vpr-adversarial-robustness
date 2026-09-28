import numpy as np

# SuperVLAD's datasets_ws uses np.float, removed in NumPy 1.24.
if not hasattr(np, "float"):
    np.float = float

__all__ = []
