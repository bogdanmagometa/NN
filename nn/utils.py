"""
utilities.py

Useful functions
"""

from typing import Union

import numpy as np


def sigmoid(x):
    """
    Parameters
    ----------
    x: float or N-D numpy array
    """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """
    Parameters
    ----------
    x: float or N-D numpy array
    """
    return np.max(x, 0)

def derivative_wrt_itself(shape):
    grids = np.meshgrid(*map(range, shape), *map(range, shape))
    der = np.ones(shape + shape, dtype=np.bool8)
    for i in range(len(grids) // 2):
        der &= (grids[i] == grids[len(shape) + i])
    return der.astype(np.float64)

if __name__ == "__main__":
    print(derivative_wrt_itself((2, 2)))
