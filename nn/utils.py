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
