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
    return np.maximum(x, 0)

def derivative_wrt_itself(shape):
    grids = np.meshgrid(*map(range, shape), *map(range, shape), indexing='ij')
    der = np.ones(shape + shape, dtype=np.bool8)
    for i in range(len(grids) // 2):
        d = (grids[i] == grids[len(shape) + i])
        der &= d
    return der.astype(np.float64)

def derivative_wrt_left_matrix_product_factor(left_shape: tuple, rmatr: Union[np.ndarray, int, float, np.float64], axes: int):
    right_shape = np.shape(rmatr)
    output_shape = left_shape[:-axes] + right_shape[axes:]
    grids = np.meshgrid(*(range(l) for l in output_shape + left_shape), indexing='ij')

    i = grids[:(len(left_shape) - axes)]
    j = grids[(len(left_shape) - axes):len(output_shape)]
    r = grids[len(output_shape):(len(grids) - axes)]
    f = grids[(len(grids) - axes):]

    condition = np.ones(output_shape + left_shape, dtype=np.bool8)
    for c in map(lambda x, y: x == y, i, r):
        condition &= c

    return rmatr[tuple(f + j)] * condition

def derivative_wrt_right_matrix_product_factor(lmatr: Union[np.ndarray, int, float, np.float64], right_shape: tuple, axes: int):
    # TODO: the following was done intuitively, check if it is correct
    left_shape = np.shape(lmatr)
    output_shape = left_shape[:-axes] + right_shape[axes:]
    derivative_shape = output_shape + right_shape

    local_derivative_wrt_right = derivative_wrt_left_matrix_product_factor(tuple(reversed(right_shape)), np.transpose(lmatr), axes)
    permutation = list(reversed(range(len(output_shape))))
    permutation += list(reversed(range(len(output_shape), len(derivative_shape))))
    local_derivative_wrt_right = np.transpose(local_derivative_wrt_right, permutation)

    return local_derivative_wrt_right

if __name__ == "__main__":
    print(derivative_wrt_itself((2, 2)))
