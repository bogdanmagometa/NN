import random
from typing import List
from abc import ABC, abstractmethod

import numpy as np

from .tensor import Tensor

class Module(ABC):
    @abstractmethod
    def __call__(self, X: List[List[float]]):
        raise NotImplemented
    @abstractmethod
    def parameters(self):
        raise NotImplemented

class Linear(Module):
    def __init__(self, num_inputs, num_outputs):
        assert num_inputs > 0 and num_outputs > 0
        self._num_outputs = num_outputs
        self._weights = Tensor(np.random.randn(num_inputs, num_outputs))
        self._biases = Tensor(np.random.randn(num_outputs))

    def __call__(self, X: Tensor):
        logits = X.tensordot(self._weights, 1)
        logits_shape = np.shape(logits.data)
        logits = logits + self._biases.reshape((1, logits_shape[1])).broadcast_to(logits_shape)
        return logits

    def parameters(self):
        return [self._biases, self._weights]

class SoftmaxActivation(Module):
    def __init__(self, num_nodes: int):
        assert num_nodes > 0
        self._num_nodes = num_nodes

    def __call__(self, X: Tensor):
        exponentiated = np.e**X
        denom = exponentiated.sum(1)

        X_shape = np.shape(X.data)

        denom = denom.reshape((X_shape[0], 1)).broadcast_to(X_shape)

        activations = exponentiated / denom

        return activations

    def parameters(self):
        return []

class ReLUActivation(Module):
    def __init__(self, num_nodes: int):
        assert num_nodes > 0
        self._num_nodes = num_nodes

    def __call__(self, X: Tensor):
        return X.relu()

    def parameters(self):
        return []

class SigmoidActivation(Module):
    def __init__(self, num_nodes: int):
        assert num_nodes > 0
        self._num_nodes = num_nodes

    def __call__(self, X: Tensor):
        return X.sigmoid()

    def parameters(self):
        return []

