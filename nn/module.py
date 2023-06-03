import random
from typing import List
from abc import ABC, abstractmethod

import numpy as np

from .nn import Value

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
        self._weights = [[Value(random.random() * 2 - 1) for _ in range(num_inputs)] for _ in range(num_outputs)]
        self._biases = [Value(random.random() * 2 - 1) for _ in range(num_outputs)]

    def __call__(self, x: List[Value]):
        logits = self._biases.copy()
        for out_idx, out_weights in enumerate(self._weights):
            for in_idx, weight in enumerate(out_weights):
                logits[out_idx] += weight * x[in_idx]
        return logits

    def parameters(self):
        return self._biases + sum(self._weights, [])

class SoftmaxActivation(Module):
    def __init__(self, num_nodes: int):
        assert num_nodes > 0
        self._num_nodes = num_nodes

    def __call__(self, x: List[Value]):
        exponentiated = [np.e**regressor for regressor in x]
        denom = exponentiated[0]
        for i in range(1, self._num_nodes):
            denom = denom + exponentiated[i]
        return [activation / denom for activation in exponentiated]

    def parameters(self):
        return []

class ReLUActivation(Module):
    def __init__(self, num_nodes: int):
        assert num_nodes > 0
        self._num_nodes = num_nodes

    def __call__(self, x: List[Value]):
        return [x_i.relu() for x_i in x]

    def parameters(self):
        return []

class SigmoidActivation(Module):
    def __init__(self, num_nodes: int):
        assert num_nodes > 0
        self._num_nodes = num_nodes

    def __call__(self, x: List[Value]):
        return [x_i.sigmoid() for x_i in x]

    def parameters(self):
        return []

