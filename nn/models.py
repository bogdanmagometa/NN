import random
from typing import List
from abc import ABC, abstractmethod

from .nn import Value

class Model(ABC):
    @abstractmethod
    def __call__(self, X: List[List[float]]):
        raise NotImplemented
    @abstractmethod
    def parameters(self):
        raise NotImplemented

class Perceptron(Model):
    def __init__(self, no_weights: int):
        assert no_weights > 0, "Number of weights should be greater than 0"
        self._weights = [Value(random.random() * 2 - 1) for _ in range(no_weights)]
        self._bias = Value(random.random() * 2 - 1)

    def __call__(self, x: List[float]) -> Value:
        logit = self._bias
        for regressor, weight in zip(x, self._weights):
            logit = logit + regressor * weight
        return logit.sigmoid()

    def parameters(self):
        return [self._bias] + self._weights