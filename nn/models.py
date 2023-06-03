import random
from typing import List

from nn.module import Linear, SoftmaxActivation
from nn.nn import Value

class Perceptron:
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

