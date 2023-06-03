from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    def zero_grad(self):
        for parameter in self._parameters:
            parameter.grad = 0
    @abstractmethod
    def step(self):
        pass


class RMSPropOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0):
        self._parameters = params
        self._lr = lr
        self._alpha = alpha
        self._eps = eps
        self._weight_decay = weight_decay

        self._cur_iteration = 1

        self._second_moment = np.zeros(len(params))

    def step(self):
        gradient = np.array([param.grad for param in self._parameters])
        if self._weight_decay != 0:
            gradient += self._weight_decay * np.array([param.data for param in self._parameters])

        self._second_moment = self._alpha * self._second_moment + (1 - self._alpha) * gradient**2

        corrected_second_moment = self._second_moment / (1 - self._alpha**self._cur_iteration)

        delta_parameters = - self._lr * gradient / (np.sqrt(corrected_second_moment) + self._eps)

        for param_idx, param in enumerate(self._parameters):
            param.data += delta_parameters[param_idx]

        self._cur_iteration += 1


class AdamOptimizer(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self._parameters = params
        self._lr = lr
        self._betas = betas
        self._eps = eps
        self._weight_decay = weight_decay

        self._cur_iteration = 1
        self._moment = np.zeros(len(params))
        self._second_moment = np.zeros(len(params))

    def step(self):
        gradient = np.array([param.grad for param in self._parameters])
        if self._weight_decay != 0:
            gradient += self._weight_decay * np.array([param.data for param in self._parameters])

        self._moment = self._betas[0] * self._moment + (1 - self._betas[0]) * gradient
        self._second_moment = self._betas[1] * self._second_moment + (1 - self._betas[1]) * gradient**2

        corrected_moment = self._moment / (1 - self._betas[0]**self._cur_iteration)
        corrected_second_moment = self._second_moment / (1 - self._betas[1]**self._cur_iteration)

        delta_parameters = - self._lr * corrected_moment / (np.sqrt(corrected_second_moment) + self._eps)

        for param_idx, param in enumerate(self._parameters):
            param.data += delta_parameters[param_idx]

        self._cur_iteration += 1
