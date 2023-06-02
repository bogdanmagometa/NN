"""
nn.py

Contains implementation of Value
"""

from typing import List, Optional, Union, Set

import numpy as np

from . import utils


class Value:
    def __init__(self, data: float, label: Optional[str] = None):
        assert isinstance(data, (int, float))
        self.data = data
        self.label = None
        self._children = []
        self.grad = 0.0
        self._backward = lambda: None

    def __add__(self, other: Union['Value', float]) -> 'Value':
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data)
        out._children = [self, other]

        def backward():
            # this should be called when gradient w.r.t. out is already calculated
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward

        return out

    def __radd__(self, other: float) -> 'Value':
        return self + other

    def __sub__(self, other: Union['Value', float]) -> 'Value':
        return self + (-1) * other

    def __rsub__(self, other: float) -> 'Value':
        return Value(other) + (-1) * self

    def __mul__(self, other: 'Value') -> 'Value':
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data)
        out._children = [self, other]

        def backward():
            # this should be called when gradient w.r.t. out is already calculated
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward

        return out

    def __rmul__(self, other: float):
        return self * other

    def __pow__(self, exponent: Union['Value', float]) -> 'Value':
        if not isinstance(exponent, Value):
            exponent = Value(exponent)

        out = Value(self.data**exponent.data)
        out._children = [self, exponent]

        def backward():
            # this should be called when gradient w.r.t. out is already calculated
            local_grad_wrt_base = exponent.data * self.data**(exponent.data - 1)
            local_grad_wrt_exponent = np.log(self.data) * out.data
            self.grad += local_grad_wrt_base * out.grad
            exponent.grad += local_grad_wrt_exponent * out.grad

        out._backward = backward

        return out

    def __rpow__(self, base: Union['Value', float]) -> 'Value':
        return Value(base) ** self

    def __neg__(self):
        return self * (-1)

    def __div__(self, other: Union['Value', float]):
        return self * other**(-1)

    def __rdiv__(self, other: float):
        return other * self**(-1)

    def sigmoid(self) -> 'Value':
        out = Value(utils.sigmoid(self.data))
        out._children = [self]

        def backward():
            # this should be called when gradient w.r.t. out is already calculated
            local_gradient = out.data * (1 - out.data)
            self.grad += local_gradient * out.grad

        out._backward = backward

        return out

    def log(self):
        out = Value(np.log(self.data))
        out._children = [self]

        def backward():
            # this should be called when gradient w.r.t. out is already calculated
            self.grad += 1 / self.data * out.grad

        out._backward = backward

        return out

    @staticmethod
    def _toposort(first_value: 'Value') -> List['Value']:
        """Return topologically ordered nodes of the computational graph.
        """
        topo_order = []
        visited = {first_value}

        def dfs(value: Value):
            for child in value._children:
                if child not in visited:
                    visited.add(value)
                    dfs(child)
            # ------ post order action ------
            topo_order.append(value)
            # --- end of post order action --

        dfs(first_value)

        return topo_order

    @staticmethod
    def _dfs(start_value: 'Value'):
        frontier = [start_value]
        visited = {start_value}        
        while frontier:
            cur_value = frontier.pop()
            yield cur_value
            for child in cur_value._children:
                if child not in visited:
                    visited.add(cur_value)
                    frontier.append(child)

    @staticmethod
    def back_prop(value: 'Value') -> None:
        """Calculate and set the gradients of value and of its "descendants".

        Parameters
        ----------
        value: Value
        """

        value.grad = 1
        for val in reversed(Value._toposort(value)):
            val._backward()

    @staticmethod
    def zeroout_grad(start_value: 'Value'):
        for val in Value._dfs(start_value):
            val.grad = 0
