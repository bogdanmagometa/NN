from typing import List, Union, Optional
import numpy as np

from . import utils

class Tensor:
    def __init__(self, data: Union[np.ndarray, float, np.float64]):
        if isinstance(data, np.ndarray):
            assert data.dtype == np.float64
        self.data = data
        self._children = []
        self._backward = lambda: None
        
        # this will be a tensor when back_prob is run
        # for now make it zero because we don't know what are the dimentions of the tensor that we will differentiate
        # TODO: maybe restrict that we can only differentiate a 0-D tensor (i.e. when data is 0-D numpy array)
        self.grad = 0

    @staticmethod
    def _assert_type(other):
        assert isinstance(other, (Tensor, np.ndarray, float, int, np.float64))
        if not isinstance(other, (Tensor, np.ndarray)):
            other = np.array(other, dtype=np.float64)
            other = Tensor(other)
        if isinstance(other, np.ndarray):
            assert other.dtype == np.float64
            other = Tensor(other)
        return other

    def __add__(self, other: Union['Tensor', np.ndarray, float, int, np.float64]):
        other = Tensor._assert_type(other)
        assert np.shape(other.data) == np.shape(self.data)

        out = Tensor(self.data + other.data)
        out._children = [self, other]

        def backward():
            self.grad = self.grad + out.grad
            other.grad = other.grad + out.grad

        out._backward = backward

        return out

    def __radd__(self, other: Union['Tensor', np.ndarray]):
        return self + other

    def __mul__(self, other: Union['Tensor', np.ndarray]):
        other = Tensor._assert_type(other)
        assert np.shape(other.data) == np.shape(self.data)

        out = Tensor(np.array(self.data * other.data))
        out._children = [self, other]

        def backward():
            ndims_to_add = list(range(np.ndim(out.grad) - np.ndim(self.data)))
            self.grad = self.grad + np.expand_dims(other.data, ndims_to_add) * out.grad
            other.grad = other.grad + np.expand_dims(self.data, ndims_to_add) * out.grad

        out._backward = backward

        return out

    def __truediv__(self, other: Union['Tensor', np.ndarray, int, float, np.float64]):
        return self * other ** (-1)

    def __rtruediv__(self, other: Union[np.ndarray, int, float, np.float64]):
        return Tensor(other) / self

    def __rmul__(self, other: Union['Tensor', np.ndarray]):
        return self * other

    def __neg__(self):
        return (-1) * self

    def __sub__(self, other: Union['Tensor', np.ndarray, int, float, np.float64]):
        return self + (-1) * other
    
    def __rsub__(self, other: Union['Tensor', np.ndarray, int, float, np.float64]):
        return other + (-1) * self

    def sum(self, axis: Optional[Union[int, List[int]]] = None) -> 'Tensor':
        if isinstance(axis, int):
            assert 0 <= axis < np.ndim(self.data)
            axis = [axis]
        elif axis is None:
            axis = list(range(np.ndim(self.data)))
            axis = sorted(set(axis))
        for ax in axis:
            assert 0 <= ax < np.ndim(self.data)
        axis = tuple(axis)

        out_data = np.sum(self.data, axis, keepdims=True)
        sum_shape = np.shape(out_data)
        out = Tensor(np.sum(out_data, axis))
        out._children = [self]

        def backward():
            out_data = out.data
            #TODO: refactor
            out_shape = np.shape(out_data)
            global_grad_shape = np.shape(out_data)[:len(out_shape)-len(sum_shape)] + np.shape(self.data)
            global_grad = np.broadcast_to(np.reshape(out.grad, np.shape(out_data)[:len(out_shape)-len(sum_shape)] + sum_shape), global_grad_shape)
            self.grad = np.add(self.grad, global_grad)

        out._backward = backward

        return out

    # @staticmethod
    # def concatenate(self, lst: List[Union['Tensor', np.ndarray]], axis: int):
    #     assert isinstance(axis, int)
    #     assert axis >= 0

    #     #TODO: finish method
    #     lst = [Tensor._assert_type(item) for item in lst]
    #     assert all(tensor.data.shape for tensor in lst)

    def __pow__(self, exponent: Union['Tensor', np.ndarray, float, int, np.float64]):
        propagate_exponent = isinstance(exponent, Tensor)
        if propagate_exponent:
            return self._exponentiate(exponent)
        else:
            return self._power(exponent)

    def __rpow__(self, base: Union[np.ndarray, float, int, np.float64]):
        if isinstance(base, np.ndarray):
            assert np.shape(base) == self.data.shape
            assert base.dtype == np.float64
        out = Tensor(base ** self.data)
        out._children = [self]

        def backward():
            ndims_to_add = list(range(np.ndim(out.grad) - np.ndim(self.data)))
            local_grad = np.log(base) * out.data
            self.grad = np.add(self.grad, np.expand_dims(local_grad, ndims_to_add) * out.grad)

        out._backward = backward

        return out

    def _exponentiate(self, exponent: 'Tensor') -> 'Tensor':
        exponent = Tensor._assert_type(exponent)
        
        out = Tensor(self.data ** exponent.data)
        out._children = [self, exponent]

        def backward():
            ndims_to_add = list(range(np.ndim(out.grad) - np.ndim(self.data)))
            local_grad_wrt_base = exponent.data * self.data ** (exponent.data - 1)
            local_grad_wrt_exponent = np.log(self.data) * out.data

            self.grad = np.add(self.grad, np.expand_dims(local_grad_wrt_base, ndims_to_add) * out.grad)
            exponent.grad = np.add(exponent.grad, np.expand_dims(local_grad_wrt_exponent, ndims_to_add) * out.grad)

        out._backward = backward
        return out

    def _power(self, exponent):
        if isinstance(exponent, np.ndarray):
            assert exponent.shape == np.shape(self.data)
            assert exponent.dtype == np.float64
        out = Tensor(self.data ** exponent)
        out._children = [self]
        def backward():
            ndims_to_add = list(range(np.ndim(out.grad) - np.ndim(self.data)))
            local_gradient = exponent * self.data**(exponent - 1)
            self.grad = np.add(self.grad, np.expand_dims(local_gradient, ndims_to_add) * out.grad)

        out._backward = backward
        return out

    def reshape(self, newshape) -> 'Tensor':
        out = Tensor(np.reshape(self.data, newshape))
        out._children = [self]
        
        def backward():
            oldshape = np.shape(self.data)
            grad_shape = np.shape(out.grad)[:np.ndim(out.grad)-np.ndim(out.data)] + oldshape #TODO: maybe I goofed somewhere else also???
            self.grad = np.add(self.grad, np.reshape(out.grad, grad_shape))

        out._backward = backward

        return out

    def broadcast_to(self, newshape) -> 'Tensor':
        out = Tensor(np.broadcast_to(self.data, newshape))
        out._children = [self]

        def backward():
            one_element_dimentions = np.argwhere(np.array(np.shape(self.data)) == 1).reshape(-1)
            one_element_dimentions = tuple(one_element_dimentions)
            self.grad = np.add(self.grad, np.sum(out.grad, axis=one_element_dimentions, keepdims=True))

        out._backward = backward

        return out

    def sigmoid(self) -> 'Tensor':
        out = Tensor(utils.sigmoid(self.data))
        out._children = [self]

        def backward():
            ndims_to_add = list(range(np.ndim(out.grad) - np.ndim(self.data)))
            local_gradient = out.data * (1 - out.data)
            self.grad = np.add(self.grad, np.expand_dims(local_gradient, ndims_to_add) * out.grad)

        out._backward = backward

        return out

    def log(self) -> 'Tensor':
        out = Tensor(np.log(self.data))
        out._children = [self]

        def backward():
            ndims_to_add = list(range(np.ndim(out.grad) - np.ndim(self.data)))
            local_grad = self.data ** (-1)
            self.grad = np.add(self.grad, np.expand_dims(local_grad, ndims_to_add) * out.grad)

        out._backward = backward

        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0))
        out._children = [self]

        def backward():
            ndims_to_add = list(range(np.ndim(out.grad) - np.ndim(self.data)))
            local_gradient = (self.data > 0).astype(np.float64)
            self.grad = np.add(self.grad, np.expand_dims(local_gradient, ndims_to_add) * out.grad)

        out._backward = backward

        return out

    def tensordot(self, other: Union['Tensor', np.ndarray, int, float, np.float64], axes: int) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        assert axes <= np.ndim(self.data) and axes <= np.ndim(other.data)
        self_shape = np.shape(self.data)
        assert self_shape[len(self_shape)-axes:] == np.shape(other.data)[:axes]

        out = Tensor(np.tensordot(self.data, other.data, axes))

        out._children = [self, other]

        def backward():
            # wrt self
            local_derivative_wrt_left = utils.derivative_wrt_left_matrix_product_factor(np.shape(self.data), other.data, axes)
            global_derivative_wrt_left = np.tensordot(out.grad, local_derivative_wrt_left, np.ndim(out.data))
            self.grad = np.add(self.grad, global_derivative_wrt_left)
            # wrt other
            local_derivative_wrt_right = utils.derivative_wrt_right_matrix_product_factor(self.data, np.shape(other.data), axes)
            global_derivative_wrt_right = np.tensordot(out.grad, local_derivative_wrt_right, np.ndim(out.data))
            other.grad = np.add(other.grad, global_derivative_wrt_right)
        out._backward = backward

        return out

    @staticmethod
    def _toposort(first_tensor: 'Tensor') -> List['Tensor']:
        """Return topologically ordered nodes of the computational graph.
        """
        topo_order = []
        visited = {first_tensor}

        def dfs(value: Tensor):
            for child in value._children:
                if child not in visited:
                    visited.add(value)
                    dfs(child)
            # ------ post order action ------
            topo_order.append(value)
            # --- end of post order action --

        dfs(first_tensor)

        return topo_order

    def back_prop(self) -> None:
        """Calculate and set the gradients of the tensor and of its "descendants".
        """

        self.grad = utils.derivative_wrt_itself(self.data.shape)
        for val in reversed(Tensor._toposort(self)):
            val._backward()

