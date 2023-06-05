import unittest

import numpy as np

from nn.tensor import Tensor


class TensorTestCase(unittest.TestCase):
    def test_add(self):
        # Arrange: Set up the necessary preconditions for the test
        t1 = Tensor(np.array([1, 2, 3], dtype=float))
        t2 = Tensor(np.array([4, 5, 6], dtype=float))

        # Act: Execute the function or code to be tested
        s1 = t1 + t2

        s1.back_prop()

        # Assert: Verify the expected results
        self.assertTrue(np.allclose(s1.data, [5, 7, 9]))
        self.assertTrue(np.allclose(s1.grad, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.assertTrue(np.allclose(t1.grad, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.assertTrue(np.allclose(t2.grad, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

        # Example:
        # self.assertEqual(actual_result, expected_result)
        # self.assertTrue(condition)
        # self.assertFalse(condition)
        # self.assertAlmostEqual(first_value, second_value)
        # self.assertIn(item, list)
        # ...

        # Cleanup (if needed): Perform any necessary cleanup after the test

    def test_sum(self):
        t1 = Tensor(np.array([1, 2, 3], dtype=float))
        t2 = Tensor(np.array([4, 5, 6], dtype=float))

        s1 = t1 + t2
        s2 = s1.sum()

        s2.back_prop()

        self.assertTrue(np.allclose(s2.grad, 1))

    def test_single_value(self):
        t1 = Tensor(np.array(1, dtype=float))
        t2 = Tensor(np.array(4, dtype=float))

        s1 = t1 + t2

        s2 = s1 * 4.0

        s2.back_prop()

        self.assertAlmostEqual(s2.data, 20)
        self.assertAlmostEqual(s2.grad, 1)
        self.assertAlmostEqual(s1.data, 5)
        self.assertAlmostEqual(s1.grad, 4)
        self.assertAlmostEqual(t1.data, 1)
        self.assertAlmostEqual(t1.grad, 4)
        self.assertAlmostEqual(t2.data, 4)
        self.assertAlmostEqual(t2.grad, 4)

    def test_mul(self):
        t1 = Tensor(np.array([[1, 2], 
                              [3, 4]], dtype=float))
        t2 = Tensor(np.array([[1, 1], 
                              [2, 2]], dtype=float))

        s1 = t1 * t2
        s2 = s1.sum()

        s2.back_prop()

        self.assertAlmostEqual(s2.data, 17)
        self.assertAlmostEqual(s2.grad, 1)
        self.assertTrue(np.allclose(s1.grad, np.array([[1, 1], [1, 1]])))

    def test_reshape(self):
        t1 = Tensor(np.array([[1, 2], 
                              [3, 4]], dtype=float))
        t2 = t1.reshape((-1,))

        t3 = t2 * np.array([0, -1, 2, 3], dtype=np.float64)

        s1 = t3.sum()

        s1.back_prop()

        self.assertAlmostEqual(s1.data, 16)
        self.assertAlmostEqual(s1.grad, 1)
        self.assertTrue(np.allclose(t3.data.shape, (4,)))
        self.assertTrue(np.allclose(t3.data, [0, -2, 6, 12]))
        self.assertTrue(np.allclose(t3.grad, [1, 1, 1, 1]))
        self.assertTrue(np.allclose(t2.data.shape, (4,)))
        self.assertTrue(np.allclose(t2.data, [1, 2, 3, 4]))
        self.assertTrue(np.allclose(t2.grad.shape, (4,)))
        self.assertTrue(np.allclose(t2.grad, [0, -1, 2, 3]))
        self.assertTrue(np.allclose(t1.data.shape, (2, 2)))
        self.assertTrue(np.allclose(t1.data, [[1, 2], [3, 4]]))
        self.assertTrue(np.allclose(t1.grad.shape, (2, 2)))
        self.assertTrue(np.allclose(t1.grad, [[0, -1], [2, 3]]))

    def test_broadcast_to(self):
        t1 = Tensor(np.array([[1, 2]], dtype=float))
        t2 = t1.broadcast_to((2, 2))

        s1 = t2.sum()

        s1.back_prop()

        self.assertAlmostEqual(s1.data, 6)
        self.assertAlmostEqual(s1.grad, 1)
        self.assertTrue(np.allclose(t2.data.shape, (2, 2)))
        self.assertTrue(np.allclose(t2.data, [[1, 2], [1, 2]]))
        self.assertTrue(np.allclose(t2.grad, [[1, 1], [1, 1]]))
        self.assertTrue(np.allclose(t1.data.shape, (1, 2)))
        self.assertTrue(np.allclose(t1.data, [[1, 2]]))
        self.assertTrue(np.allclose(t1.grad.shape, (1, 2)))
        self.assertTrue(np.allclose(t1.grad, [[2, 2]]))


    def test_sigmoid(self):
        t1 = Tensor(np.array([0.0, 10**6], dtype=np.float64))
        t2 = t1.sigmoid()

        t2.back_prop()

        self.assertTrue(np.allclose(t2.data, [0.5, 1]))
        self.assertTrue(np.allclose(t2.grad, [[1, 0], [0, 1]]))
        self.assertTrue(np.allclose(t1.data, [0, 10**6]))
        self.assertTrue(np.allclose(t1.grad, [[0.25, 0], [0, 0]]))

    def test_exponent(self):
        t1 = Tensor(np.array([0.0, 3], dtype=np.float64))
        t2 = t1 ** 2

        s1 = t2.sum()

        s1.back_prop()

        self.assertTrue(np.allclose(s1.data, 9))
        self.assertTrue(np.isclose(s1.grad, 1))
        self.assertTrue(np.allclose(t2.data, [0, 9]))
        self.assertTrue(np.allclose(t2.grad, [1, 1]))
        self.assertTrue(np.allclose(t1.data, [0, 3]))
        self.assertTrue(np.allclose(t1.grad, [0, 6]))

    def test_exponent2(self):
        t1 = Tensor(np.array([0.0, 3], dtype=np.float64))
        t2 = Tensor(np.array([1, -1], dtype=np.float64))

        t3 = t1 ** t2

        s1 = t3.sum()

        s1.back_prop()

        self.assertTrue(np.allclose(s1.data, 1/3))
        self.assertTrue(np.isclose(s1.grad, 1))
        self.assertTrue(np.allclose(t3.data, [0, 1/3]))
        self.assertTrue(np.allclose(t3.grad, [1, 1]))
        self.assertTrue(np.allclose(t1.data, [0, 3]))
        self.assertTrue(np.allclose(t1.grad, [1, - (3**(-2))]))
        self.assertTrue(np.allclose(t2.data, [1, -1]))
        self.assertTrue(np.isnan(t2.grad[0]))
        self.assertTrue(t2.grad[1] == 1/3 * np.log(3))

    def test_exponent3(self):        
        t1 = Tensor(np.array([0.0, 3], dtype=np.float64))
        t2 = np.e ** t1

        s1 = t2.sum()

        s1.back_prop()

        self.assertTrue(np.allclose(s1.data, 1 + np.e**3))
        self.assertTrue(np.isclose(s1.grad, 1))
        self.assertTrue(np.allclose(t2.data, [1, np.e**3]))
        self.assertTrue(np.allclose(t2.grad, [1, 1]))
        self.assertTrue(np.allclose(t1.data, [0, 3]))
        self.assertTrue(np.allclose(t1.grad, [1, np.e**3]))

    def test_log(self):
        pass

    def test_relu(self):
        pass

    def test_tensordot(self):
        W = Tensor(np.array([[1, 2], 
                             [3, 4]], dtype=np.float64))
        h = Tensor(np.array([[5], [6]], dtype=np.float64))

        prod = W.tensordot(h, 1)

        prod.back_prop()

        self.assertTrue(np.allclose(prod.data, [[17], [39]]))
        self.assertTrue(np.allclose(prod.grad.shape, (2, 1, 2, 1)))
        self.assertTrue(np.allclose(prod.grad, [[[[1], [0]]], [[[0], [1]]]]))
        self.assertTrue(np.allclose(h.grad.shape, (2, 1, 2, 1)))
        self.assertTrue(np.allclose(h.grad, [[[[1], [2]]], 
                                             [[[3], [4]]]]))

    def test_tensordot(self):
        W = Tensor(np.array([[1, 2], 
                             [3, 4]], dtype=np.float64))
        h = Tensor(np.array([5, 6], dtype=np.float64))

        prod = W.tensordot(h, 1)

        prod.back_prop()

        self.assertTrue(np.allclose(prod.data, [17, 39]))
        self.assertTrue(np.allclose(np.shape(prod.grad), (2, 2)))
        self.assertTrue(np.allclose(prod.grad, np.eye(2)))
        self.assertTrue(np.allclose(h.grad.shape, (2, 2)))
        self.assertTrue(np.allclose(h.grad, W.data))


if __name__ == '__main__':
    unittest.main()
