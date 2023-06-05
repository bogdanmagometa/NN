import unittest

from nn.value import Value

class ValueTestCase(unittest.TestCase):
    def test_add(self):
        # Arrange: Set up the necessary preconditions for the test
        val1 = Value(-1.0)
        val2 = Value(3.3)
        val3 = Value(1.2)
        val4 = Value(1.2)

        # Act: Execute the function or code to be tested
        s1 = val1 + val2
        s2 = val3 + 2
        s3 = 5 + val4

        Value.back_prop(s1)
        Value.back_prop(s2)
        Value.back_prop(s3)

        # Assert: Verify the expected results
        self.assertAlmostEqual(s1.data, 2.3)
        self.assertAlmostEqual(s1.grad, 1)
        self.assertAlmostEqual(val1.grad, 1)
        self.assertAlmostEqual(val2.grad, 1)

        # Example:
        # self.assertEqual(actual_result, expected_result)
        # self.assertTrue(condition)
        # self.assertFalse(condition)
        # self.assertAlmostEqual(first_value, second_value)
        # self.assertIn(item, list)
        # ...
        
        # Cleanup (if needed): Perform any necessary cleanup after the test

if __name__ == '__main__':
    unittest.main()