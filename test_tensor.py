from flowlite.core.tensor import Tensor
import unittest


class TestTensor(unittest.TestCase):
    
    def test_addition(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a + b
        self.assertEqual(c.data, [5, 7, 9])
        
    def test_subtraction(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a - b
        self.assertEqual(c.data, [-3, -3, -3])
        
    def test_multiplication(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a * b
        self.assertEqual(c.data, [4, 10, 18])

    def test_shape(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(a.shape, (2, 3))


if __name__ == '__main__':
    unittest.main()
