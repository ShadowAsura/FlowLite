import unittest
from flowlite.core.tensor_module import Tensor

class TestTensor(unittest.TestCase):
    
    def test_initialization(self):
        a = Tensor([1, 2, 3], dtype='float')
        self.assertEqual(a.data.tolist(), [1.0, 2.0, 3.0])

    def test_addition(self):
        a = Tensor([1, 2, 3], dtype='float')
        b = Tensor([4, 5, 6], dtype='float')
        c = a + b
        print(f'Value of c.data: {c.data.tolist()}')
        self.assertEqual(c.data.tolist(), [5.0, 7.0, 9.0])

    def test_addition_diff_dtype(self):
        a = Tensor([1, 2, 3], dtype='float')
        b = Tensor([4, 5, 6], dtype='int')
        with self.assertRaises(ValueError):
            c = a + b

    def test_subtraction(self):
        a = Tensor([1, 2, 3], dtype='float')
        b = Tensor([4, 5, 6], dtype='float')
        c = a - b
        print(f'Value of c.data: {c.data.tolist()}')
        self.assertEqual(c.data.tolist(), [-3.0, -3.0, -3.0])

    def test_multiplication(self):
        a = Tensor([1, 2, 3], dtype='float')
        b = Tensor([4, 5, 6], dtype='float')
        c = a * b
        print(f'Value of c.data: {c.data.tolist()}')
        self.assertEqual(c.data.tolist(), [4.0, 10.0, 18.0])

if __name__ == '__main__':
    unittest.main()
