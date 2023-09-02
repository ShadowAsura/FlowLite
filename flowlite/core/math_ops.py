from tensor import Tensor
import array
import math

def exp(tensor):
    """
    Compute the exponential of each element in the tensor.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: New tensor where each element is e^x, where x is the corresponding element in the input tensor.
    """
    new_data = array.array('d', [math.exp(x) for x in tensor.data])
    return Tensor(tensor.shape, tensor.dtype, data=new_data)

def log(tensor):
    """
    Compute the natural logarithm of each element in the tensor.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: New tensor where each element is ln(x), where x is the corresponding element in the input tensor.
    """
    new_data = array.array('d', [math.log(x) for x in tensor.data])
    return Tensor(tensor.shape, tensor.dtype, data=new_data)

def sqrt(tensor):
    """
    Compute the square root of each element in the tensor.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: New tensor where each element is sqrt(x), where x is the corresponding element in the input tensor.
    """
    new_data = array.array('d', [math.sqrt(x) for x in tensor.data])
    return Tensor(tensor.shape, tensor.dtype, data=new_data)
