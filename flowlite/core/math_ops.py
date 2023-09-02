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



def div(tensor1, tensor2):
    """
    Element-wise division of two tensors.
    Assumes tensors have the same shape.

    Args:
        tensor1, tensor2 (Tensor): Input tensors.

    Returns:
        Tensor: New tensor with element-wise division result.
    """
    new_data = array.array('d', [a / b for a, b in zip(tensor1.data, tensor2.data)])
    return Tensor(tensor1.shape, dtype=tensor1.dtype, data=new_data)

def pow(tensor, exponent):
    """
    Raise each element in the tensor to a specific power.

    Args:
        tensor (Tensor): Input tensor.
        exponent (float): The exponent to raise each element to.

    Returns:
        Tensor: New tensor with each element raised to the given power.
    """
    new_data = array.array('d', [math.pow(x, exponent) for x in tensor.data])
    return Tensor(tensor.shape, dtype=tensor.dtype, data=new_data)
