from tensor import Tensor
from math_ops import exp

def relu(tensor):
    """
    ReLU (Rectified Linear Unit) activation function.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor where negative values are replaced by zero.
    """
    return tensor.maximum(0)

def sigmoid(tensor):
    """
    Sigmoid activation function.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor where values are squashed to the range (0, 1).
    """
    return 1 / (1 + exp(-tensor))

def tanh(tensor):
    """
    Hyperbolic Tangent (tanh) activation function.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor where values are squashed to the range (-1, 1).
    """
    pos = exp(tensor)
    neg = exp(-tensor)
    return (pos - neg) / (pos + neg)
