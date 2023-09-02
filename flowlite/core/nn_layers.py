from tensor import Tensor
from matrix_ops import matmul
from activations import relu, sigmoid, tanh

class Dense:
    """
    Fully connected (Dense) layer.

    Args:
        input_dim (int): The dimensionality of the input.
        output_dim (int): The dimensionality of the output.
        activation (callable, optional): An optional activation function to apply after the linear transformation.
    """
    def __init__(self, input_dim, output_dim, activation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        # Map activation names to functions.
        self.activation_map = {'relu': relu, 'sigmoid': sigmoid, 'tanh': tanh}

        # Initialize weights and biases
        self.weights = Tensor((input_dim, output_dim), dtype='d')  # Initialize with small random values
        self.biases = Tensor((1, output_dim), dtype='d')  # Initialize with zeros

    def forward(self, x):
        """
        Forward pass through the layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # Perform the linear transformation
        z = matmul(x, self.weights)  # shape will be (batch_size, output_dim)

        # Add the bias term
        z = z + self.biases  # Broadcasting should take care of shape alignment

        # Optionally apply the activation function
        if self.activation:
            return self.activation_map[self.activation](z)

        return z
