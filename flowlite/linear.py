import numpy as np

class LinearLayer:
    """
    The LinearLayer class represents a fully connected layer in a neural network.
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initializes the layer with random weights and zero biases.

        Parameters:
            input_dim (int): The number of input features.
            output_dim (int): The number of output features.
        """
        self.weights = Tensor(np.random.randn(input_dim, output_dim))
        self.bias = Tensor(np.zeros(output_dim))
        
    def forward(self, input_tensor):
        """
        Performs the forward pass of the layer.

        Parameters:
            input_tensor (Tensor): The input data.

        Returns:
            Tensor: The output data.
        """
        self.input = input_tensor
        # Dot product of the input with the weights and then adding the bias
        return input_tensor.dot(self.weights) + self.bias