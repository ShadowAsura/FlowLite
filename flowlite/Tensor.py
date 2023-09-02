import numpy as np

class Tensor:
    """
    The Tensor class represents a multi-dimensional array used for numerical computations.
    """
    
    def __init__(self, data):
        """
        Initializes a tensor object with the given data.

        Parameters:
            data (array-like): The initial data for the tensor.
        """
        # Store the data as a numpy array
        self.data = np.array(data)
        
        # Store the shape of the tensor for quick reference
        self.shape = self.data.shape

    def __repr__(self):
        """
        Represents the tensor object as a string for easy debugging.

        Returns:
            str: A string representation of the tensor.
        """
        return f"Tensor{self.shape} \n{self.data}"
    
    def mean(self):
        """
        Calculates the mean of the tensor.

        Returns:
            float: The mean value.
        """
        return np.mean(self.data)

    def reshape(self, new_shape):
        """
        Reshapes the tensor.

        Parameters:
            new_shape (tuple): The new shape for the tensor.
        """
        self.data = self.data.reshape(new_shape)
        self.shape = self.data.shape

    def transpose(self):
        """
        Transposes the tensor.
        """
        self.data = self.data.T
        self.shape = self.data.shape