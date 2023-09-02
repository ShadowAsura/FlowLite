import array
import math

class Tensor:
    """
    A simple Tensor class for educational purposes.
    """

    def __init__(self, shape, dtype='float', data=None):
        """
        Initialize a new tensor.

        Args:
            shape (tuple): Shape of the tensor.
            dtype (str, optional): Data type of the tensor elements.
            data (list, optional): Flattened list of elements for the tensor.
        """
        self.shape = shape
        self.size = 1
        for dim in shape:
            self.size *= dim
        self.dtype = dtype
        if data is not None:
            self.data = array.array('d', data)  # Using 'd' for double-precision float
        else:
            self.data = array.array('d', [0] * self.size)


    def __del__(self):
        """
        Destructor to deallocate memory.
        """
        del self.data

    def get_element(self, indices):
        """
        Get an element from the tensor at the given indices.

        Args:
            indices (tuple): Indices of the element.

        Returns:
            Element at the given indices.
        """
        flat_index = 0
        multiplier = 1
        for i in range(len(self.shape) - 1, -1, -1):
            flat_index += indices[i] * multiplier
            multiplier *= self.shape[i]
        return self.data[flat_index]

    def reshape(self, new_shape):
        """
        Reshape the tensor.

        Args:
            new_shape (tuple): New shape for the tensor.

        Raises:
            ValueError: If the total size of the new shape doesn't match the original size.
        """
        new_size = 1
        for dim in new_shape:
            new_size *= dim

        if new_size != self.size:
            raise ValueError("New shape must have the same number of elements as the original shape")

        self.shape = new_shape

    def slice(self, start_indices, end_indices):
        """
        Create a sub-tensor by slicing the original tensor.

        Args:
            start_indices (tuple): Starting indices for each dimension.
            end_indices (tuple): Ending indices for each dimension.

        Returns:
            Tensor: A new tensor containing the sliced data.
        """
        pass  # Implement slicing logic here

    def validate_indices(self, indices):
        """
        Validate indices for element access.

        Args:
            indices (tuple): Indices to validate.

        Raises:
            ValueError: If indices are out of bounds or don't match the number of dimensions.
        """
        if len(indices) != len(self.shape):
            raise ValueError("Number of indices must match the number of dimensions")

        for i, index in enumerate(indices):
            if index >= self.shape[i]:
                raise ValueError("Index out of bounds")

    def add(self, other_tensor):
        """
        Element-wise addition of two tensors, with broadcasting support.

        Args:
            other_tensor (Tensor): Tensor to add.

        Returns:
            Tensor: New tensor resulting from the addition.
        """
        # Broadcasting logic
        if self.shape != other_tensor.shape:
            self = self._broadcast_to_match(other_tensor)
            other_tensor = other_tensor._broadcast_to_match(self)

        # Step 1: Check if shapes match
        if self.shape != other_tensor.shape:
            # Here, you would add your broadcasting logic
            raise ValueError("Shapes must match for addition")

        # Step 2: Perform addition
        new_data = array.array('d', [0] * self.size)
        for i in range(self.size):
            new_data[i] = self.data[i] + other_tensor.data[i]
        
        return Tensor(self.shape, self.dtype, data=new_data)

    def sub(self, other_tensor):
        """
        Element-wise subtraction of two tensors.

        Args:
            other_tensor (Tensor): Tensor to subtract.

        Returns:
            Tensor: New tensor resulting from the subtraction.
        """
        if self.shape != other_tensor.shape:
            # Add broadcasting logic here
            raise ValueError("Shapes must match for subtraction")

        new_data = array.array('d', [0] * self.size)
        for i in range(self.size):
            new_data[i] = self.data[i] - other_tensor.data[i]

        return Tensor(self.shape, self.dtype, data=new_data)

    def mul(self, other_tensor):
        """
        Element-wise multiplication of two tensors.

        Args:
            other_tensor (Tensor): Tensor to multiply.

        Returns:
            Tensor: New tensor resulting from the multiplication.
        """
        if self.shape != other_tensor.shape:
            # Add broadcasting logic here
            raise ValueError("Shapes must match for multiplication")

        new_data = array.array('d', [0] * self.size)
        for i in range(self.size):
            new_data[i] = self.data[i] * other_tensor.data[i]

        return Tensor(self.shape, self.dtype, data=new_data)

    def __repr__(self):
        """
        Create a string representation of the tensor for debugging.

        Returns:
            str: A string representation of the tensor.
        """
        return str(self.data)

    def save(self, filename):
        """
        Serialize the tensor to disk.

        Args:
            filename (str): File name for saving the tensor.
        """
        with open(filename, 'wb') as f:
            self.data.tofile(f)

    def load(self, filename):
        """
        Deserialize a tensor from disk and load it into memory.

        Args:
            filename (str): File name from which to load the tensor.
        """
        with open(filename, 'rb') as f:
            self.data.fromfile(f, self.size)

    def _broadcast_to_match(self, other_tensor):
        """
        Internal function to broadcast a tensor to match the shape of another.

        Args:
            other_tensor (Tensor): The tensor whose shape we want to match.

        Returns:
            Tensor: A new tensor broadcasted to the shape of other_tensor.
        """
        # Step 1: Make sure broadcasting is possible
        if len(self.shape) > len(other_tensor.shape):
            raise ValueError("Broadcasting is not possible")

        # Step 2: Determine the resulting shape
        new_shape = []
        for dim1, dim2 in zip(self.shape[::-1], other_tensor.shape[::-1]):
            if dim1 == dim2:
                new_shape.append(dim1)
            elif dim1 == 1:
                new_shape.append(dim2)
            else:
                raise ValueError("Broadcasting is not possible")
        new_shape = new_shape[::-1]

        # Step 3: Create new data array with broadcasted shape
        new_data = array.array('d', [0] * self.size * len(new_shape))

        # Add your broadcasting logic here. This involves populating the new_data array.

        return Tensor(new_shape, self.dtype, data=new_data)

    def __getitem__(self, indices):
        """
        Retrieve elements by indices.

        Args:
            indices (tuple): Indices of the elements to get.

        Returns:
            Tensor: New tensor containing the retrieved elements.
        """
        # Add advanced indexing logic here
        # For now, only slices without strides are implemented
        new_shape = []
        new_data = []
        for dim, slice_or_index in enumerate(indices):
            if isinstance(slice_or_index, slice):
                new_shape.append(slice_or_index.stop - slice_or_index.start)
                # Apply slice to data in this dimension
            elif isinstance(slice_or_index, int):
                # Apply single index to data in this dimension
            else:
                raise IndexError("Only slices and integers are valid indices.")

        return Tensor(new_shape, self.dtype, data=new_data)

    def add_(self, other_tensor):
        """
        In-place element-wise addition of two tensors.

        Args:
            other_tensor (Tensor): Tensor to add.

        Returns:
            None
        """
        if self.shape != other_tensor.shape:
            raise ValueError("Shapes must match for in-place addition")

        for i in range(self.size):
            self.data[i] += other_tensor.data[i]

    # Within the Tensor class

def concat(self, other_tensor, axis):
    """
    Concatenate tensor with another tensor along a specified axis.

    Args:
        other_tensor (Tensor): The tensor to concatenate.
        axis (int): The axis along which to concatenate.

    Returns:
        Tensor: A new tensor resulting from the concatenation.
    """
    # Step 1: Validate that concatenation is possible along the given axis
    if axis >= len(self.shape) or axis < -len(self.shape):
        raise ValueError("Invalid axis")

    for dim in range(len(self.shape)):
        if dim == axis:
            continue
        if self.shape[dim] != other_tensor.shape[dim]:
            raise ValueError("All dimensions except the axis must match for concatenation")

    # Step 2: Determine the shape of the resulting tensor
    new_shape = list(self.shape)
    new_shape[axis] += other_tensor.shape[axis]
    new_size = 1
    for dim in new_shape:
        new_size *= dim

    # Step 3: Create new data array
    new_data = array.array('d', [0] * new_size)

    # Step 4: Populate new data array
    index_self = 0  # To keep track of where to read from self.data
    index_other = 0  # To keep track of where to read from other_tensor.data
    index_new = 0  # To keep track of where to write to new_data

    # We'll assume 2D tensor for illustration, but this should be generalized for n-D
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            if axis == 0:
                if i < self.shape[0]:
                    new_data[index_new] = self.data[index_self]
                    index_self += 1
                else:
                    new_data[index_new] = other_tensor.data[index_other]
                    index_other += 1
            elif axis == 1:
                if j < self.shape[1]:
                    new_data[index_new] = self.data[index_self]
                    index_self += 1
                else:
                    new_data[index_new] = other_tensor.data[index_other]
                    index_other += 1
            index_new += 1

    return Tensor(new_shape, self.dtype, data=new_data)
