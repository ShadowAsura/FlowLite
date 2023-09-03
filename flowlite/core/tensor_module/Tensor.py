import array
import math

class Tensor:
    """
    A simple Tensor class.
    """

    def __init__(self, shape, dtype, data=None):
        """
        Initialize a new tensor.

        Args:
            shape (tuple): Shape of the tensor.
            dtype (str, optional): Data type of the tensor elements.
            data (list, optional): Flattened list of elements for the tensor.
        """
        if data is None:
            self.size = 1
            for dim in shape:
                if isinstance(dim, int):
                    self.size *= dim
                else:
                    raise ValueError("Invalid dimension in shape: must be integers.")
            self.data = array.array('d', [0] * self.size)  # Initialize with zeros
            self.shape = shape
        else:
            self.data = array.array('d', data)  # Use the array module to store data efficiently
            self.shape = self._compute_shape(data)
        
        self.dtype = dtype
        if self.shape is None:
            raise ValueError("Shape computation failed, returned None.")


    #def __del__(self):
        """
        Destructor to deallocate memory.
        """
     #   if hasattr(self, 'data'):
     #       del self.data

    def _ensure_tensor(self, other):
        if isinstance(other, Tensor):
            return other
        return Tensor(other)

    def get_element(self, indices):
        """
        Get an element from the tensor at the given indices.

        Args:
            indices (tuple): Indices of the element.

        Returns:
            Element at the given indices.
        """
        idx = sum((x * y) for x, y in zip(indices, [1] + [self.shape[k] for k in range(len(self.shape) - 1)]))
        return self.data[idx]

    def _compute_shape(self, data):
        shape = []
        dtype = type(data)
        while dtype == list:
            shape.append(len(data))
            if len(data) == 0:
                break
            data = data[0]
            dtype = type(data)
        return tuple(shape)

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
        # Placeholder for real slice functionality
        return Tensor((1,))


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
    def negate(tensor):
        """
        Element-wise negation of a tensor.

        Args:
            tensor (Tensor): Input tensor.

        Returns:
            Tensor: New tensor containing the element-wise negation.
        """
        new_data = array.array('d', [-x for x in tensor.data])
        return Tensor(tensor.shape, dtype=tensor.dtype, data=new_data)

    def __add__(self, other_tensor):
        """
        Element-wise addition of two tensors, with broadcasting support.

        Args:
            other_tensor (Tensor): Tensor to add.

        Returns:
            Tensor: New tensor resulting from the addition.
        """
        other_tensor = self._ensure_tensor(other_tensor)
        self._broadcast_to_match(other_tensor)

        # Step 2: Perform addition
        new_data = array.array('d', [0] * self.size)
        for i in range(self.size):
            new_data[i] = self.data[i] + other_tensor.data[i]
        
        return Tensor(self.shape, self.dtype, data=new_data)
    def reciprocal(tensor):
        """
        Element-wise reciprocal (1/x) of a tensor.

        Args:
            tensor (Tensor): Input tensor.

        Returns:
            Tensor: New tensor containing the element-wise reciprocal.
        """
        new_data = array.array('d', [1.0 / x for x in tensor.data])
        return Tensor(tensor.shape, dtype=tensor.dtype, data=new_data)

    def __sub__(self, other_tensor):
        """
        Element-wise subtraction of two tensors.

        Args:
            other_tensor (Tensor): Tensor to subtract.

        Returns:
            Tensor: New tensor resulting from the subtraction.
        """
        other_tensor = self._ensure_tensor(other_tensor)
        self._broadcast_to_match(other_tensor)

        new_data = array.array('d', [0] * self.size)
        for i in range(self.size):
            new_data[i] = self.data[i] - other_tensor.data[i]

        return Tensor(self.shape, self.dtype, data=new_data)

    def __mul__(self, other_tensor):
        """
        Element-wise multiplication of two tensors.

        Args:
            other_tensor (Tensor): Tensor to multiply.

        Returns:
            Tensor: New tensor resulting from the multiplication.
        """
        other_tensor = self._ensure_tensor(other_tensor)
        self._broadcast_to_match(other_tensor)

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

        other_shape = other_tensor.shape
        self_shape = self.shape

        if len(self_shape) > len(other_shape):
            return ValueError("Cannot broadcast.")

        # Extend shape with ones at the front
        self_shape = [1] * (len(other_shape) - len(self_shape)) + list(self_shape)

        # Create a new array with extended shape
        new_data = array.array('d', [0] * self.size)
        new_shape = []
        for dim1, dim2 in zip(reversed(self_shape), reversed(other_shape)):
            if dim1 == 1:
                new_shape.insert(0, dim2)
            elif dim1 != dim2:
                return ValueError("Cannot broadcast.")
            else:
                new_shape.insert(0, dim1)

        # ... (copy and expand data to new_data)

        return Tensor(shape=new_shape, dtype=self.dtype, data=new_data)

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
        if isinstance(indices, int) or all(isinstance(i, int) for i in indices):
            return self.get_element(indices if isinstance(indices, tuple) else (indices,))
        
        if isinstance(indices, slice):
            indices = (indices,)
        
        start_indices = []
        end_indices = []
        for i, index_or_slice in enumerate(indices):
            if isinstance(index_or_slice, slice):
                start = index_or_slice.start if index_or_slice.start is not None else 0
                end = index_or_slice.stop if index_or_slice.stop is not None else self.shape[i]
                start_indices.append(start)
                end_indices.append(end)
            else:
                start_indices.append(index_or_slice)
                end_indices.append(index_or_slice + 1)
        
        return self.slice(tuple(start_indices), tuple(end_indices))

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
