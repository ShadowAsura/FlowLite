from tensor import Tensor
import array
import statistics

def sum(tensor, axis=None):
    """
    Compute the sum along the specified axis for a 2D tensor.

    Args:
        tensor (Tensor): Input tensor.
        axis (int, optional): The axis along which to sum. If None, sum all elements.

    Returns:
        Tensor: A new tensor containing the sum.
    """
    if axis is None:
        total_sum = sum(tensor.data)
        return Tensor([], dtype=tensor.dtype, data=[total_sum])

    rows, cols = tensor.shape
    
    if axis == 0:
        # Summing along columns
        new_shape = [cols]
        new_data = array.array('d', [0]*cols)
        for i in range(rows):
            for j in range(cols):
                new_data[j] += tensor.data[i*cols + j]
    elif axis == 1:
        # Summing along rows
        new_shape = [rows]
        new_data = array.array('d', [0]*rows)
        for i in range(rows):
            for j in range(cols):
                new_data[i] += tensor.data[i*cols + j]
    else:
        raise ValueError("Invalid axis")
    
    return Tensor(new_shape, tensor.dtype, data=new_data)

def mean(tensor, axis=None):
    """
    Compute the mean along the specified axis for a 2D tensor.

    Args:
        tensor (Tensor): Input tensor.
        axis (int, optional): The axis along which to compute the mean. If None, computes the global mean.

    Returns:
        Tensor: A new tensor containing the mean.
    """
    if axis is None:
        total_sum = sum(tensor.data)
        return Tensor([], dtype=tensor.dtype, data=[total_sum / tensor.size])

    rows, cols = tensor.shape
    new_data = array.array('d')
    
    if axis == 0:
        # Compute mean along columns
        for j in range(cols):
            col_sum = 0
            for i in range(rows):
                col_sum += tensor.data[i*cols + j]
            new_data.append(col_sum / rows)
        return Tensor([cols], dtype=tensor.dtype, data=new_data)

    elif axis == 1:
        # Compute mean along rows
        for i in range(rows):
            row_sum = 0
            for j in range(cols):
                row_sum += tensor.data[i*cols + j]
            new_data.append(row_sum / cols)
        return Tensor([rows], dtype=tensor.dtype, data=new_data)

def max(tensor, axis=None):
    """
    Compute the maximum value along the specified axis for a 2D tensor.

    Args:
        tensor (Tensor): Input tensor.
        axis (int, optional): The axis along which to compute the max. If None, computes the global max.

    Returns:
        Tensor: A new tensor containing the max value.
    """
    if axis is None:
        return Tensor([], dtype=tensor.dtype, data=[max(tensor.data)])

    rows, cols = tensor.shape
    new_data = array.array('d')
    
    if axis == 0:
        # Compute max along columns
        for j in range(cols):
            col_max = float('-inf')
            for i in range(rows):
                col_max = max(col_max, tensor.data[i*cols + j])
            new_data.append(col_max)
        return Tensor([cols], dtype=tensor.dtype, data=new_data)

    elif axis == 1:
        # Compute max along rows
        for i in range(rows):
            row_max = float('-inf')
            for j in range(cols):
                row_max = max(row_max, tensor.data[i*cols + j])
            new_data.append(row_max)
        return Tensor([rows], dtype=tensor.dtype, data=new_data)

def flatten_to_2d(shape, data):
    """
    Flattens the tensor data into a 2D list based on the specified axis.

    Args:
        shape (tuple): Shape of the tensor.
        data (array.array): Flat array containing tensor data.

    Returns:
        List of lists: 2D list containing tensor data.
    """
    rows = math.prod(shape[:-1])
    cols = shape[-1]
    two_d_data = [data[i * cols : (i + 1) * cols] for i in range(rows)]
    return two_d_data

def min(tensor, axis=None):
    """
    Compute the minimum value along the specified axis.

    Args:
        tensor (Tensor): Input tensor.
        axis (int, optional): Axis along which to find the minimum value. If None, finds the global minimum.

    Returns:
        Tensor: A new tensor containing the minimum value.
    """
    if axis is None:
        # Global minimum across all elements
        min_value = min(tensor.data)
        return Tensor([], dtype=tensor.dtype, data=array.array('d', [min_value]))

    elif axis == -1:
        # Minimum along the last axis
        two_d_data = flatten_to_2d(tensor.shape, tensor.data)
        min_values = [min(row) for row in two_d_data]
        return Tensor(tensor.shape[:-1], dtype=tensor.dtype, data=array.array('d', min_values))

    else:
        # More complex axis-specific logic could go here
        raise NotImplementedError("Arbitrary axis not yet implemented")

def std(tensor, axis=None):
    """
    Compute the standard deviation along the specified axis.

    Args:
        tensor (Tensor): Input tensor.
        axis (int, optional): Axis along which to find the standard deviation. If None, finds the global standard deviation.

    Returns:
        Tensor: A new tensor containing the standard deviation.
    """
    if axis is None:
        # Global standard deviation across all elements
        std_value = statistics.stdev(tensor.data)
        return Tensor([], dtype=tensor.dtype, data=array.array('d', [std_value]))

    elif axis == -1:
        # Standard deviation along the last axis
        two_d_data = flatten_to_2d(tensor.shape, tensor.data)
        std_values = [statistics.stdev(row) for row in two_d_data]
        return Tensor(tensor.shape[:-1], dtype=tensor.dtype, data=array.array('d', std_values))

    else:
        # More complex axis-specific logic could go here
        raise NotImplementedError("Arbitrary axis not yet implemented")


