from tensor import Tensor
import array

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
