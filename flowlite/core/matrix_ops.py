from tensor import Tensor
import array

def matmul(tensor1, tensor2):
    """
    Matrix multiplication between two tensors.

    Args:
        tensor1, tensor2 (Tensor): Input tensors.

    Returns:
        Tensor: Result of matrix multiplication.
    """
    rows = tensor1.shape[0]
    shared_dim = tensor1.shape[1]
    cols = tensor2.shape[1]

    if shared_dim != tensor2.shape[0]:
        raise ValueError("Incompatible shapes for matrix multiplication.")

    new_data = array.array('d', [0.0] * (rows * cols))
    for i in range(rows):
        for j in range(cols):
            for k in range(shared_dim):
                new_data[i * cols + j] += tensor1.data[i * shared_dim + k] * tensor2.data[k * cols + j]

    new_shape = (rows, cols)
    return Tensor(new_shape, dtype=tensor1.dtype, data=new_data)

def transpose(tensor):
    """
    Transpose a matrix.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Transposed tensor.
    """
    rows = tensor.shape[0]
    cols = tensor.shape[1]
    new_data = array.array('d', [0.0] * (rows * cols))
    
    for i in range(rows):
        for j in range(cols):
            new_data[j * rows + i] = tensor.data[i * cols + j]

    new_shape = (cols, rows)
    return Tensor(new_shape, dtype=tensor.dtype, data=new_data)

def inverse(tensor):
    """
    Compute the inverse of a square matrix using Gauss-Jordan elimination.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Inverse of the input tensor.
    """
    n = tensor.shape[0]
    if n != tensor.shape[1]:
        raise ValueError("Inverse only defined for square matrices.")

    # Create an augmented matrix [A|I]
    augmented_data = array.array('d', [0.0] * (n * n * 2))
    for i in range(n):
        for j in range(n):
            augmented_data[i * (n * 2) + j] = tensor.data[i * n + j]  # Copy A
            augmented_data[i * (n * 2) + (n + j)] = 1.0 if i == j else 0.0  # Initialize I

    # Gauss-Jordan elimination
    for i in range(n):
        # Find the pivot row and swap
        max_row = max(range(i, n), key=lambda r: abs(augmented_data[r * (n * 2) + i]))
        augmented_data[i * (n * 2): (i + 1) * (n * 2)], augmented_data[max_row * (n * 2): (max_row + 1) * (n * 2)] = \
        augmented_data[max_row * (n * 2): (max_row + 1) * (n * 2)], augmented_data[i * (n * 2): (i + 1) * (n * 2)]

        # Normalize pivot row
        pivot = augmented_data[i * (n * 2) + i]
        for j in range(i, n * 2):
            augmented_data[i * (n * 2) + j] /= pivot

        # Eliminate other rows
        for r in range(n):
            if r == i:
                continue
            factor = augmented_data[r * (n * 2) + i]
            for j in range(i, n * 2):
                augmented_data[r * (n * 2) + j] -= factor * augmented_data[i * (n * 2) + j]

    # Extract the inverse matrix from [A|I] -> [I|A^-1]
    new_data = array.array('d', [augmented_data[i * (n * 2) + (n + j)] for i in range(n) for j in range(n)])
    return Tensor(tensor.shape, dtype=tensor.dtype, data=new_data)

