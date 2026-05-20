import numpy as np
from numpy.typing import NDArray

# PRE-COMPUTED BAYER MATRICES
bayer_matrix_2x2 = np.array([
    [0, 2],
    [3, 1]
]) / 4.0

bayer_matrix_4x4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5]
]) / 16.0

bayer_matrix_8x8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21]
]) / 64.0

matrices = {
    '2x2': bayer_matrix_2x2,
    '4x4': bayer_matrix_4x4,
    '8x8': bayer_matrix_8x8
}

def generate_bayer_matrix(order: int) -> NDArray[np.float32]:
    """Generates a Bayer matrix of size (2^order x 2^order).
    
    Args:
        order (int): The order of the matrix (e.g., 1 for 2x2, 2 for 4x4, 3 for 8x8).
    
    Returns:
        NDArray[np.float32]: The normalized Bayer matrix.
    """
    
    matrix = np.array([[0, 2], 
                       [3, 1]], dtype=np.float32)
    
    for _ in range(order - 1):
        # M_2n = [[4*Mn, 4*Mn + 2], [4*Mn + 3, 4*Mn + 1]]
        n = matrix.shape[0]
        top = np.hstack([4 * matrix, 4 * matrix + 2])
        bot = np.hstack([4 * matrix + 3, 4 * matrix + 1])
        matrix = np.vstack([top, bot])
        
    return matrix / (matrix.max() + 1)

# Example:
# bayer_4x4 = generate_bayer_matrix(2)