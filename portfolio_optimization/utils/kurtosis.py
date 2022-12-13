import numpy as np
from numba import jit


@jit(nopython=True)
def elimination_matrix(n: int) -> np.ndarray:
    rows = int((n * (n + 1)) / 2)
    cols = n * n
    out = np.zeros((rows, cols), np.int64)
    for j in range(n):
        e_j = np.zeros((1, n), np.int64)
        e_j[0, j] = 1
        for i in range(j, n):
            u = np.zeros((rows, 1), np.int64)
            row = int(j * n + i - ((j + 1) * j) / 2)
            u[row, 0] = 1
            e_i = np.zeros((1, n), np.int64)
            e_i[0, i] = 1
            out += np.kron(u, np.kron(e_j, e_i))
    return out

