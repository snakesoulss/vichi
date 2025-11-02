from matrix_lib import Matrix
import numpy as np

def lup(A: Matrix, b: Matrix):
    n = A.rows
    A = [row[:] for row in A.data]
    b = [row[0] for row in b.data]
    counter = 0

    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[max_row] = A[max_row], A[i]
        b[i], b[max_row] = b[max_row], b[i]

        for j in range(i + 1, n):
            A[j][i] /= A[i][i]
            for k in range(i + 1, n):
                A[j][k] -= A[j][i] * A[i][k]
                counter += 2

    y = [0] * n
    for i in range(n):
        y[i] = b[i] - sum(A[i][j] * y[j] for j in range(i))
        counter += 2 * i

    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
        counter += 2 * (n - i)

    return Matrix(x, "col"), counter
