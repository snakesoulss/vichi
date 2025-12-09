import numpy as np
from matrix_lib import Matrix

def gauss(A: Matrix, b: Matrix, pivot=True):
    n = A.rows
    A = [row[:] for row in A.data]
    b = [row[0] for row in b.data]
    counter = 0
    det = 1

    for i in range(n):
        if pivot:
            max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]

        for k in range(i + 1, n):
            factor = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
                counter += 2
            b[k] -= factor * b[i]
            counter += 2
        det *= A[i][i]

    x = [0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - s) / A[i][i]
        counter += 2 * (n - i)

    return Matrix(x, "col"), det, counter

if __name__ == "__main__":
    sizes = [5, 8, 10, 12]
    for n in sizes:
        A_data = [[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)]
        A = Matrix(A_data)
        x_true = Matrix([1] * n, "col")
        b = A * x_true

        cond = A.condition_number()
        x_gauss, det, counter = gauss(A, b)
        error = max(abs(x_gauss.data[i][0] - 1) for i in range(n))

        print(f"n = {n}")
        print(f"  cond(A)        = {cond:.3e}")
        print(f"  det            = {det:.3e}")
        print(f"  max error      = {error:.3e}")
        print(f"  operations     = {counter}")
