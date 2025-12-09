from matrix_lib import Matrix
from task7_1 import gauss
from task7_3 import thomas
import numpy as np
from time import perf_counter

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


def build_tridiagonal(n, diag=4, off_diag=1):
    data = [[0] * n for _ in range(n)]
    for i in range(n):
        data[i][i] = diag
        if i > 0:
            data[i][i - 1] = off_diag
        if i < n - 1:
            data[i][i + 1] = off_diag
    return Matrix(data)


if __name__ == "__main__":
    sizes = [50, 100, 200]
    print("Сравнение скорости на трёхдиагональных СЛАУ:")
    for n in sizes:
        A = build_tridiagonal(n)
        x_true = Matrix([1] * n, "col")
        b = A * x_true

        start = perf_counter()
        x_t, ops_t = thomas(A, b)
        t_t = perf_counter() - start

        start = perf_counter()
        x_g, _, ops_g = gauss(A, b)
        t_g = perf_counter() - start

        err_t = max(abs(x_t.data[i][0] - 1) for i in range(n))
        err_g = max(abs(x_g.data[i][0] - 1) for i in range(n))

        print(f"n = {n}")
        print(f"  Thomas: {t_t*1e3:7.2f} ms, ops = {ops_t:7d}, max error = {err_t:.2e}")
        print(f"  Gauss : {t_g*1e3:7.2f} ms, ops = {ops_g:7d}, max error = {err_g:.2e}")
