import numpy as np
from matrix_lib import Matrix

def cholesky(A: Matrix, b: Matrix):
    if not A.is_positive_definite():
        raise ValueError("Матрица не является положительно определённой")

    n = A.rows
    A = np.array(A.data)
    b = np.array([r[0] for r in b.data])

    L = np.zeros((n, n))
    counter = 0

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            counter += 2 * j
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - s)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
            counter += 2

    # Прямой ход
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
        counter += 2 * i + 1

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) / L[i][i]
        counter += 2 * (n - i) + 1

    det = np.prod(np.diag(L)) ** 2
    return Matrix(x.tolist(), "col"), det, counter

if __name__ == "__main__":
    n = 5
    A_data = [[1/(i+j-1) for j in range(1, n+1)] for i in range(1, n+1)]
    for i in range(n):
        A_data[i][i] += 1  # чтобы сделать A положительно определённой

    A = Matrix(A_data)
    x_true = Matrix([1]*n, "col")
    b = A * x_true

    x_ch, det, counter = cholesky(A, b)
    print("x =", [round(v[0], 5) for v in x_ch.data])
    print("det =", det)
    print("ops =", counter)
