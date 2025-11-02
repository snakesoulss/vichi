from matrix_lib import Matrix
import numpy as np

def thomas(A: Matrix, b: Matrix):
    n = A.rows
    a = [A.data[i][i-1] if i > 0 else 0 for i in range(n)]
    c = [A.data[i][i+1] if i < n-1 else 0 for i in range(n)]
    d = [b.data[i][0] for i in range(n)]
    b_main = [A.data[i][i] for i in range(n)]
    counter = 0

    for i in range(1, n):
        w = a[i] / b_main[i - 1]
        b_main[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
        counter += 5

    x = [0] * n
    x[-1] = d[-1] / b_main[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b_main[i]
        counter += 3

    return Matrix(x, "col"), counter

if __name__ == "__main__":
    n = 5
    A_data = [[0]*n for _ in range(n)]
    for i in range(n):
        A_data[i][i] = 4
        if i > 0:
            A_data[i][i-1] = 1
        if i < n - 1:
            A_data[i][i+1] = 1
    A = Matrix(A_data)
    x_true = Matrix([1]*n, "col")
    b = A * x_true
    x, counter = thomas(A, b)
    print("x =", [round(v[0], 5) for v in x.data])
    print("ops =", counter)
