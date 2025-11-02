import numpy as np

class Matrix:
    def __init__(self, data, vector_type='row'):
        if not isinstance(data[0], list):
            if vector_type == 'row':
                data = [data]
            elif vector_type == 'col':
                data = [[x] for x in data]
            else:
                raise ValueError("vector_type должен быть 'row' или 'col'")
        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError("Все строки должны иметь одинаковую длину")

        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

    def __add__(self, other):
        return Matrix([[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def __sub__(self, other):
        return Matrix([[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def __mul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Несовместимые размеры матриц")
        result = []
        for i in range(self.rows):
            row = []
            for j in range(other.cols):
                val = sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                row.append(val)
            result.append(row)
        return Matrix(result)

    def transpose(self):
        return Matrix([[self.data[i][j] for i in range(self.rows)] for j in range(self.cols)])

    def norm_1(self):
        return max(sum(abs(self.data[i][j]) for i in range(self.rows)) for j in range(self.cols))

    def norm_infty(self):
        return max(sum(abs(self.data[i][j]) for j in range(self.cols)) for i in range(self.rows))

    def norm_2(self):
        mat = np.array(self.data)
        return np.sqrt(max(np.linalg.eigvals(mat.T @ mat)))

    def det(self):
        return np.linalg.det(np.array(self.data))

    def is_positive_definite(self):
        A = np.array(self.data)
        for i in range(1, self.rows + 1):
            if np.linalg.det(A[:i, :i]) <= 0:
                return False
        return True

    def print(self):
        for row in self.data:
            print(" ".join(f"{x:10.6f}" for x in row))
