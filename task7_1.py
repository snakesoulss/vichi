from task7_0 import Matrix
import math
import numpy as np


def analyze(A: Matrix):
    norms = [1, 2, 'inf']
    for norm in norms:
        cond = A.condition_number(norm if norm != 'inf' else float('inf'))
        print(f"Число обусловленности (норма {norm}): {cond:.6e}")
        if cond > 1e15:
            print("Матрица плохо обусловлена")
        else:
            print("Матрица хорошо обусловлена")


def gauss_solve(A: Matrix, b: Matrix, pivot_type='column'):
    n = A.rows

    mat = [row[:] for row in A.data]
    vec = [b.data[i][0] for i in range(n)]

    operations = 0
    det = 1.0
    sign = 1

    for k in range(n):
        if pivot_type == 'column':
            max_row = max(range(k, n), key=lambda i: abs(mat[i][k]))
            operations += (n - k)
            if max_row != k:
                mat[k], mat[max_row] = mat[max_row], mat[k]
                vec[k], vec[max_row] = vec[max_row], vec[k]
                sign *= -1

        elif pivot_type == 'row':
            max_col = max(range(k, n), key=lambda j: abs(mat[k][j]))
            operations += (n - k)
            if max_col != k:
                for i in range(n):
                    mat[i][k], mat[i][max_col] = mat[i][max_col], mat[i][k]

        elif pivot_type == 'full':
            max_i, max_j = k, k
            max_val = abs(mat[k][k])
            for i in range(k, n):
                for j in range(k, n):
                    if abs(mat[i][j]) > max_val:
                        max_val = abs(mat[i][j])
                        max_i, max_j = i, j
            operations += (n - k) ** 2

            if max_i != k:
                mat[k], mat[max_i] = mat[max_i], mat[k]
                vec[k], vec[max_i] = vec[max_i], vec[k]
                sign *= -1

            if max_j != k:
                for i in range(n):
                    mat[i][k], mat[i][max_j] = mat[i][max_j], mat[i][k]

        pivot = mat[k][k]
        if abs(pivot) < 1e-16:
            return None, 0.0, operations

        det *= pivot
        operations += 1

        for i in range(k + 1, n):
            factor = mat[i][k] / pivot
            operations += 1
            for j in range(k + 1, n):
                mat[i][j] -= factor * mat[k][j]
                operations += 2
            vec[i] -= factor * vec[k]
            operations += 2

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = vec[i]
        for j in range(i + 1, n):
            s -= mat[i][j] * x[j]
            operations += 2
        x[i] = s / mat[i][i]
        operations += 1

    return Matrix(x, "col"), det * sign, operations


def compare_operations(n, actual_ops):
    """Сравнение фактического количества операций с теоретическим"""
    theoretical_ops = (2 / 3) * n ** 3 + (3 / 2) * n ** 2 - (7 / 6) * n
    print(f"Теоретическое количество операций: {theoretical_ops:.0f}")
    print(f"Фактическое количество операций: {actual_ops}")
    print(f"Отношение (факт/теория): {actual_ops / theoretical_ops:.3f}")


def test_hilbert_matrix(n):
    """Тест с матрицей Гильберта"""
    A_data = [[1 / (i + j + 1) for j in range(n)] for i in range(n)]
    A = Matrix(A_data)

    x_exact = Matrix([1.0] * n, "col")
    b = A * x_exact

    print(f"\n{'=' * 60}")
    print(f"n = {n}")
    print(f"{'=' * 60}")

    analyze(A)

    strategies = ['none', 'column', 'row', 'full']

    for strategy in strategies:
        print(f"\n--- Стратегия: {strategy} ---")
        try:
            x_calc, det, ops = gauss_solve(A, b, pivot_type=strategy)

            if x_calc:
                error = max(abs(x_calc.data[i][0] - 1.0) for i in range(n))
                r = A * x_calc - b
                residual = max(abs(r.data[i][0]) for i in range(n))

                print(f"Определитель: {det:.3e}")
                print(f"Максимальная ошибка: {error:.3e}")
                print(f"Максимальная невязка: {residual:.3e}")
                print(f"Количество операций: {ops}")
                compare_operations(n, ops)
            else:
                print("Система вырождена")
        except Exception as e:
            print(f"Ошибка: {e}")


def test_random_matrix(n):
    """Тест со случайной матрицей"""
    print(f"\n{'=' * 60}")
    print(f"ТЕСТ СО СЛУЧАЙНОЙ МАТРИЦЕЙ (n={n})")
    print(f"{'=' * 60}")

    np.random.seed(42)
    A_data = np.random.uniform(-10, 10, (n, n)).tolist()
    A = Matrix(A_data)

    x_exact = Matrix([1.0] * n, "col")
    b = A * x_exact

    analyze(A)

    strategies = ['none', 'column', 'row', 'full']

    for strategy in strategies:
        print(f"\n--- Стратегия: {strategy} ---")
        try:
            x_calc, det, ops = gauss_solve(A, b, pivot_type=strategy)

            if x_calc:
                error = max(abs(x_calc.data[i][0] - 1.0) for i in range(n))
                r = A * x_calc - b
                residual = max(abs(r.data[i][0]) for i in range(n))

                print(f"Определитель: {det:.3e}")
                print(f"Максимальная ошибка: {error:.3e}")
                print(f"Максимальная невязка: {residual:.3e}")
                print(f"Количество операций: {ops}")
                compare_operations(n, ops)
            else:
                print("Система вырождена")
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    ns = [5, 8, 10, 12, 14]

    print("=" * 60)
    print("МЕТОД ГАУССА. МАТРИЦА ГИЛЬБЕРТА")
    print("=" * 60)

    for n in ns:
        print(f"\nТестирование для n = {n}")
        print("-" * 40)

        A_data = [[1 / (i + j + 1) for j in range(n)] for i in range(n)]
        A = Matrix(A_data)

        x_exact = Matrix([1.0] * n, "col")
        b = A * x_exact

        analyze(A)

        x_calc, det_gauss, ops = gauss_solve(A, b, pivot_type='column')

        if x_calc:
            error = max(abs(x_calc.data[i][0] - 1.0) for i in range(n))
            r = A * x_calc - b
            residual = max(abs(r.data[i][0]) for i in range(n))

            print(f"\nРезультаты:")
            print(f"Определитель (метод Гаусса): {det_gauss:.3e}")
            print(f"Максимальная ошибка: {error:.3e}")
            print(f"Максимальная невязка: {residual:.3e}")
            print(f"Количество операций: {ops}")

            compare_operations(n, ops)

            try:
                det_numpy = np.linalg.det(np.array(A_data))
                print(f"Определитель (numpy): {det_numpy:.3e}")
                print(f"Разница определителей: {abs(det_gauss - det_numpy):.3e}")
            except:
                pass
        else:
            print("Матрица вырождена!")

    # Дополнительные тесты
    print("\n" + "=" * 60)
    print("ДОПОЛНИТЕЛЬНЫЕ ТЕСТЫ")
    print("=" * 60)

    # Тест со случайной матрицей
    test_random_matrix(5)

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ СТРАТЕГИЙ ВЫБОРА ГЛАВНОГО ЭЛЕМЕНТА")
    print("=" * 60)
    test_hilbert_matrix(8)