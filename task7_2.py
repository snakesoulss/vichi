from task7_0 import Matrix
from task7_1 import gauss_solve
from task7_3 import thomas
import numpy as np
from time import perf_counter
import random


class LUSolver:
    def __init__(self):
        self.L = None
        self.U = None
        self.P = None

    def lu_decomposition(self, A: Matrix):
        n = A.rows
        L_data = [[0.0] * n for _ in range(n)]
        U_data = [row[:] for row in A.data]
        counter = 0

        for k in range(n):
            L_data[k][k] = 1.0

            for i in range(k + 1, n):
                if abs(U_data[k][k]) < 1e-12:
                    raise ValueError(f"Нулевой диагональный элемент a[{k}][{k}], требуется выбор главного элемента")

                L_data[i][k] = U_data[i][k] / U_data[k][k]
                counter += 1

                for j in range(k, n):
                    U_data[i][j] -= L_data[i][k] * U_data[k][j]
                    counter += 2

        self.L = Matrix(L_data)
        self.U = Matrix(U_data)
        return counter

    def lup_decomposition(self, A: Matrix):
        n = A.rows
        P_data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        A_data = [row[:] for row in A.data]
        counter = 0

        for k in range(n):
            max_row = k
            max_val = abs(A_data[k][k])

            for i in range(k + 1, n):
                if abs(A_data[i][k]) > max_val:
                    max_val = abs(A_data[i][k])
                    max_row = i

            if max_row != k:
                A_data[k], A_data[max_row] = A_data[max_row], A_data[k]
                P_data[k], P_data[max_row] = P_data[max_row], P_data[k]

            if abs(A_data[k][k]) < 1e-12:
                raise ValueError("Матрица вырождена")

            for i in range(k + 1, n):
                A_data[i][k] /= A_data[k][k]
                counter += 1
                for j in range(k + 1, n):
                    A_data[i][j] -= A_data[i][k] * A_data[k][j]
                    counter += 2

        L_data = [[0.0] * n for _ in range(n)]
        U_data = [[0.0] * n for _ in range(n)]

        for i in range(n):
            L_data[i][i] = 1.0
            for j in range(i):
                L_data[i][j] = A_data[i][j]

            for j in range(i, n):
                U_data[i][j] = A_data[i][j]

        self.L = Matrix(L_data)
        self.U = Matrix(U_data)
        self.P = Matrix(P_data)
        return counter

    def solve(self, b: Matrix):
        if self.L is None or self.U is None:
            raise ValueError("Сначала выполните разложение!")

        n = self.L.rows

        if self.P is not None:
            b = self.P * b

        y_data = [0.0] * n
        b_data = [row[0] for row in b.data]

        for i in range(n):
            sum_ly = 0.0
            for j in range(i):
                sum_ly += self.L.data[i][j] * y_data[j]
            y_data[i] = b_data[i] - sum_ly

        x_data = [0.0] * n
        for i in range(n - 1, -1, -1):
            sum_ux = 0.0
            for j in range(i + 1, n):
                sum_ux += self.U.data[i][j] * x_data[j]
            x_data[i] = (y_data[i] - sum_ux) / self.U.data[i][i]

        return Matrix([[x] for x in x_data])

    def determinant(self):
        if self.U is None:
            raise ValueError("Сначала выполните разложение!")

        n = self.U.rows
        det = 1.0
        for i in range(n):
            det *= self.U.data[i][i]

        if self.P is not None:
            perm = []
            for i in range(n):
                for j in range(n):
                    if self.P.data[i][j] == 1:
                        perm.append(j)
                        break

            inversions = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if perm[i] > perm[j]:
                        inversions += 1

            P_det = 1.0 if inversions % 2 == 0 else -1.0
            det *= P_det

        return det


def generate_test_matrix(n):
    data = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        data[i][i] = sum(abs(data[i][j]) for j in range(n) if j != i) + random.uniform(1, 5)
    return Matrix(data)


def generate_random_vector(n, a=-10, b=10):
    data = [[random.uniform(a, b)] for _ in range(n)]
    return Matrix(data)


def test_lu_vs_lup():
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ LU/LUP РАЗЛОЖЕНИЙ")
    print("=" * 60)

    n = 5
    A = generate_test_matrix(n)
    x_true = generate_random_vector(n)
    b = A * x_true

    print("\nМатрица A:")
    A.print()
    print(f"\nПравая часть b:")
    for i in range(n):
        print(f"  b[{i}] = {b.data[i][0]:.6f}")

    print("\n" + "-" * 40)
    print("1. LU-РАЗЛОЖЕНИЕ (без выбора главного элемента)")
    print("-" * 40)

    try:
        solver_lu = LUSolver()
        ops_lu = solver_lu.lu_decomposition(A)

        print("\nМатрица L:")
        solver_lu.L.print()
        print("\nМатрица U:")
        solver_lu.U.print()

        A_reconstructed = solver_lu.L * solver_lu.U
        try:
            error_norm = (A - A_reconstructed).norm_1()
        except:
            error_norm = (A - A_reconstructed).normL1()
        print(f"\nПроверка A = L*U:")
        print(f"  ||A - L*U|| = {error_norm:.2e}")

        x_lu = solver_lu.solve(b)
        try:
            error_lu = (x_true - x_lu).norm_1()
        except:
            error_lu = (x_true - x_lu).normL1()
        print(f"\nРешение системы:")
        print(f"  Операций разложения: {ops_lu}")
        print(f"  Ошибка: {error_lu:.2e}")
        print(f"  Определитель: {solver_lu.determinant():.6f}")

    except ValueError as e:
        print(f"  Ошибка: {e}")

    print("\n" + "-" * 40)
    print("2. LUP-РАЗЛОЖЕНИЕ (с выбором главного элемента)")
    print("-" * 40)

    solver_lup = LUSolver()
    ops_lup = solver_lup.lup_decomposition(A)

    print("\nМатрица P:")
    solver_lup.P.print()
    print("\nМатрица L:")
    solver_lup.L.print()
    print("\nМатрица U:")
    solver_lup.U.print()

    PA = solver_lup.P * A
    LU = solver_lup.L * solver_lup.U
    try:
        error_norm = (PA - LU).norm_1()
    except:
        error_norm = (PA - LU).normL1()
    print("\nПроверка P*A = L*U:")
    print(f"  ||P*A - L*U|| = {error_norm:.2e}")

    x_lup = solver_lup.solve(b)
    try:
        error_lup = (x_true - x_lup).norm_1()
    except:
        error_lup = (x_true - x_lup).normL1()
    print(f"\nРешение системы:")
    print(f"  Операций разложения: {ops_lup}")
    print(f"  Ошибка: {error_lup:.2e}")
    print(f"  Определитель: {solver_lup.determinant():.6f}")

    print("\n" + "-" * 40)
    print("3. СРАВНЕНИЕ С ПРЯМЫМ РЕШЕНИЕМ")
    print("-" * 40)

    start = perf_counter()
    x_gauss, _, ops_gauss = gauss_solve(A, b)
    time_gauss = perf_counter() - start

    try:
        error_gauss = (x_true - x_gauss).norm_1()
    except:
        error_gauss = (x_true - x_gauss).normL1()

    print(f"Метод Гаусса:")
    print(f"  Операций: {ops_gauss}")
    print(f"  Время: {time_gauss * 1e3:.2f} мс")
    print(f"  Ошибка: {error_gauss:.2e}")


def test_multiple_rhs():
    print("\n" + "=" * 60)
    print("ТЕСТ: РЕШЕНИЕ С НЕСКОЛЬКИМИ ПРАВЫМИ ЧАСТЯМИ")
    print("=" * 60)

    n = 4
    A = generate_test_matrix(n)

    print("\nМатрица A:")
    A.print()

    solver = LUSolver()
    ops_decomp = solver.lup_decomposition(A)
    print(f"\nВыполнено LUP-разложение ({ops_decomp} операций)")

    num_tests = 3
    total_ops_solve = 0

    print(f"\nРешаем {num_tests} систем с разными правыми частями:")

    for i in range(num_tests):
        x_true = generate_random_vector(n)
        b = A * x_true

        x_solved = solver.solve(b)

        try:
            error = (x_true - x_solved).norm_1()
        except:
            error = (x_true - x_solved).normL1()

        print(f"\nТест {i + 1}:")
        print(f"  Истинное решение: {[x_true.data[j][0] for j in range(n)]}")
        print(f"  Найденное решение: {[x_solved.data[j][0] for j in range(n)]}")
        print(f"  Ошибка: {error:.2e}")

    print("\n" + "-" * 40)
    print("СРАВНЕНИЕ ЧИСЛА ОПЕРАЦИЙ:")
    print("-" * 40)

    ops_gauss_per_system = 2 / 3 * n ** 3 + 3 / 2 * n ** 2 - 7 / 6 * n
    ops_gauss_total = ops_gauss_per_system * num_tests

    ops_lup_solve_per_system = 2 * n ** 2
    ops_lup_total = ops_decomp + ops_lup_solve_per_system * num_tests

    print(f"Метод Гаусса (каждую систему отдельно):")
    print(f"  {ops_gauss_total:.0f} операций ({num_tests} × {ops_gauss_per_system:.0f})")

    print(f"LUP-разложение + решения:")
    print(f"  {ops_lup_total:.0f} операций")
    print(f"    Разложение: {ops_decomp:.0f}")
    print(f"    Решения: {num_tests} × {ops_lup_solve_per_system:.0f}")

    print(f"\nЭкономия: {ops_gauss_total - ops_lup_total:.0f} операций")
    print(f"  (в {ops_gauss_total / ops_lup_total:.2f} раза меньше)")


def build_tridiagonal(n, diag=4, off_diag=1):
    data = [[0] * n for _ in range(n)]
    for i in range(n):
        data[i][i] = diag
        if i > 0:
            data[i][i - 1] = off_diag
        if i < n - 1:
            data[i][i + 1] = off_diag
    return Matrix(data)


def compare_tridiagonal():
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МЕТОДОВ ДЛЯ ТРЕХДИАГОНАЛЬНЫХ СИСТЕМ")
    print("=" * 60)

    sizes = [10, 50, 100]

    for n in sizes:
        print(f"\nРазмерность n = {n}")
        print("-" * 40)

        A = build_tridiagonal(n)
        x_true = Matrix([[1.0] for _ in range(n)])
        b = A * x_true

        start = perf_counter()
        x_t, ops_t = thomas(A, b)
        time_t = perf_counter() - start
        try:
            err_t = (x_true - x_t).norm_1()
        except:
            err_t = (x_true - x_t).normL1()

        start = perf_counter()
        x_g, _, ops_g = gauss_solve(A, b)
        time_g = perf_counter() - start
        try:
            err_g = (x_true - x_g).norm_1()
        except:
            err_g = (x_true - x_g).normL1()

        start = perf_counter()
        solver = LUSolver()
        ops_decomp = solver.lup_decomposition(A)
        x_lup = solver.solve(b)
        time_lup = perf_counter() - start
        try:
            err_lup = (x_true - x_lup).norm_1()
        except:
            err_lup = (x_true - x_lup).normL1()
        ops_lup_total = ops_decomp + 2 * n ** 2

        print(f"Метод Томаса:  {time_t * 1e3:7.2f} мс, ops = {ops_t:7d}, ошибка = {err_t:.2e}")
        print(f"Метод Гаусса:  {time_g * 1e3:7.2f} мс, ops = {ops_g:7d}, ошибка = {err_g:.2e}")
        print(f"LUP:           {time_lup * 1e3:7.2f} мс, ops = {ops_lup_total:7d}, ошибка = {err_lup:.2e}")


if __name__ == "__main__":
    test_lu_vs_lup()
    test_multiple_rhs()
    compare_tridiagonal()

    print("\n" + "=" * 60)
    print("ВЫВОДЫ ПО ЗАДАНИЮ 7.2:")
    print("=" * 60)
    print("1. Реализовано LU-разложение (без выбора главного элемента)")
    print("2. Реализовано LUP-разложение (с выбором главного элемента)")
    print("3. Реализовано решение систем с использованием готовых разложений")
    print("4. Реализовано вычисление определителя через разложение")
    print("5. Продемонстрирована эффективность решения нескольких систем")
    print("   с одной матрицей и разными правыми частями")
    print("6. Проведено сравнение числа арифметических операций")
    print("7. Включена проверка корректности разложений (A = L*U, P*A = L*U)")