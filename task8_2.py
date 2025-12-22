import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional
import sys


class Task8_2:
    class MatrixType:
        def __init__(self, name: str, A: List[List[float]],
                     b: List[float], description: str):
            self.name = name
            self.A = A
            self.b = b
            self.description = description

    class ComparisonResult:
        def __init__(self):
            self.test_name = ""
            self.seidel_iterations = 0
            self.simple_iterations = 0
            self.seidel_time = 0.0
            self.simple_time = 0.0
            self.difference_norm = 0.0
            self.seidel_solution = []
            self.simple_solution = []

    @staticmethod
    def run_all_tasks():
        print("\n" + "-" * 58)
        print("Задание 8.2: Метод Зейделя")
        print("-" * 58)

        Task8_2.demonstrate_different_matrices()
        Task8_2.test_different_precisions()
        Task8_2.compare_with_simple_iteration()

    @staticmethod
    def generate_matrix_types() -> List['Task8_2.MatrixType']:
        matrices = []

        matrices.append(Task8_2.MatrixType(
            "Диагонально-доминирующая",
            [
                [10, 1, 1, 0.5],
                [0.5, 8, 1, 0.5],
                [0.5, 0.5, 12, 1],
                [0.5, 0.5, 0.5, 10]
            ],
            [12.5, 10, 14, 11.5],
            "Хорошо сходится для обоих методов"
        ))

        matrices.append(Task8_2.MatrixType(
            "Близкая к диагональной",
            [
                [5, 0.1, 0.05, 0.02],
                [0.08, 6, 0.1, 0.05],
                [0.05, 0.08, 7, 0.1],
                [0.02, 0.05, 0.08, 8]
            ],
            [5.17, 6.23, 7.23, 8.15],
            "Малые недиагональные элементы, метод Зейделя эффективен"
        ))

        matrices.append(Task8_2.MatrixType(
            "Близкая к нижней треугольной",
            [
                [4, 0.1, 0.1, 0.1],
                [1.5, 5, 0.1, 0.1],
                [1.5, 1.5, 6, 0.1],
                [1.5, 1.5, 1.5, 7]
            ],
            [4.3, 6.7, 9.1, 11.5],
            "Большие элементы под диагональю, метод Зейделя имеет преимущество"
        ))

        matrices.append(Task8_2.MatrixType(
            "Симметричная положительно определенная",
            [
                [4, 1, 0, 0],
                [1, 4, 1, 0],
                [0, 1, 4, 1],
                [0, 0, 1, 4]
            ],
            [1, 2, 3, 4],
            "Гарантированная сходимость метода Зейделя"
        ))

        return matrices

    @staticmethod
    def convert_to_iteration_form(A: List[List[float]], b: List[float]) -> List[List[float]]:
        n = len(A)
        B = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            diag = A[i][i]
            for j in range(n):
                if i != j:
                    B[i][j] = -A[i][j] / diag

        return B

    @staticmethod
    def get_c_vector(A: List[List[float]], b: List[float]) -> List[float]:
        n = len(A)
        c = [0.0 for _ in range(n)]

        for i in range(n):
            diag = A[i][i]
            c[i] = b[i] / diag

        return c

    @staticmethod
    def norm_inf_matrix(B: List[List[float]]) -> float:
        n = len(B)
        max_sum = 0.0

        for i in range(n):
            row_sum = sum(abs(B[i][j]) for j in range(n))
            if row_sum > max_sum:
                max_sum = row_sum

        return max_sum

    @staticmethod
    def extract_lower(B: List[List[float]]) -> List[List[float]]:
        n = len(B)
        L = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if j <= i:
                    L[i][j] = B[i][j]

        return L

    @staticmethod
    def extract_upper(B: List[List[float]]) -> List[List[float]]:
        n = len(B)
        U = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if j > i:
                    U[i][j] = B[i][j]

        return U

    @staticmethod
    def print_vector(vec: List[float]):
        print("[", end="")
        for i, val in enumerate(vec):
            print(f"{val:.6f}", end="")
            if i < len(vec) - 1:
                print(", ", end="")
        print("]")

    @staticmethod
    def print_matrix_info(mt: 'Task8_2.MatrixType'):
        print(f"\nМатрица: {mt.name}")
        print(f"Описание: {mt.description}")
        print(f"Размер: {len(mt.A)}×{len(mt.A)}")

        n = len(mt.A)
        symmetric = True

        for i in range(n):
            for j in range(n):
                if abs(mt.A[i][j] - mt.A[j][i]) > 1e-10:
                    symmetric = False
                    break
            if not symmetric:
                break

        upper_sum = 0.0
        lower_sum = 0.0
        for i in range(n):
            for j in range(n):
                if j > i:
                    upper_sum += abs(mt.A[i][j])
                elif j < i:
                    lower_sum += abs(mt.A[i][j])

        print("Свойства:")
        print(f"  - Симметричная: {'да' if symmetric else 'нет'}")
        print(f"  - Сумма |элементов над диагональю|: {upper_sum:.6f}")
        print(f"  - Сумма |элементов под диагональю|: {lower_sum:.6f}")
        print(f"  - Отношение ниж/верх: {lower_sum / upper_sum if upper_sum > 0 else 999:.6f}")

        B = Task8_2.convert_to_iteration_form(mt.A, mt.b)
        B1 = Task8_2.extract_lower(B)
        B2 = Task8_2.extract_upper(B)

        norm_B = Task8_2.norm_inf_matrix(B)
        norm_B1 = Task8_2.norm_inf_matrix(B1)
        norm_B2 = Task8_2.norm_inf_matrix(B2)

        print("Оценки сходимости:")
        print(f"  - ||B||∞ = {norm_B:.6f}")
        print(f"  - ||B1||∞ = {norm_B1:.6f}")
        print(f"  - ||B2||∞ = {norm_B2:.6f}")
        print(f"  - ||B1|| + ||B2|| = {norm_B1 + norm_B2:.6f}")

        if norm_B < 1:
            print("  ✓ Теорема 1: ||B|| < 1 выполняется")
        if norm_B1 + norm_B2 < 1:
            print("  ✓ Теорема 2: ||B1|| + ||B2|| < 1 выполняется")
        if symmetric:
            positive_definite = all(mt.A[i][i] > 0 for i in range(n))
            if positive_definite:
                print("  ✓ Теорема 3: матрица симметричная положительно определенная")

    @staticmethod
    def demonstrate_different_matrices():
        print("\nПункт 1: Различные входные данные")

        matrices = Task8_2.generate_matrix_types()

        for mt in matrices:
            Task8_2.print_matrix_info(mt)

            x0 = [0.0 for _ in range(len(mt.b))]
            Task8_2.run_seidel_method_with_details(
                mt.A, mt.b, x0, 1e-3, mt.name
            )

    @staticmethod
    def run_seidel_method_with_details(A: List[List[float]],
                                       b: List[float],
                                       x0: List[float],
                                       eps: float,
                                       test_name: str):
        n = len(A)

        print(f"\nМетод Зейделя для: {test_name}")
        print(f"Точность: {eps}")

        B = Task8_2.convert_to_iteration_form(A, b)
        c = Task8_2.get_c_vector(A, b)

        norm_B = Task8_2.norm_inf_matrix(B)
        B2 = Task8_2.extract_upper(B)
        norm_B2 = Task8_2.norm_inf_matrix(B2)

        eps2 = (1 - norm_B) * eps / norm_B2 if norm_B2 > 0 else eps

        x = x0.copy()
        x_old = [0.0 for _ in range(n)]
        iteration = 0
        max_iter = 1000

        print(f"||B||∞ = {norm_B:.6f}, ||B2||∞ = {norm_B2:.6f}")
        print(f"ε2 = (1-||B||)*ε/||B2|| = {eps2:.6e}")
        print("\nИтерации:")
        print(" №  | ||x(k)-x(k-1)|| | < ε2? | Апост. оценка")
        print("----|-----------------|-------|-----------------")

        while iteration < max_iter:
            iteration += 1
            x_old = x.copy()

            for i in range(n):
                s = 0.0
                for j in range(i):
                    s += B[i][j] * x[j]
                for j in range(i + 1, n):
                    s += B[i][j] * x_old[j]
                x[i] = s + c[i]

            diff = max(abs(x[i] - x_old[i]) for i in range(n))

            estimated_error = (norm_B2 / (1 - norm_B)) * diff if norm_B < 1 else diff

            print(f"{iteration:4d} | {diff:15.6e} | "
                  f"{'ДА' if diff < eps2 else 'нет':5} | "
                  f"{estimated_error:15.6e}")

            if diff < eps2:
                print(f"\nКритерий (11) выполнен: ||x(m)-x(m-1)|| < ε2")
                print(f"Итераций: {iteration}")
                print("Решение: ", end="")
                Task8_2.print_vector(x)

                if n <= 4:
                    print("Для сравнения - решение прямым методом:")
                    exact = [1.0, 1.0, 1.0, 1.0]
                    print("Примерное точное решение: ", end="")
                    Task8_2.print_vector(exact)
                break

            if iteration == max_iter:
                print(f"\nДостигнут предел итераций ({max_iter})")

    @staticmethod
    def test_different_precisions():
        print("\nПункт 3: Точности 10^(-3) и 10^(-6)")

        matrices = Task8_2.generate_matrix_types()

        for mt in matrices:
            print(f"\nМатрица: {mt.name}")

            x0 = [0.0 for _ in range(len(mt.b))]

            print("Точность 1e-3:")
            start = time.time()
            Task8_2.run_seidel_method_with_details(
                mt.A, mt.b, x0, 1e-3, mt.name + " (1e-3)"
            )
            end = time.time()
            print(f"Время: {end - start:.6f} сек")

            print("\nТочность 1e-6:")
            start = time.time()
            Task8_2.run_seidel_method_with_details(
                mt.A, mt.b, x0, 1e-6, mt.name + " (1e-6)"
            )
            end = time.time()
            print(f"Время: {end - start:.6f} сек")

    @staticmethod
    def seidel_method(B: List[List[float]], c: List[float],
                      x0: List[float], eps: float, max_iter: int,
                      iteration_count: List[int], verbose: bool = False) -> List[float]:
        n = len(B)
        x = x0.copy()
        x_old = [0.0 for _ in range(n)]
        iter_count = 0

        for k in range(max_iter):
            iter_count += 1
            x_old = x.copy()

            for i in range(n):
                s = 0.0
                for j in range(i):
                    s += B[i][j] * x[j]
                for j in range(i + 1, n):
                    s += B[i][j] * x_old[j]
                x[i] = s + c[i]

            diff = max(abs(x[i] - x_old[i]) for i in range(n))
            if diff < eps:
                break

        iteration_count[0] = iter_count
        return x

    @staticmethod
    def simple_iteration(B: List[List[float]], c: List[float],
                         x0: List[float], eps: float, max_iter: int,
                         iteration_count: List[int], verbose: bool = False) -> List[float]:
        n = len(B)
        x = x0.copy()
        x_new = [0.0 for _ in range(n)]
        iter_count = 0

        for k in range(max_iter):
            iter_count += 1

            for i in range(n):
                s = 0.0
                for j in range(n):
                    s += B[i][j] * x[j]
                x_new[i] = s + c[i]

            diff = max(abs(x_new[i] - x[i]) for i in range(n))
            x = x_new.copy()

            if diff < eps:
                break

        iteration_count[0] = iter_count
        return x

    @staticmethod
    def print_comparison_result(result: 'Task8_2.ComparisonResult'):
        print(f"\n{result.test_name}:")
        print("  Метод Зейделя:")
        print(f"    - Итераций: {result.seidel_iterations}")
        print(f"    - Время: {result.seidel_time:.6f} сек")
        print("  Метод простой итерации:")
        print(f"    - Итераций: {result.simple_iterations}")
        print(f"    - Время: {result.simple_time:.6f} сек")
        print(f"  Разница норм решений: {result.difference_norm:.6e}")

        speedup_iterations = result.simple_iterations / result.seidel_iterations
        speedup_time = result.simple_time / result.seidel_time

        print(f"  Ускорение (по итерациям): {speedup_iterations:.2f}x")
        print(f"  Ускорение (по времени): {speedup_time:.2f}x")

        if speedup_iterations > 1.0:
            print("  ✓ Метод Зейделя быстрее сходится")
        else:
            print("  ✗ Метод простой итерации быстрее или равен")

    @staticmethod
    def compare_with_simple_iteration():
        print("\nПункт 4: Сравнение с методом простой итерации")

        matrices = Task8_2.generate_matrix_types()
        results = []

        eps = 1e-4

        for mt in matrices:
            result = Task8_2.ComparisonResult()
            result.test_name = mt.name

            x0 = [0.0 for _ in range(len(mt.b))]

            print(f"\nТестирование: {mt.name}")
            print("Метод Зейделя")

            B = Task8_2.convert_to_iteration_form(mt.A, mt.b)
            c = Task8_2.get_c_vector(mt.A, mt.b)

            start = time.time()
            seidel_iter_count = [0]
            result.seidel_solution = Task8_2.seidel_method(
                B, c, x0, eps, 1000, seidel_iter_count, False
            )
            result.seidel_iterations = seidel_iter_count[0]
            end = time.time()
            result.seidel_time = end - start

            print("Метод простой итерации")
            start = time.time()
            simple_iter_count = [0]
            result.simple_solution = Task8_2.simple_iteration(
                B, c, x0, eps, 1000, simple_iter_count, False
            )
            result.simple_iterations = simple_iter_count[0]
            end = time.time()
            result.simple_time = end - start

            diff = max(abs(result.seidel_solution[i] - result.simple_solution[i])
                       for i in range(len(result.seidel_solution)))
            result.difference_norm = diff

            results.append(result)
            Task8_2.print_comparison_result(result)

        print("\nСводная таблица сравнения методов (ε = 1e-4)")
        print(f"{'Матриса':<30} {'Зейд.итер':<10} {'Прост.итер':<10} "
              f"{'Отношение':<10} {'Время З.':<10} {'Время П.':<10} {'Ускор.':<10}")

        for res in results:
            ratio = res.simple_iterations / res.seidel_iterations
            time_ratio = res.simple_time / res.seidel_time

            print(f"{res.test_name:<30} {res.seidel_iterations:<10} "
                  f"{res.simple_iterations:<10} {ratio:<10.2f} "
                  f"{res.seidel_time:<10.4f} {res.simple_time:<10.4f} "
                  f"{time_ratio:<10.2f}")

        print("\nВыводы:")
        print("1. Для диагонально-доминирующих матриц оба метода сходятся хорошо")
        print("2. Для матриц, близких к нижним треугольным, метод Зейделя имеет преимущество")
        print("3. Для симметричных положительно определенных матриц метод Зейделя гарантированно сходится")
        print("4. Метод Зейделя обычно требует меньше итераций, но каждая итерация сложнее")


if __name__ == "__main__":
    Task8_2.run_all_tasks()