import numpy as np
import time
import random
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt


class MatrixUtils:
    @staticmethod
    def print_matrix(A: np.ndarray):
        """Вывод матрицы"""
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                print(f"{A[i, j]:10.6f}", end=" ")
            print()

    @staticmethod
    def print_vector(v: np.ndarray):
        """Вывод вектора"""
        print("[", end="")
        for i, val in enumerate(v):
            if i > 0:
                print(", ", end="")
            print(f"{val:10.6f}", end="")
        print("]")

    @staticmethod
    def norm_inf_matrix(A: np.ndarray) -> float:
        """Вычисление нормы матрицы ||A||∞"""
        n = A.shape[0]
        max_sum = 0.0
        for i in range(n):
            row_sum = 0.0
            for j in range(n):
                row_sum += abs(A[i, j])
            if row_sum > max_sum:
                max_sum = row_sum
        return max_sum

    @staticmethod
    def norm_inf_vector(v: np.ndarray) -> float:
        """Вычисление нормы вектора ||v||∞"""
        return np.max(np.abs(v))


class IterativeUtils:
    @staticmethod
    def convert_to_iteration_form(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Преобразование системы Ax=b к виду x = Bx + c"""
        n = A.shape[0]
        B = np.zeros((n, n))
        c = np.zeros(n)

        for i in range(n):
            if abs(A[i, i]) < 1e-12:
                raise ValueError(f"Нулевой диагональный элемент a[{i},{i}]")

            c[i] = b[i] / A[i, i]

            for j in range(n):
                if i != j:
                    B[i, j] = -A[i, j] / A[i, i]

        return B, c


class Task8_1:
    class TestCase:
        def __init__(self, name: str, A: np.ndarray, b: np.ndarray, description: str):
            self.name = name
            self.A = A
            self.b = b
            self.description = description

    class IterationStats:
        def __init__(self, iteration: int, x: np.ndarray, diff: float, estimated_error: float):
            self.iteration = iteration
            self.x = x
            self.diff = diff
            self.estimated_error = estimated_error

    def run_all_tasks(self):
        print("\n" + "=" * 60)
        print("Задание 8.1: Метод простой итерации")
        print("=" * 60)

        self.demonstrate_different_inputs()
        self.test_different_initial_approximations()
        self.test_different_precisions()
        self.compare_stopping_criteria()
        self.evaluate_operation_count()

    def generate_test_cases(self) -> List[TestCase]:
        """Генерация тестовых систем"""
        test_cases = []

        test_cases.append(self.TestCase(
            name="Диагонально-доминирующая",
            A=np.array([[10, 1, 1],
                        [1, 10, 1],
                        [1, 1, 10]], dtype=float),
            b=np.array([12, 12, 12], dtype=float),
            description="Сильная диагональная доминация, быстро сходится"
        ))

        test_cases.append(self.TestCase(
            name="Близкая к диагональной",
            A=np.array([[5, 0.5, 0.1],
                        [0.3, 4, 0.2],
                        [0.1, 0.2, 6]], dtype=float),
            b=np.array([5.6, 4.5, 6.3], dtype=float),
            description="Малые недиагональные элементы"
        ))

        test_cases.append(self.TestCase(
            name="Слабая диагональная доминация",
            A=np.array([[5, 2, 2],
                        [1, 4, 2],
                        [1, 1, 3]], dtype=float),
            b=np.array([9, 7, 5], dtype=float),
            description="Диагональная доминация на грани сходимости"
        ))

        test_cases.append(self.TestCase(
            name="Симметричная положительно определенная",
            A=np.array([[4, 1, 0],
                        [1, 3, 1],
                        [0, 1, 2]], dtype=float),
            b=np.array([1, 2, 3], dtype=float),
            description="Гарантированная сходимость по теореме"
        ))

        return test_cases

    def print_test_case(self, tc: TestCase):
        """Вывод информации о тестовом случае"""
        print(f"\nТест: {tc.name}")
        print(f"Описание: {tc.description}")
        print("Матрица A:")
        MatrixUtils.print_matrix(tc.A)
        print("Вектор b: ", end="")
        MatrixUtils.print_vector(tc.b)

        B, _ = IterativeUtils.convert_to_iteration_form(tc.A, tc.b)
        normB = MatrixUtils.norm_inf_matrix(B)
        print(f"Норма матрицы B (||B||∞): {normB:.6f}")
        if normB < 1:
            print("✓ Условие сходимости ||B|| < 1 выполняется")
        else:
            print("✗ Условие сходимости ||B|| < 1 НЕ выполняется")

    def demonstrate_different_inputs(self):
        """Пункт 1: Различные входные данные"""
        print("\nПункт 1: Различные входные данные")
        print("-" * 40)

        test_cases = self.generate_test_cases()

        for tc in test_cases:
            self.print_test_case(tc)

            x0 = np.zeros_like(tc.b)
            self.run_simple_iteration_with_details(tc.A, tc.b, x0, 1e-3, tc.name)

    def test_different_initial_approximations(self):
        """Пункт 2: Различные начальные приближения"""
        print("\nПункт 2: Различные начальные приближения")
        print("-" * 40)

        test_cases = self.generate_test_cases()
        tc = test_cases[0]

        print(f"Тестовая система: {tc.name}")
        print(f"Точность: 1e-6\n")

        n = len(tc.b)

        print("1. Нулевой вектор:")
        x0_zero = np.zeros(n)
        self.run_simple_iteration_with_details(tc.A, tc.b, x0_zero, 1e-6, "Нулевой вектор")

        print("\n2. Вектор из единиц:")
        x0_ones = np.ones(n)
        self.run_simple_iteration_with_details(tc.A, tc.b, x0_ones, 1e-6, "Вектор из единиц")

        print("\n3. Вектор, равный правой части:")
        x0_b = tc.b.copy()
        self.run_simple_iteration_with_details(tc.A, tc.b, x0_b, 1e-6, "Правая часть")

        print("\n4. Случайный вектор:")
        random.seed(time.time())
        x0_random = np.array([random.random() * 10 for _ in range(n)])
        print("Начальное приближение: ", end="")
        MatrixUtils.print_vector(x0_random)
        self.run_simple_iteration_with_details(tc.A, tc.b, x0_random, 1e-6, "Случайный вектор")

    def print_iteration_details(self, stats: IterationStats, n: int):
        """Вывод деталей итерации"""
        print(f"{stats.iteration:4d} | ", end="")

        # Первые 3 компоненты вектора
        for i in range(min(3, n)):
            print(f"{stats.x[i]:12.6f}", end=" ")

        if n > 3:
            print("...", end=" ")

        print(f"| {stats.diff:12.2e} | {stats.estimated_error:12.2e}")

    def run_simple_iteration_with_details(self, A: np.ndarray, b: np.ndarray,
                                          x0: np.ndarray, eps: float, test_name: str):
        """Запуск метода простой итерации с выводом деталей"""
        n = A.shape[0]

        B, c = IterativeUtils.convert_to_iteration_form(A, b)
        normB = MatrixUtils.norm_inf_matrix(B)
        eps1 = (1 - normB) * eps / normB if normB > 0 else eps / 100

        print(f"Начальное приближение: ", end="")
        MatrixUtils.print_vector(x0)
        print(f"||B||∞ = {normB:.6f}, ε1 = {eps1:.2e}")

        x_old = x0.copy()
        x_new = np.zeros(n)
        iteration = 0
        max_iter = 1000

        print("\nИтерации:")
        print(" №  | x (первые 3 компоненты)      | ||x(k)-x(k-1)|| | Апост. оценка")
        print("-" * 70)

        while iteration < max_iter:
            iteration += 1

            for i in range(n):
                sum_val = 0.0
                for j in range(n):
                    sum_val += B[i, j] * x_old[j]
                x_new[i] = sum_val + c[i]

            diff = 0.0
            for i in range(n):
                d = abs(x_new[i] - x_old[i])
                if d > diff:
                    diff = d

            estimated_error = (normB / (1 - normB)) * diff if normB < 1 else diff

            stats = self.IterationStats(
                iteration=iteration,
                x=x_new.copy(),
                diff=diff,
                estimated_error=estimated_error
            )

            self.print_iteration_details(stats, n)

            criterion18 = (diff < eps1)
            criterion19 = (diff < eps)

            if criterion18 or criterion19:
                print("\nДостигнута точность!")
                print(f"Критерий (18) {'✓' if criterion18 else '✗'}, "
                      f"Критерий (19) {'✓' if criterion19 else '✗'}")
                print(f"Итераций: {iteration}")
                print("Решение: ", end="")
                MatrixUtils.print_vector(x_new)
                break

            x_old = x_new.copy()

            if iteration == max_iter:
                print(f"\nДостигнут предел итераций ({max_iter})")

    def test_different_precisions(self):
        """Пункт 4: Точности 10^(-3) и 10^(-6)"""
        print("\nПункт 4: Точности 10^(-3) и 10^(-6)")
        print("-" * 40)

        test_cases = self.generate_test_cases()

        for tc in test_cases:
            print(f"\nСистема: {tc.name}")

            x0 = np.zeros_like(tc.b)

            print("Точность 1e-3:")
            start_time = time.time()
            self.run_simple_iteration_with_details(tc.A, tc.b, x0, 1e-3, f"{tc.name} (1e-3)")
            end_time = time.time()
            print(f"Время: {end_time - start_time:.6f} сек")

            print("\nТочность 1e-6:")
            start_time = time.time()
            self.run_simple_iteration_with_details(tc.A, tc.b, x0, 1e-6, f"{tc.name} (1e-6)")
            end_time = time.time()
            print(f"Время: {end_time - start_time:.6f} сек")

    def compare_stopping_criteria(self):
        """Пункт 5: Сравнение критериев окончания"""
        print("\nПункт 5: Сравнение критериев окончания")
        print("-" * 40)

        test_cases = self.generate_test_cases()
        tc = test_cases[2]

        print(f"Тестовая система: {tc.name}")
        print(f"{tc.description}\n")

        B, _ = IterativeUtils.convert_to_iteration_form(tc.A, tc.b)
        normB = MatrixUtils.norm_inf_matrix(B)

        print(f"||B||∞ = {normB:.6f}")
        print(f"(1 - ||B||)/||B|| = {(1 - normB) / normB:.6f}\n")

        if normB <= 0.5:
            print("||B|| ≤ 0.5, оба критерия обоснованы")
        else:
            print("||B|| > 0.5, критерий (19) может приводить к преждевременной остановке")

        print("\nСравнение на практике (ε = 1e-4):")

        eps = 1e-4
        eps1 = (1 - normB) * eps / normB if normB > 0 else eps / 100

        print(f"ε = {eps:.1e}, ε1 = {eps1:.1e}")
        print(f"ε1/ε = {eps1 / eps:.6f} (во сколько раз строже критерий (18))")

        # Практическое сравнение
        x0 = np.zeros_like(tc.b)

        print("\nПрактическое сравнение:")
        print("Критерий (18) (правильный):")
        self.run_simple_iteration_with_details(tc.A, tc.b, x0, eps, "Критерий (18)")

        print("\nКритерий (19) (простой):")
        x0 = np.zeros_like(tc.b)
        self.run_simple_iteration_simple_criterion(tc.A, tc.b, x0, eps, "Критерий (19)")

    def run_simple_iteration_simple_criterion(self, A: np.ndarray, b: np.ndarray,
                                              x0: np.ndarray, eps: float, test_name: str):
        """Метод простой итерации с простым критерием остановки"""
        n = A.shape[0]

        B, c = IterativeUtils.convert_to_iteration_form(A, b)

        x_old = x0.copy()
        x_new = np.zeros(n)
        iteration = 0
        max_iter = 1000

        while iteration < max_iter:
            iteration += 1

            for i in range(n):
                sum_val = 0.0
                for j in range(n):
                    sum_val += B[i, j] * x_old[j]
                x_new[i] = sum_val + c[i]

            diff = 0.0
            for i in range(n):
                d = abs(x_new[i] - x_old[i])
                if d > diff:
                    diff = d

            if diff < eps:
                print(f"Итераций: {iteration}")
                print("Решение: ", end="")
                MatrixUtils.print_vector(x_new)
                break

            x_old = x_new.copy()

    def evaluate_operation_count(self):
        """Пункт 6: Оценка числа арифметических операций"""
        print("\nПункт 6: Оценка числа арифметических операций")
        print("-" * 40)

        print("Для системы размера n×n на одной итерации метода простой итерации:")
        print("1. Умножение B*x: n² умножений и n² сложений")
        print("2. Добавление вектора c: n сложений")
        print("3. Норма разности: n вычитаний и n сравнений")
        print("Итого на итерацию: ~2n² + 2n арифметических операций\n")

        sizes = [3, 10, 50, 100]

        for n in sizes:
            ops_per_iteration = 2.0 * n * n + 2.0 * n
            print(f"n = {n}: {ops_per_iteration:.0f} операций/итерация")

            total_ops = 100 * ops_per_iteration
            print(f"   За 100 итераций: {total_ops:.0f} операций")
            print(f"   (~{total_ops / 1e6:.2f} млн. операций)\n")

    def visualize_convergence(self):
        """Визуализация сходимости метода"""
        print("\nДополнительно: Визуализация сходимости")
        print("-" * 40)

        # Генерация нескольких систем разного размера
        sizes = [3, 5, 10]
        iterations_data = []
        norms_data = []

        for n in sizes:
            # Генерация случайной диагонально-доминирующей матрицы
            A = np.random.randn(n, n) * 0.1
            for i in range(n):
                A[i, i] = 1.0 + abs(np.random.randn()) * 2
                # Гарантируем диагональное преобладание
                row_sum = np.sum(np.abs(A[i, :])) - abs(A[i, i])
                A[i, i] += row_sum + 0.1

            # Генерация правой части
            x_true = np.random.randn(n)
            b = A @ x_true

            # Запуск метода
            x0 = np.zeros(n)
            B, c = IterativeUtils.convert_to_iteration_form(A, b)
            normB = MatrixUtils.norm_inf_matrix(B)

            x_old = x0.copy()
            diff_history = []
            residual_history = []

            for k in range(50):
                x_new = B @ x_old + c
                diff = np.max(np.abs(x_new - x_old))
                residual = np.max(np.abs(A @ x_new - b))

                diff_history.append(diff)
                residual_history.append(residual)

                x_old = x_new.copy()

            iterations_data.append((n, diff_history))
            norms_data.append((n, normB))

        # Построение графиков
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # График 1: Сходимость для разных размеров
        ax1 = axes[0, 0]
        for n, diff_history in iterations_data:
            ax1.semilogy(diff_history, label=f'n={n}')
        ax1.set_xlabel('Итерация')
        ax1.set_ylabel('||x(k)-x(k-1)||∞')
        ax1.set_title('Сходимость метода для разных размеров систем')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График 2: Нормы матриц B
        ax2 = axes[0, 1]
        sizes, normBs = zip(*norms_data)
        ax2.bar([str(s) for s in sizes], normBs)
        ax2.set_xlabel('Размер системы n')
        ax2.set_ylabel('||B||∞')
        ax2.set_title('Нормы матриц B для разных размеров')
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Граница сходимости')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # График 3: Зависимость числа итераций от нормы B
        ax3 = axes[1, 0]
        iteration_counts = []
        for n, diff_history in iterations_data:
            # Находим итерацию, где достигнута точность 1e-6
            for i, diff in enumerate(diff_history):
                if diff < 1e-6:
                    iteration_counts.append((n, i + 1))
                    break
            else:
                iteration_counts.append((n, len(diff_history)))

        sizes, counts = zip(*iteration_counts)
        ax3.plot(sizes, counts, 'bo-', linewidth=2, markersize=8)
        ax3.set_xlabel('Размер системы n')
        ax3.set_ylabel('Итераций до точности 1e-6')
        ax3.set_title('Зависимость числа итераций от размера системы')
        ax3.grid(True, alpha=0.3)

        # График 4: Сравнение критериев остановки
        ax4 = axes[1, 1]
        test_cases = self.generate_test_cases()
        tc = test_cases[1]

        B, c = IterativeUtils.convert_to_iteration_form(tc.A, tc.b)
        normB = MatrixUtils.norm_inf_matrix(B)

        eps = 1e-4
        eps1 = (1 - normB) * eps / normB

        # Запуск с обоими критериями
        x0 = np.zeros_like(tc.b)
        diff_history_18 = []
        diff_history_19 = []

        x_old = x0.copy()
        for k in range(50):
            x_new = B @ x_old + c
            diff = np.max(np.abs(x_new - x_old))
            diff_history_18.append(diff)
            diff_history_19.append(diff)
            x_old = x_new.copy()

        ax4.semilogy(diff_history_18, 'b-', label='Разность на итерациях')
        ax4.axhline(y=eps, color='g', linestyle='--', label=f'ε = {eps:.0e}')
        ax4.axhline(y=eps1, color='r', linestyle='--', label=f'ε1 = {eps1:.0e}')
        ax4.set_xlabel('Итерация')
        ax4.set_ylabel('||x(k)-x(k-1)||∞')
        ax4.set_title('Сравнение критериев остановки')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    """Главная функция"""
    task = Task8_1()

    print("=" * 60)
    print("МЕТОД ПРОСТОЙ ИТЕРАЦИИ ДЛЯ СИСТЕМ ЛИНЕЙНЫХ УРАВНЕНИЙ")
    print("=" * 60)

    # Выполнение всех заданий
    task.run_all_tasks()

    # Дополнительная визуализация
    task.visualize_convergence()


if __name__ == "__main__":
    main()