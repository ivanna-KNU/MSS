import numpy as np
import math

DELTA_T = 0.01
T = 5.0
N = int(T / DELTA_T)

def read_data(filename):
    with open(filename, 'r') as f:
        content = f.read().strip()
        values = [float(x) for x in content.split() if x]
    return np.array(values)

def dft(x):
    """
    Дискретне Перетворення Фур'є
    c_x(k) = (1/N) Σ_{m=0}^{N-1} x(m)e^(-i2πkm/N)
    """
    N = len(x)
    c = np.zeros(N, dtype=complex)
    
    for k in range(N):
        sum_val = 0.0 + 0.0j
        for m in range(N):
            angle = -2 * math.pi * k * m / N
            sum_val += x[m] * (math.cos(angle) + 1j * math.sin(angle))
        c[k] = sum_val / N
    
    return c

def find_dominant_frequencies(dft_coeffs, num_freqs=None, threshold=None):
    """
    Знаходить домінуючі частоти з DFT спектру
    Повертає частоти та їх індекси
    """
    amplitudes = np.abs(dft_coeffs)
    
    half_N = len(amplitudes) // 2
    relevant_amplitudes = amplitudes[1:half_N+1]
    relevant_indices = np.arange(1, half_N+1)
    
    if threshold is not None:
        mask = relevant_amplitudes > threshold
        selected_indices = relevant_indices[mask]
        selected_amplitudes = relevant_amplitudes[mask]
    elif num_freqs is not None:
        top_indices = np.argsort(relevant_amplitudes)[-num_freqs:]
        selected_indices = relevant_indices[top_indices]
        selected_amplitudes = relevant_amplitudes[top_indices]
    else:
        top_indices = np.argsort(relevant_amplitudes)[-5:]
        selected_indices = relevant_indices[top_indices]
        selected_amplitudes = relevant_amplitudes[top_indices]
    
    frequencies = selected_indices / (N * DELTA_T)
    
    sort_order = np.argsort(frequencies)
    frequencies = frequencies[sort_order]
    selected_indices = selected_indices[sort_order]
    selected_amplitudes = selected_amplitudes[sort_order]
    
    return frequencies, selected_indices, selected_amplitudes

def build_least_squares_system(t, y_observed, frequencies):
    """
    Побудова системи рівнянь для методу найменших квадратів
    y(t) = a₁t³ + a₂t² + a₃t + Σ aᵢ sin(2πfᵢ₋₃t) + aₖ₊₁
    
    Повертає матрицю A та вектор b
    """
    num_points = len(t)
    num_freqs = len(frequencies)
    num_coeffs = 3 + num_freqs + 1
    
    A = np.zeros((num_points, num_coeffs))
    
    A[:, 0] = t**3
    A[:, 1] = t**2
    A[:, 2] = t
    
    for i, freq in enumerate(frequencies):
        A[:, 3 + i] = np.sin(2 * math.pi * freq * t)
    
    A[:, -1] = 1.0
    
    b = y_observed
    
    return A, b

def solve_coefficients(A, b):
    coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return coeffs, residuals

def main():
    print("Лабораторна робота 1: Визначення моделі функції з використанням DFT")
    print("=" * 70)
    
    print("\n1. Читання даних з f10.txt...")
    y_observed = read_data('f10.txt')
    print(f"   Прочитано {len(y_observed)} значень")
    
    if len(y_observed) != N:
        print(f"   Увага: очікувалось {N} точок, отримано {len(y_observed)}")
        actual_N = len(y_observed)
    else:
        actual_N = N
    
    t = np.array([i * DELTA_T for i in range(actual_N)])
    
    print(f"   Інтервал часу: [{t[0]:.2f}, {t[-1]:.2f}]")
    print(f"   Крок часу: {DELTA_T}")
    
    print("\n2. Обчислення DFT...")
    dft_coeffs = dft(y_observed)
    print(f"   Обчислено {len(dft_coeffs)} коефіцієнтів DFT")
    
    print("\n3. Знаходження домінуючих частот...")
    frequencies, freq_indices, freq_amplitudes = find_dominant_frequencies(
        dft_coeffs, num_freqs=5
    )
    print(f"   Знайдено {len(frequencies)} домінуючих частот:")
    for i, (freq, amp) in enumerate(zip(frequencies, freq_amplitudes)):
        print(f"   f{i+1} = {freq:.6f} Гц (амплітуда: {amp:.6f})")
    
    print("\n4. Побудова системи рівнянь для методу найменших квадратів...")
    A, b = build_least_squares_system(t, y_observed, frequencies)
    print(f"   Розмір матриці A: {A.shape}")
    print(f"   Кількість коефіцієнтів: {A.shape[1]}")
    
    print("\n5. Розв'язання системи...")
    coeffs, residuals = solve_coefficients(A, b)
    print("   Систему розв'язано успішно")
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТИ:")
    print("=" * 70)
    
    print("\nКоефіцієнти моделі:")
    print(f"  a₁ (коефіцієнт при t³): {coeffs[0]:.10f}")
    print(f"  a₂ (коефіцієнт при t²): {coeffs[1]:.10f}")
    print(f"  a₃ (коефіцієнт при t):  {coeffs[2]:.10f}")
    
    print("\nЧастоти та їх коефіцієнти:")
    for i, freq in enumerate(frequencies):
        print(f"  f{i+1} = {freq:.10f} Гц, a{i+4} = {coeffs[3+i]:.10f}")
    
    print(f"\n  aₖ₊₁ (константа): {coeffs[-1]:.10f}")
    
    y_model = A @ coeffs
    mse = np.mean((y_observed - y_model)**2)
    rmse = np.sqrt(mse)
    print(f"\nПохибка апроксимації:")
    print(f"  MSE (середньоквадратична похибка): {mse:.10f}")
    print(f"  RMSE (середньоквадратичне відхилення): {rmse:.10f}")
    
    print("\n" + "=" * 70)
    print("Модель функції:")
    print("y(t) = a₁t³ + a₂t² + a₃t + Σ aᵢ sin(2πfᵢ₋₃t) + aₖ₊₁")
    print("=" * 70)

if __name__ == "__main__":
    main()

