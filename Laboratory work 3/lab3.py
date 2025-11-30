import numpy as np

T0 = 0.0
TK = 50.0
DT = 0.2

C4 = 0.12
M1 = 12.0
M2 = 28.0
M3 = 18.0

BETA_0 = np.array([0.1, 0.1, 0.4])


def read_observed_data(filename):
    """
    Читає експериментальні дані з файлу.
    Файл містить 6 рядків, кожен з яких містить значення однієї координати стану.
    Повертає матрицю розміру (6, N), де N - кількість точок часу.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        if line.strip():
            values = [float(x) for x in line.split()]
            data.append(values)
    
    # Перевірка, що всі рядки мають однакову довжину
    if len(set(len(row) for row in data)) > 1:
        raise ValueError("Рядки в файлі мають різну довжину")
    
    return np.array(data)


def build_matrix_A(c1, c2, c3, c4, m1, m2, m3):
    """
    Побудова матриці A системи dy/dt = Ay.
    Матриця має розмір 6×6.
    """
    A = np.zeros((6, 6))
    
    A[0, 1] = 1.0
    
    A[1, 0] = -(c2 + c1) / m1
    A[1, 2] = c2 / m1
    
    A[2, 3] = 1.0
    
    A[3, 0] = c2 / m2
    A[3, 2] = -(c2 + c3) / m2
    A[3, 4] = c3 / m2
    
    A[4, 5] = 1.0
    
    A[5, 2] = c3 / m3
    A[5, 4] = -(c4 + c3) / m3
    
    return A


def compute_dA_dbeta(c1, c2, c3, c4, m1, m2, m3):
    """
    Обчислює ∂(Ay)/∂βᵀ - похідну матриці A по параметрах β = (c₁, c₂, c₃)ᵀ.
    Повертає масив розміру (6, 6, 3), де останній індекс відповідає параметрам.
    Або можна повернути як список з 3 матриць 6×6.
    """
    dA_dc1 = np.zeros((6, 6))
    dA_dc1[1, 0] = -1.0 / m1
    
    dA_dc2 = np.zeros((6, 6))
    dA_dc2[1, 0] = -1.0 / m1
    dA_dc2[1, 2] = 1.0 / m1
    dA_dc2[3, 0] = 1.0 / m2
    dA_dc2[3, 2] = -1.0 / m2
    
    dA_dc3 = np.zeros((6, 6))
    dA_dc3[3, 2] = -1.0 / m2
    dA_dc3[3, 4] = 1.0 / m2
    dA_dc3[5, 2] = 1.0 / m3
    dA_dc3[5, 4] = -1.0 / m3
    
    return [dA_dc1, dA_dc2, dA_dc3]


def state_derivative(y, t, beta, c4, m1, m2, m3):
    c1, c2, c3 = beta
    A = build_matrix_A(c1, c2, c3, c4, m1, m2, m3)
    return A @ y


def sensitivity_derivative(U, y, t, beta, c4, m1, m2, m3):
    """
    Обчислює похідну матриці чутливості:
    dU/dt = (∂(Ay)/∂yᵀ)U + (∂(Ay)/∂βᵀ)
    де (∂(Ay)/∂yᵀ) = A
    """
    c1, c2, c3 = beta
    A = build_matrix_A(c1, c2, c3, c4, m1, m2, m3)
    dA_dbeta = compute_dA_dbeta(c1, c2, c3, c4, m1, m2, m3)
    
    dU_dt = A @ U
    
    for i in range(3):
        dU_dt[:, i] += dA_dbeta[i] @ y
    
    return dU_dt


def runge_kutta_4(f, y0, t_span, h, *args):
    """
    Метод Рунге-Кутта 4-го порядку для системи диференціальних рівнянь.
    
    Параметри:
    f: функція похідної f(y, t, *args)
    y0: початкові умови
    t_span: кортеж (t0, tk)
    h: крок інтегрування
    *args: додаткові аргументи для функції f
    
    Повертає:
    t: масив моментів часу
    y: масив розв'язків (форма залежить від y0)
    """
    t0, tk = t_span
    t = np.arange(t0, tk + h/2, h)
    n = len(t)
    
    if y0.ndim == 1:
        m = len(y0)
        y = np.zeros((n, m))
        y[0] = y0.copy()
        
        for i in range(n - 1):
            k1 = h * f(y[i], t[i], *args)
            k2 = h * f(y[i] + 0.5 * k1, t[i] + 0.5 * h, *args)
            k3 = h * f(y[i] + 0.5 * k2, t[i] + 0.5 * h, *args)
            k4 = h * f(y[i] + k3, t[i] + h, *args)
            y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    else:
        m, p = y0.shape
        y = np.zeros((n, m, p))
        y[0] = y0.copy()
        
        for i in range(n - 1):
            k1 = h * f(y[i], y[i], t[i], *args)
            k2 = h * f(y[i] + 0.5 * k1, y[i], t[i] + 0.5 * h, *args)
            k3 = h * f(y[i] + 0.5 * k2, y[i], t[i] + 0.5 * h, *args)
            k4 = h * f(y[i] + k3, y[i], t[i] + h, *args)
            y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    return t, y


def integrate_state_system(beta, y0, t_span, h, c4, m1, m2, m3):
    def f(y, t, beta, c4, m1, m2, m3):
        return state_derivative(y, t, beta, c4, m1, m2, m3)
    
    t, y = runge_kutta_4(f, y0, t_span, h, beta, c4, m1, m2, m3)
    return t, y


def integrate_sensitivity_system(beta, y_trajectory, t_span, h, c4, m1, m2, m3):
    """
    Інтегрує систему матриць чутливості dU/dt = AU + ∂(Ay)/∂βᵀ
    з початковими умовами U(t₀) = 0.
    
    Параметри:
    y_trajectory: траєкторія стану системи (n, 6)
    """
    n = len(y_trajectory)
    U0 = np.zeros((6, 3))
    
    t0, tk = t_span
    t = np.arange(t0, tk + h/2, h)
    
    U = np.zeros((n, 6, 3))
    U[0] = U0.copy()
    
    for i in range(n - 1):
        y_current = y_trajectory[i]
        
        k1 = h * sensitivity_derivative(U[i], y_current, t[i], beta, c4, m1, m2, m3)
        
        y_mid = 0.5 * (y_trajectory[i] + y_trajectory[i+1])
        k2 = h * sensitivity_derivative(U[i] + 0.5 * k1, y_mid, t[i] + 0.5 * h, beta, c4, m1, m2, m3)
        k3 = h * sensitivity_derivative(U[i] + 0.5 * k2, y_mid, t[i] + 0.5 * h, beta, c4, m1, m2, m3)
        
        y_next = y_trajectory[i+1]
        k4 = h * sensitivity_derivative(U[i] + k3, y_next, t[i] + h, beta, c4, m1, m2, m3)
        
        U[i + 1] = U[i] + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    return t, U


def compute_delta_beta(U_trajectory, y_observed, y_model, t, h):
    """
    Обчислює Δβ за формулою:
    Δβ = (∫_{t₀}^{t_k} Uᵀ(t)U(t)dt)⁻¹ ∫_{t₀}^{t_k} Uᵀ(t)(ȳ(t) - y(t))dt
    
    Параметри:
    U_trajectory: масив матриць чутливості (n, 6, 3)
    y_observed: спостережені дані (6, n) або (n, 6)
    y_model: моделі значення (n, 6)
    t: масив моментів часу
    h: крок інтегрування
    """
    n = len(t)
    
    if y_observed.shape[0] == 6 and y_observed.shape[1] == n:
        y_obs = y_observed.T
    else:
        y_obs = y_observed
    
    integral_UTU = np.zeros((3, 3))
    integral_UT_diff = np.zeros(3)
    
    for i in range(n - 1):
        diff = y_obs[i] - y_model[i]
        
        UTU = U_trajectory[i].T @ U_trajectory[i]
        UT_diff = U_trajectory[i].T @ diff
        
        if i == 0:
            weight = 0.5 * h
        elif i == n - 2:
            weight = 0.5 * h
        else:
            weight = h
        
        integral_UTU += weight * UTU
        integral_UT_diff += weight * UT_diff
    
    try:
        delta_beta = np.linalg.solve(integral_UTU, integral_UT_diff)
    except np.linalg.LinAlgError:
        delta_beta = np.linalg.pinv(integral_UTU) @ integral_UT_diff
    
    return delta_beta


def compute_quality_criterion(y_observed, y_model, t, h):
    """
    Обчислює критерій якості:
    I(β) = ∫_{t₀}^{t_k} (ȳ(t) - y(t))ᵀ(ȳ(t) - y(t))dt
    """
    n = len(t)
    
    if y_observed.shape[0] == 6 and y_observed.shape[1] == n:
        y_obs = y_observed.T
    else:
        y_obs = y_observed
    
    integral = 0.0
    
    for i in range(n - 1):
        diff = y_obs[i] - y_model[i]
        squared_norm = np.dot(diff, diff)
        
        if i == 0 or i == n - 2:
            weight = 0.5 * h
        else:
            weight = h
        
        integral += weight * squared_norm
    
    return integral


def optimize_parameters(beta0, y0, y_observed, t_span, h, c4, m1, m2, m3, max_iter=10, tol=1e-6):
    beta = beta0.copy()
    quality_history = []
    
    print(f"\nПочаткові параметри: β = [{beta[0]:.6f}, {beta[1]:.6f}, {beta[2]:.6f}]")
    
    for iteration in range(max_iter):
        t, y_model = integrate_state_system(beta, y0, t_span, h, c4, m1, m2, m3)
        
        t, U_trajectory = integrate_sensitivity_system(beta, y_model, t_span, h, c4, m1, m2, m3)
        
        delta_beta = compute_delta_beta(U_trajectory, y_observed, y_model, t, h)
        
        beta_new = beta + delta_beta
        
        I_old = compute_quality_criterion(y_observed, y_model, t, h)
        quality_history.append(I_old)
        
        if np.linalg.norm(delta_beta) < tol:
            print(f"\nДосягнуто збіжність на ітерації {iteration + 1}")
            break
        
        beta = beta_new
        
        print(f"\nІтерація {iteration + 1}:")
        print(f"  Δβ = [{delta_beta[0]:.6e}, {delta_beta[1]:.6e}, {delta_beta[2]:.6e}]")
        print(f"  β = [{beta[0]:.6f}, {beta[1]:.6f}, {beta[2]:.6f}]")
        print(f"  I(β) = {I_old:.6e}")
    
    t, y_model = integrate_state_system(beta, y0, t_span, h, c4, m1, m2, m3)
    I_final = compute_quality_criterion(y_observed, y_model, t, h)
    
    return beta, I_final, quality_history


def main():
    print("=" * 70)
    print("Лабораторна робота 3: Параметрична ідентифікація параметрів")
    print("за допомогою функцій чутливості")
    print("=" * 70)
    
    print("\n1. Читання експериментальних даних з y10.txt...")
    try:
        y_observed = read_observed_data('y10.txt')
        print(f"   Завантажено дані: {y_observed.shape[0]} координат, {y_observed.shape[1]} точок часу")
    except FileNotFoundError:
        print("   Помилка: файл y10.txt не знайдено!")
        return
    except Exception as e:
        print(f"   Помилка при читанні файлу: {e}")
        return
    
    n_points = y_observed.shape[1]
    expected_points = int((TK - T0) / DT) + 1
    if n_points != expected_points:
        print(f"   Увага: очікувалось {expected_points} точок, отримано {n_points}")
    
    y0 = np.array([y_observed[i, 0] for i in range(6)])
    
    print("\n2. Параметри задачі:")
    print(f"   Час: t₀ = {T0}, t_k = {TK}, Δt = {DT}")
    print(f"   Відомі параметри: c₄ = {C4}, m₁ = {M1}, m₂ = {M2}, m₃ = {M3}")
    print(f"   Початкове наближення: β₀ = [{BETA_0[0]}, {BETA_0[1]}, {BETA_0[2]}]")
    print(f"   Початкові умови стану: y₀ = {y0}")
    
    print("\n3. Оптимізація параметрів...")
    beta_optimal, I_final, quality_history = optimize_parameters(
        BETA_0, y0, y_observed, (T0, TK), DT, C4, M1, M2, M3,
        max_iter=10, tol=1e-6
    )
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТИ:")
    print("=" * 70)
    print("\nЗнайдені параметри:")
    print(f"  c₁ = {beta_optimal[0]:.10f}")
    print(f"  c₂ = {beta_optimal[1]:.10f}")
    print(f"  c₃ = {beta_optimal[2]:.10f}")
    print(f"\nКритерій якості: I(β) = {I_final:.10e}")
    
    print("\nІсторія зміни критерію якості:")
    for i, I_val in enumerate(quality_history):
        print(f"  Ітерація {i+1}: I(β) = {I_val:.10e}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

