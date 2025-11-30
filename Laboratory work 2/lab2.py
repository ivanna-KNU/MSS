import numpy as np
from PIL import Image

def load_bmp_image(filename):
    """
    Завантажує BMP зображення та конвертує його в numpy масив.
    Нормалізує значення пікселів до діапазону [0, 1].
    
    Args:
        filename: шлях до BMP файлу
        
    Returns:
        numpy масив з нормалізованими значеннями [0, 1]
    """
    img = Image.open(filename)
    img_array = np.array(img, dtype=np.float64)
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    return img_array

def prepare_matrices(x_image, y_image):
    """
    Підготовка матриць X та Y згідно з формулою (1).
    
    X = ([x₁ x₂ ... xₙ] / [1 1 ... 1])
    де кожен стовпець xⱼ - це вхідний сигнал.
    
    Args:
        x_image: numpy масив вхідного зображення (m × n_x)
        y_image: numpy масив вихідного зображення (p × n_y)
        
    Returns:
        X: матриця розміром (m+1) × n, де n = min(n_x, n_y)
        Y: матриця розміром p × n
    """
    m, n_x = x_image.shape
    p, n_y = y_image.shape
    
    # Використовуємо мінімальну кількість стовпців
    n = min(n_x, n_y)
    
    X_data = x_image[:, :n]
    
    # Додаємо рядок одиниць знизу
    ones_row = np.ones((1, n))
    X = np.vstack([X_data, ones_row])
    
    Y = y_image[:, :n]
    
    return X, Y

def compute_pseudoinverse(X):
    """
    Обчислює псевдообернену матрицю X⁺ використовуючи формулу Мура-Пенроуза.
    
    Використовує numpy.linalg.pinv(), який реалізує:
    X⁺ = lim(δ→0) Xᵀ(XXᵀ + δ²E)⁻¹
    
    Args:
        X: матриця розміром (m+1) × n
        
    Returns:
        X_pinv: псевдообернена матриця розміром n × (m+1)
    """
    return np.linalg.pinv(X)

def compute_operator(Y, X_pinv):
    """
    Обчислює матрицю оператора A = YX⁺.
    
    Args:
        Y: матриця вихідних сигналів розміром p × n
        X_pinv: псевдообернена матриця розміром n × (m+1)
        
    Returns:
        A: матриця оператора розміром p × (m+1)
    """
    return Y @ X_pinv

def evaluate_accuracy(A, X, Y):
    """
    Перевіряє точність моделі, обчислюючи Y_pred = AX та метрики похибки.
    
    Args:
        A: матриця оператора розміром p × (m+1)
        X: матриця вхідних сигналів розміром (m+1) × n
        Y: матриця вихідних сигналів розміром p × n
        
    Returns:
        Y_pred: передбачені значення розміром p × n
        mse: середньоквадратична похибка
        rmse: середньоквадратичне відхилення
        max_error: максимальна похибка
        mean_error: середня похибка
    """
    Y_pred = A @ X
    
    error = Y - Y_pred
    
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(error))
    mean_error = np.mean(np.abs(error))
    
    return Y_pred, mse, rmse, max_error, mean_error

def main():
    print("Лабораторна робота 2: Побудова лінійної моделі з допомогою псевдообернених операторів")
    print("=" * 80)
    
    print("\n1. Завантаження зображень...")
    x_image = load_bmp_image('x2.bmp')
    y_image = load_bmp_image('y5.bmp')
    print(f"   x2.bmp: розмір {x_image.shape}, діапазон [{x_image.min():.3f}, {x_image.max():.3f}]")
    print(f"   y5.bmp: розмір {y_image.shape}, діапазон [{y_image.min():.3f}, {y_image.max():.3f}]")
    
    print("\n2. Підготовка матриць X та Y...")
    X, Y = prepare_matrices(x_image, y_image)
    print(f"   Матриця X: розмір {X.shape}")
    print(f"   Матриця Y: розмір {Y.shape}")
    
    print("\n3. Обчислення псевдооберненої матриці X⁺...")
    X_pinv = compute_pseudoinverse(X)
    print(f"   X⁺: розмір {X_pinv.shape}")
    
    print("\n4. Обчислення матриці оператора A = YX⁺...")
    A = compute_operator(Y, X_pinv)
    print(f"   Матриця A: розмір {A.shape}")
    
    print("\n5. Перевірка точності моделі...")
    Y_pred, mse, rmse, max_error, mean_error = evaluate_accuracy(A, X, Y)
    print(f"   MSE (середньоквадратична похибка): {mse:.10f}")
    print(f"   RMSE (середньоквадратичне відхилення): {rmse:.10f}")
    print(f"   Максимальна похибка: {max_error:.10f}")
    print(f"   Середня похибка: {mean_error:.10f}")
    
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТИ:")
    print("=" * 80)
    
    print(f"\nРозміри матриць:")
    print(f"  X (вхідні сигнали): {X.shape}")
    print(f"  Y (вихідні сигнали): {Y.shape}")
    print(f"  X⁺ (псевдообернена): {X_pinv.shape}")
    print(f"  A (оператор): {A.shape}")
    
    print(f"\nСтатистика похибки:")
    print(f"  MSE:  {mse:.10f}")
    print(f"  RMSE: {rmse:.10f}")
    print(f"  Максимальна похибка: {max_error:.10f}")
    print(f"  Середня похибка: {mean_error:.10f}")
    
    print(f"\nІнформація про матрицю оператора A:")
    print(f"  Мінімальне значення: {A.min():.10f}")
    print(f"  Максимальне значення: {A.max():.10f}")
    print(f"  Середнє значення: {A.mean():.10f}")
    print(f"  Стандартне відхилення: {A.std():.10f}")
    
    print("\n" + "=" * 80)
    print("Модель: Y = AX, де A = YX⁺")
    print("=" * 80)

if __name__ == "__main__":
    main()

