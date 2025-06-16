import numpy as np
from scipy.linalg import hessenberg, qr

np.random.seed(42)

def print_header(title):
    print("\n" + "=" * 50)
    print(f"=== {title} ===")
    print("=" * 50)

# === 0. Построение рабочей матрицы A ===

n = 5
Lambda = np.diag(np.sort(np.random.uniform(-10, 10, size = n))[::-1])  # Сортируем по убыванию
C = np.random.randn(n, n)
while np.linalg.cond(C) > 1e10:
    C = np.random.randn(n, n)
C_inv = np.linalg.inv(C)
A = C_inv @ Lambda @ C

print_header("Исходные данные")
print("Истинные собственные значения (по убыванию):")
for i, val in enumerate(np.diag(Lambda), start = 1):
    print(f"λ{i}: {val:.8f}")

# === 1. Степенной метод ===

print(A)

def power_method(A, max_iter = 1000, tol = 1e-10):
    n = A.shape[0]
    y = np.random.rand(n)
    z = y / np.linalg.norm(y)
    lambda_old = 0
    iterations = 0
    
    for i in range(max_iter):
        y = A @ z
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-12:
            raise ValueError(
                "Норма вектора слишком мала — метод может не сходиться."
            )
        z = y / norm_y
        lambda_new = np.dot(z, A @ z)
        
        if np.abs(lambda_new - lambda_old) < tol:
            iterations = i + 1
            break
        
        lambda_old = lambda_new
        
    return lambda_new, z, iterations

# === 2а. Обратный степенной метод — все (фиксированные сдвиги) ===

def inverse_power_method_all_fixed_shifts(
    A,
    shift_guesses,
    max_iter = 1000,
    tol = 1e-10
):
    n = A.shape[0]
    eigenvalues = []
    eigenvectors = []
    iterations_used = []
    
    for sigma in shift_guesses:
        sigma = sigma + np.random.uniform(-0.5, 0.5)
        y = np.random.rand(n)
        z = y / np.linalg.norm(y)
        lambda_old = 0
        
        for i in range(max_iter):
            try:
                y = np.linalg.solve(A - sigma * np.eye(n), z)
            except np.linalg.LinAlgError:
                break
                
            norm_y = np.linalg.norm(y)
            if norm_y < 1e-12:
                break
                
            z = y / norm_y
            lambda_new = np.dot(z.T, A @ z)
            
            if np.abs(lambda_new - lambda_old) < tol:
                iterations_used.append(i + 1)
                break
                
            lambda_old = lambda_new
        
        if all(np.abs(lambda_new - l) > tol for l in eigenvalues):
            eigenvalues.append(lambda_new)
            eigenvectors.append(z)
    
    return np.array(eigenvalues), np.array(eigenvectors), iterations_used

# === 2б. Обратный степенной метод — все (переменные сдвиги) ===

def inverse_power_method_all_variable_shifts(
    A,
    shift_guesses,
    max_iter = 1000,
    tol = 1e-10
):
    n = A.shape[0]
    eigenvalues = []
    eigenvectors = []
    iterations_used = []
    
    for sigma0 in shift_guesses:
        sigma = sigma0 + np.random.uniform(-0.5, 0.5)
        y = np.random.rand(n)
        z = y / np.linalg.norm(y)
        
        for i in range(max_iter):
            try:
                y = np.linalg.solve(A - sigma * np.eye(n), z)
            except np.linalg.LinAlgError:
                break
                
            norm_y = np.linalg.norm(y)
            if norm_y < 1e-12:
                break
                
            z = y / norm_y
            sigma_new = np.dot(z.T, A @ z)
            
            if np.abs(sigma_new - sigma) < tol:
                iterations_used.append(i + 1)
                break
                
            sigma = sigma_new
        
        if all(np.abs(sigma - l) > tol for l in eigenvalues):
            eigenvalues.append(sigma)
            eigenvectors.append(z)
    
    return np.array(eigenvalues), np.array(eigenvectors), iterations_used

# === 3. QR-алгоритм со сдвигами ===

def qr_algorithm_with_shift(A, tol = 1e-10, max_iter = 1000):
    A_hess = hessenberg(A)
    n = A.shape[0]
    Ak = A_hess.copy()
    eigenvalues = []
    total_iterations = 0
    
    while n > 1:
        for i in range(max_iter):
            shift = Ak[n - 1, n - 1]
            Q, R = qr(Ak[:n, :n] - shift * np.eye(n))
            Ak[:n, :n] = R @ Q + shift * np.eye(n)
            
            if np.abs(Ak[n - 1, n - 2]) < tol * (
                np.abs(Ak[n - 1, n - 1]) + np.abs(Ak[n - 2, n - 2])
            ):
                total_iterations += i + 1
                break
        
        eigenvalues.append(Ak[n - 1, n - 1])
        n -= 1
        
    eigenvalues.append(Ak[0, 0])
    return np.sort(eigenvalues)[::-1], total_iterations

# === 4. QR без сдвигов + 2x2 блок ===

def qr_algorithm_block_detection(A, tol = 1e-10, max_iter = 1000):
    A_hess = hessenberg(A)
    Ak = A_hess.copy()
    iterations = 0
    
    for i in range(max_iter):
        Q, R = qr(Ak)
        Ak = R @ Q
        iterations += 1
        
        subdiag = np.abs(np.diag(Ak, k = -1))
        if np.all(subdiag[:-1] < tol):
            break
            
    return Ak, iterations

# === Преобразование Λ для 4-го задания ===

eigvals_mod = np.diag(Lambda).copy()
if eigvals_mod[-2] * eigvals_mod[-1] > 0:
    eigvals_mod[-1] *= -1
eigvals_mod[-1], eigvals_mod[-2] = eigvals_mod[-2], eigvals_mod[-1]
Lambda_modified = np.diag(eigvals_mod)
A_modified = np.linalg.inv(C) @ Lambda_modified @ C

# === Запуск всех методов ===
lambda_power, vec_power, iter_power = power_method(A)
shift_guesses = np.diag(Lambda)

eigvals_fixed, eigvecs_fixed, iters_fixed = (
    inverse_power_method_all_fixed_shifts(A, shift_guesses)
)
eigvals_var, eigvecs_var, iters_var = (
    inverse_power_method_all_variable_shifts(A, shift_guesses)
)

qr_eigvals, qr_iters = qr_algorithm_with_shift(A)
Ak_block, block_iters = qr_algorithm_block_detection(A_modified)

# === Вывод результатов ===
print_header("1. Степенной метод")
print(f"Найденное наибольшее по модулю собственное значение: {lambda_power:.8f}")
print(f"Истинное значение λ1: {np.diag(Lambda)[0]:.8f}")
print("Собственный вектор:\n{np.round(vec_power, 6)}")
print(f"Итераций: {iter_power}")

print_header("2а. Обратный степенной метод (фиксированные сдвиги)")
for i, (val, vec, it) in enumerate(zip(eigvals_fixed, eigvecs_fixed, iters_fixed), start = 1):
    print(f"λ{i}: найденное {val:.8f}, истинное {np.diag(Lambda)[i - 1]:.8f}")
    print(f"Собственный вектор:\n{np.round(vec, 6)}")
    print(f"Итераций: {it}")

print_header("2б. Обратный степенной метод (переменные сдвиги)")
for i, (val, vec, it) in enumerate(zip(eigvals_var, eigvecs_var, iters_var), start = 1):
    print(f"λ{i}: найденное {val:.8f}, истинное {np.diag(Lambda)[i - 1]:.8f}")
    print(f"Собственный вектор:\n{np.round(vec, 6)}")
    print(f"Итераций: {it}")

print_header("3. QR-алгоритм со сдвигами")
for i, (found, true) in enumerate(zip(qr_eigvals, np.diag(Lambda)), start = 1):
    print(f"λ{i}: найденное {found:.8f}, истинное {true:.8f}")
print(f"Всего итераций: {qr_iters}")

print_header("4. QR без сдвигов (2x2 блок)")
print("Итоговая матрица после итераций:")
print(np.round(Ak_block, 6))
block_2x2 = Ak_block[-2:, -2:]
print("\nБлок 2×2:")
print(block_2x2)
print("Его собственные значения:")
print(np.linalg.eigvals(block_2x2))
print("Истинные значения λ4 и λ5:")
print(np.diag(Lambda_modified)[-1:-4:-2])
