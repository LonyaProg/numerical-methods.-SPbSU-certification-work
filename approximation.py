import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Границы отрезка для аппроксимации
x_min, x_max = 0.1, 1

# Равноотстоящие узлы для МНК
x_ab = np.linspace(x_min, x_max, 5)

# Новая функция
def f(x):
    return np.log10(x) + 7/(2*x + 6)

# Построение матрицы Q для полинома степени 3 по МНК
def Q_matrix(x_v):
    return np.array([[1, xi, xi**2, xi**3] for xi in x_v])

Q = Q_matrix(x_ab)
Q_T = Q.T

H = Q_T @ Q
b = Q_T @ f(x_ab)

# Решаем H a = b
a = np.linalg.solve(H, b)
print(f"Полином по МНК : {a[0]:.6f} + {a[1]:.6f} x + {a[2]:.6f} x^2 + {a[3]:.6f} x^3")

# Функция-аппроксимация по МНК
def f1(x_v):
    return np.array([a[0] + a[1]*xi + a[2]*xi**2 + a[3]*xi**3 for xi in x_v])

# Ортогональные базисные полиномы Лежандра степени 0–3 на [-1,1]
# (Интегрирование всё ещё идёт по [x_min, x_max] напрямую)
def Q0(x): return 1
def Q1(x): return x
def Q2(x): return (3*x**2 - 1)/2
def Q3(x): return (5*x**3 - 3*x)/2

Q_list = [Q0, Q1, Q2, Q3]

# Вычисление коэффициентов раскладки в пространстве L2([x_min,x_max])
def ck(f, Q_list, a, b):
    coeff = []
    for Qk in Q_list:
        num, _ = integrate.quad(lambda x: f(x)*Qk(x), a, b)
        den, _ = integrate.quad(lambda x: Qk(x)**2, a, b)
        coeff.append(num/den)
    return coeff

c = ck(f, Q_list, x_min, x_max)
print(f"\nПолином в пространстве L2: {c[0]:.6f} + {c[1]:.6f} x + {c[2]:.6f} x^2 + {c[3]:.6f} x^3")

# Функция-аппроксимация по Лежандру
def f2(x_v):
    return np.array([c[0] + c[1]*xi + c[2]*xi**2 + c[3]*xi**3 for xi in x_v])

# Более плотная сетка для построения графиков
x_values = np.linspace(x_min, x_max, 200)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x_values, f1(x_values))
plt.title('Приближение по МНК')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x_values, f2(x_values))
plt.title('Приближение по полиномам Лежандра')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x_values, f(x_values))
plt.title(r'$f(x) = \log_{10}(x) + \frac{7}{2x+6}$')
plt.grid(True)

plt.tight_layout()
plt.show()
