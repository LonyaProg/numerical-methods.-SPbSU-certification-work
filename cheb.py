import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# 1) Задаём равноотстоящие узлы на [1,10]
x_values = np.linspace(1, 10, 5)

# 2) Строим узлы Чебышева на том же отрезке
# … код до этого момента остаётся без изменений …

# 2) Строим узлы Чебышева на том же отрезке, а затем приводим к numpy.array
a, b = x_values[0], x_values[-1]
n = len(x_values) - 1
x_values_cheb = [
    0.5*(b - a)*np.cos((2*i + 1)*np.pi/(2*(n+1))) + 0.5*(b + a)
    for i in range(n+1)
]
x_values_cheb = np.array(x_values_cheb)  # <-- вот эта строка!

# … остальная часть кода без изменений …


# 3) Интерполируемая функция
def f(x):
    return np.log10(x) + 7 / (2*x + 6)

# 4) Построение полинома Лагранжа (символьно)
def lagrange_polynomial(func, nodes):
    x = sp.symbols('x')
    ys = [func(xi) for xi in nodes]
    L = 0
    for i in range(len(nodes)):
        li = 1
        for j in range(len(nodes)):
            if i != j:
                li *= (x - nodes[j])/(nodes[i] - nodes[j])
        L += ys[i]*li
    return sp.simplify(L)

# 5) Преобразование символьного полинома в функцию numpy
def eval_poly(L, xs):
    f_num = sp.lambdify(sp.symbols('x'), L, 'numpy')
    return f_num(xs)

# 6) Считаем полиномы
P_eq = lagrange_polynomial(f, x_values)
P_ch = lagrange_polynomial(f, x_values_cheb)

print("Полином Лагранжа на равноотстоящих узлах:\n", sp.expand(P_eq))
print("\nПолином Лагранжа на узлах Чебышева:\n", sp.expand(P_ch))

# 7) Графики
x_plot = np.linspace(-1, 10, 200)

y_true = f(x_plot)
y_eq   = eval_poly(P_eq, x_plot)
y_ch   = eval_poly(P_ch, x_plot)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(x_plot, y_true, 'k-', label='f(x)')
plt.plot(x_plot, y_eq, 'r--', label='Lagrange (равноотст.)')
plt.scatter(x_values, f(x_values), c='r')
plt.title('Интерполяция на равноотст. узлах')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(x_plot, y_true, 'k-', label='f(x)')
plt.plot(x_plot, y_ch, 'b--', label='Lagrange (Чебышёв)')
plt.scatter(x_values_cheb, f(x_values_cheb), c='b')
plt.title('Интерполяция на узлах Чебышева')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
