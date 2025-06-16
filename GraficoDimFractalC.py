import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Parâmetros fixos ---
a = 0.2
b = 0.2
initial_state = [0.0, -6.78, 0.02]
t_span = (0, 500)
t_eval = np.linspace(t_span[0], t_span[1], 100000)

# --- Função do sistema Rössler ---
def rossler(t, state, a, b, c):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

# --- Função Box-Counting ---
def box_count(points, eps):
    points = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0))
    grids = (points / eps).astype(int)
    return len(set(map(tuple, grids)))

# --- Faixa de valores de c ---
c_values = ([2.3, 3.5, 4.1, 4.5, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.7, 5.9, 6.2, 6.7, 6.9, 7.6, 8.3, 9.0, 9.5, 10])
df_values = []

# --- Loop para calcular Df para cada c ---
for c in c_values:
    # Resolver o sistema
    sol = solve_ivp(lambda t, y: rossler(t, y, a, b, c), t_span, initial_state, t_eval=t_eval)
    x, y, z = sol.y

    # Remover transiente (burn-in)
    burn = int(0.2 * len(x))
    x, y, z = x[burn:], y[burn:], z[burn:]

    # Dados 3D
    data = np.vstack((x, y, z)).T

    # Box-counting
    epsilons = np.logspace(-2, 0, 20)
    counts = np.array([box_count(data, eps) for eps in epsilons])
    log_eps = np.log(1/epsilons)
    log_N = np.log(counts)
    slope, _ = np.polyfit(log_eps, log_N, 1)

    # Armazenar dimensão estimada
    df_values.append(slope)
    print(f"c = {c:.2f} → D_f = {slope:.4f}")

# --- Plotar gráfico Df(c) ---
plt.figure(figsize=(8, 6))
plt.plot(c_values, df_values, 'o-', color='darkblue')
plt.xlabel('Parâmetro c')
plt.ylabel('Dimensão Fractal (D_f)')
plt.title('Variação da Dimensão Fractal em função de c')
plt.grid(True)
plt.tight_layout()
plt.show()
