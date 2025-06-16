import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Parâmetros do sistema de Rössler ---
a = 0.2
b = 0.2
c = 3.6

def rossler(t, state):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

# --- Simulação do atrator ---
initial_state = [0.0, -6.78, 0.02]
t_span = (0, 500)
t_eval = np.linspace(t_span[0], t_span[1], 100000)

solution = solve_ivp(rossler, t_span, initial_state, t_eval=t_eval)
x, y, z = solution.y

# --- Remover transiente inicial (burn-in) ---
burn_in = int(0.2 * len(x))  # descarta os primeiros 20%
x = x[burn_in:]
y = y[burn_in:]
z = z[burn_in:]

# --- Função Box-Counting ---
def box_count(points, eps):
    points = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0))
    grids = (points / eps).astype(int)
    return len(set(map(tuple, grids)))

# --- Preparar dados 3D para análise ---
data = np.vstack((x, y, z)).T  # formato (N, 3)

epsilons = np.logspace(-2, 0, 20)  # eps de 0.01 a 1.0
counts = np.array([box_count(data, eps) for eps in epsilons])

# --- Ajuste linear (log-log) ---
log_eps = np.log(1/epsilons)
log_N = np.log(counts)
slope, intercept = np.polyfit(log_eps, log_N, 1)

# --- Plotar gráfico log-log ---
plt.figure(figsize=(8,6))
plt.plot(log_eps, log_N, 'o-', label=f'Dimensão fractal ≈ {slope:.4f}')
plt.xlabel('log(1/ε)')
plt.ylabel('log(N(ε))')
plt.title(f'Estimativa da Dimensão Fractal (Box-Counting 3D). c = {c}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Estimativa da dimensão fractal (com burn-in): {slope:.4f}")
