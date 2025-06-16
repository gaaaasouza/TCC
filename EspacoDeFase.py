import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parâmetros do sistema de Rössler
a = 0.2
b = 0.2
c = 4.1

# Definindo o sistema de equações diferenciais
def rossler(t, state):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

# Condições iniciais
initial_state = [0.0, -6.78, 0.02]     # sem perturbação em y   
initial_state_2 = [0.0, -6.88, 0.02]   # com perturbação em y

# Intervalo de tempo e número de pontos
t_span = (0, 500)
t_eval = np.linspace(t_span[0], t_span[1], 100000)

# Resolver o sistema
solution = solve_ivp(rossler, t_span, initial_state, t_eval=t_eval) # Chama a função solve_ivp() do SciPy, que resolve numericamente equações diferenciais.
solution_2 = solve_ivp(rossler, t_span, initial_state_2, t_eval=t_eval)


# Obter os valores de x, y, z e t para as condições sem e com perturbação
t = solution.t
x, y, z = solution.y
x2, y2, z2 = solution_2.y

# --- Gráfico 3D do atrator de Rössler ---
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, color='black', lw=0.5)
ax.plot(x2, y2, z2, color='red', lw=0.5)

# Personalizar limites dos eixos
ax.set_xticks(range(-14, 14, 4))   # Limites do eixo x
ax.set_yticks(range(-14, 14, 4))   # Limites do eixo y
ax.set_zticks(range(0, 28, 4))     # Limites do eixo z

ax.set_title(f"Espaço de Fase (c = {c})")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.tight_layout()
plt.show()