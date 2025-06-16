import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parâmetros do sistema de Rössler
a = 0.2
b = 0.2
c = 5.7

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
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 100000)

# Resolver o sistema
solution = solve_ivp(rossler, t_span, initial_state, t_eval=t_eval) # Chama a função solve_ivp() do SciPy, que resolve numericamente equações diferenciais.
solution_2 = solve_ivp(rossler, t_span, initial_state_2, t_eval=t_eval)


# Obter os valores de x, y, z e t para as condições sem e com perturbação
t = solution.t
x, y, z = solution.y
x2, y2, z2 = solution_2.y


# --- Gráficos de x(t), y(t), z(t) ---

def plot_variavel_temporal(t, var1, var2, nome_variavel, cor1, titulo, yticks_range):
    plt.figure(figsize=(8, 4))
    plt.plot(t, var1, color=cor1, label=f'{nome_variavel}')
    plt.plot(t, var2, color='black', linestyle='dotted', label=f'{nome_variavel}: sistema perturbado')
    plt.grid(True)
    plt.xlabel('t')
    plt.ylabel(f'{nome_variavel}(t)')
    plt.title(f'Variável {nome_variavel}: {titulo}')
    plt.xticks(range(0, 100, 10))
    plt.yticks(yticks_range)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.15), ncol=2)
    # plt.legend()
    plt.tight_layout()
    plt.show()

# Gráfico x(t)
plot_variavel_temporal(t, x, x2, 'x', 'blue', 'comportamento temporal', range(-10, 15, 2))

# Gráfico y(t)
plot_variavel_temporal(t, y, y2, 'y', 'green', 'comportamento temporal', range(-10, 10, 2))

# Gráfico z(t)
plot_variavel_temporal(t, z, z2, 'z', 'lightsalmon', 'comportamento temporal', np.arange(-2, 27, 2))