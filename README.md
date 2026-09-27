# EL_DEScOnTrOl
Super aprendizaje básico de control.

Material docente para una charla de laboratorio (neuroingeniería/DBS): el
péndulo invertido sobre carro (**CartPole-v1** de Gymnasium) como hilo conductor
para comparar **control clásico** (PID, LQR, MPC, fuzzy) y **aprendizaje por
refuerzo** (Q-learning, SARSA, DQN, actor-crítico) con la *misma* física, el
*mismo* estado, la *misma* señal de recompensa/coste y el *mismo* criterio de
terminación.

## Instalación

Pensado para el entorno conda `base` (Python 3.13):

```bash
python -m pip install -r requirements.txt   # versiones exactas (==)
python -m pip install -e . --no-deps        # el paquete cartpole_lab, editable
python -m pytest -q                         # debe pasar todo
python scripts/verify_environment.py        # constantes y convenciones de signo
```

## Uso mínimo

```python
from cartpole_lab import CartPoleTask, run_episode

class ZeroForce:                      # cualquier controlador: action_mode + reset() + __call__(s)
    action_mode = "continuous"
    def reset(self): pass
    def __call__(self, state): return 0.0

task = CartPoleTask(action_mode="continuous")
traj = run_episode(task, ZeroForce(), initial_state=[0.0, 0.0, 0.01, 0.0])
print(traj.termination_reason, traj.n_steps)   # pole_angle_limit 49 -> sin control, cae en 0.98 s
```

## Estructura

```
src/cartpole_lab/
    params.py        constantes físicas LEÍDAS de Gymnasium (nunca copiadas a mano)
    dynamics.py      modelo explícito: EDO continua + paso de Euler idéntico al simulador
    env.py           CartPoleTask: la interfaz común (actuador discreto/continuo, perturbaciones, métricas)
    cost.py          coste de etapa común para evaluación
    disturbances.py  pulsos de fuerza externos sobre el carro
    rollout.py       bucle de simulación común + registro de trayectorias
    seeding.py       semillas globales (random, NumPy, PyTorch)
    controllers/     PID, LQR, MPC, fuzzy                       (Fase 1, pasos 3-6)
    rl/              tabulares, DQN, actor-crítico              (Fase 2)
tests/               sanity checks y verificación contra Gymnasium
scripts/             scripts ejecutables (verificación, entrenamiento, figuras)
notebooks/           notebook docente único                     (Fase 4)
results/figures/     figuras para las diapositivas
results/tuning/      parámetros sintonizados de los controladores clásicos (JSON con metadatos)
results/weights/     pesos exportados (JSON/ONNX) para inferencia en el navegador
results/rl/          registros de entrenamiento de RL (curvas, semillas, test)
results/metrics/     métricas de la comparación (Fase 3)
docs/                decisiones de diseño documentadas
```

## Documentación

- [docs/01_entorno_cartpole.md](docs/01_entorno_cartpole.md): versión, constantes, ecuaciones, discretización, convenciones de signo y decisiones del wrapper.
- [docs/03_pid.md](docs/03_pid.md): PID en cascada, por qué Ziegler–Nichols no aplica, sintonización por optimización (ITAE) y resultados.
- [docs/04_lqr.md](docs/04_lqr.md): linealización, elección del modelo discreto (Euler frente a ZOH frente a continuo), Q y R por la regla de Bryson, y PID frente a LQR con el criterio de cada uno.
- [docs/05_mpc.md](docs/05_mpc.md): MPC lineal con restricciones, margen de seguridad, horizonte y solver elegidos con datos (Clarabel frente a OSQP).
- [docs/06_fuzzy.md](docs/06_fuzzy.md): fuzzy Takagi-Sugeno de orden 0 (membresías, tabla de reglas, inferencia), por qué equivale a un PD con saturación suave, y comparación con el PID.
- [docs/07_08_tabular.md](docs/07_08_tabular.md): discretización en cajas (BOXES de 1983, verificada en `pole.c`, y estudio de sensibilidad), Q-learning y SARSA con 10 semillas, inestabilidad y calidad del control aprendido.

## Hoja de ruta

- [x] 1. Estructura del repo y dependencias fijadas
- [x] 2. Interfaz común del entorno (`CartPoleTask`)
- [x] 3. PID en cascada (sintonizado por evolución diferencial sobre ITAE)
- [x] 4. LQR (modelo de Euler del simulador, pesos de Bryson)
- [x] 5. MPC lineal (CVXPY + Clarabel, N = 25, restricciones con margen)
- [x] 6. Fuzzy Takagi-Sugeno de orden 0 (tabla MacVicar-Whelan 5×5, mismo protocolo de sintonización que el PID)
- [x] 7-8. Discretización (BOXES 1983) + Q-learning / SARSA tabulares (10 semillas)
- [ ] 9. DQN · 10. Actor-crítico · 11. Exportación de pesos
- [ ] 12-13. Evaluación comparativa y figuras
- [ ] 14. Notebook
