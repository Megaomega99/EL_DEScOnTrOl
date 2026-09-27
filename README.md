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
results/weights/     pesos exportados (JSON/ONNX) para inferencia en el navegador
results/metrics/     métricas de la comparación (Fase 3)
docs/                decisiones de diseño documentadas
```

## Documentación

- [docs/01_entorno_cartpole.md](docs/01_entorno_cartpole.md): versión, constantes, ecuaciones, discretización, convenciones de signo y decisiones del wrapper.

## Hoja de ruta

- [x] 1. Estructura del repo y dependencias fijadas
- [x] 2. Interfaz común del entorno (`CartPoleTask`)
- [ ] 3. PID · 4. LQR · 5. MPC · 6. Fuzzy
- [ ] 7-8. Discretización + Q-learning / SARSA · 9. DQN · 10. Actor-crítico · 11. Exportación de pesos
- [ ] 12-13. Evaluación comparativa y figuras
- [ ] 14. Notebook
