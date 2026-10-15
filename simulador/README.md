# Simulador interactivo (charla)

Página web autocontenida para que el público pruebe, desde el celular, los mismos
controladores y agentes del repositorio: PID, LQR, MPC, fuzzy, Q-learning/SARSA
entrenados en vivo, y las tablas y redes guardadas en `results/`. También permite
equilibrar el péndulo a mano.

| Archivo | Qué hace |
|---|---|
| `core.js` | Dinámica de CartPole-v1 y controladores, traducidos de `src/cartpole_lab`. El MPC resuelve el mismo QP que `LinearMPC` con un punto interior de Mehrotra (sin CVXPY). |
| `app.html` | Interfaz (plantilla). |
| `build.py` | Incrusta `core.js` y los parámetros de `results/` en un único HTML. |
| `verify_core.py` | Contrasta `core.js` (ejecutado con Node) con el código Python. |

```bash
python simulador/verify_core.py   # requiere Node
python simulador/build.py         # escribe simulador/dist/simulador.html
```

Resultado de `verify_core.py` (la máxima discrepancia frente a Python):

- Paso de Euler: 2·10⁻¹⁶.
- PID y fuzzy: idénticos.
- K del LQR: 8·10⁻¹¹ relativo.
- Primera fuerza del MPC frente a CVXPY/Clarabel: 1·10⁻⁴ N.
- Acciones tabulares: 0 discrepancias.
- Redes: 4·10⁻¹⁴.
