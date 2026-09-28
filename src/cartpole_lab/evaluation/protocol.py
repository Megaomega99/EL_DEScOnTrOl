"""Constantes del protocolo de evaluación (docs/12_protocolo_evaluacion.md). Fijadas ANTES de ver resultados."""

from __future__ import annotations

import numpy as np

from cartpole_lab.sanity import SANITY_SEEDS, THETA_TOLERANCE_RAD, X_TOLERANCE_M

NOMINAL_SEEDS = SANITY_SEEDS  # 1000–1049: distribución oficial de CartPole-v1
THETA_BAND_RAD = THETA_TOLERANCE_RAD  # 0.5°
X_BAND_M = X_TOLERANCE_M  # 5 cm

IMPULSE_STEP = 250  # t = 5 s: mitad del episodio
# Enmienda (docs/12 §8): el barrido inicial acababa en 1.5 N·s, pero PID/LQR/MPC sobreviven a 1.5.
# Se amplía hasta 2.0 N·s, donde LQR y PID fallan en 10/10 CI: el límite real está entre 1.5 y 2.0.
IMPULSES_NS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)

GRID_N = 21
POLE_GRID_AXES = (np.linspace(-0.2, 0.2, GRID_N), np.linspace(-2.0, 2.0, GRID_N))  # θ0 [rad], θ̇0 [rad/s]
CART_GRID_AXES = (np.linspace(-2.3, 2.3, GRID_N), np.linspace(-3.0, 3.0, GRID_N))  # x0 [m], ẋ0 [m/s]

MAIN_METHODS = ("PID", "LQR", "MPC", "Fuzzy", "Q-learning", "SARSA", "DQN", "Actor-crítico")
SUPPLEMENTARY_METHODS = ("Double DQN", "Actor-crítico (red final)")
RL_SEEDS = tuple(range(10))


def impulse_sign(ic_index: int) -> float:
    """Signo alternado: + en las CI pares, − en las impares (políticas aprendidas pueden ser asimétricas)."""
    return 1.0 if ic_index % 2 == 0 else -1.0


def pole_grid_states() -> list[tuple[float, float, float, float]]:
    thetas, theta_dots = POLE_GRID_AXES
    return [(0.0, 0.0, float(t), float(w)) for t in thetas for w in theta_dots]


def cart_grid_states() -> list[tuple[float, float, float, float]]:
    xs, x_dots = CART_GRID_AXES
    return [(float(x), float(v), 0.0, 0.0) for x in xs for v in x_dots]
