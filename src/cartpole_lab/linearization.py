r"""Linealización de CartPole en el equilibrio vertical (s = 0, F = 0).

DERIVACIÓN (a partir de las ecuaciones de dynamics.py, las de Gymnasium)
-----------------------------------------------------------------------
Cerca de θ = 0: sin θ ≈ θ, cos θ ≈ 1, y θ̇² sin θ ≈ 0 (segundo orden). Entonces
temp ≈ F / M con M = m_c + m, y

    θ̈ ≈ (g θ − F/M) / D,        D = l (4/3 − m / M)       (longitud efectiva)
      = a θ − b F,               a = g / D,   b = 1 / (M D)
    ẍ ≈ F/M − (m l / M) θ̈ = −c a θ + (1/M + c b) F,       c = m l / M

En forma de estado, ṡ = A_c s + B_c F:

          ⎡0  1    0    0⎤          ⎡    0     ⎤
    A_c = ⎢0  0  −c·a   0⎥,   B_c = ⎢ 1/M + c·b⎥
          ⎢0  0    0    1⎥          ⎢    0     ⎥
          ⎣0  0    a    0⎦          ⎣   −b     ⎦

Autovalores de A_c: {0, 0, ±√a}. El carro es un doble integrador (dos ceros) y
el poste tiene un modo inestable en +√a ≈ 3.97 s⁻¹, con constante de tiempo de
0.25 s: sin control, un error se multiplica por e cada 0.25 s.

DISCRETIZACIÓN
--------------
* Euler (la que usa el proyecto): A_d = I + τ A_c, B_d = τ B_c. Es la
  linealización EXACTA del mapa s_{k+1} = s_k + τ f(s_k, F_k) que ejecuta
  Gymnasium (test `test_euler_discretization_is_the_jacobian_of_the_simulator_step`).
* ZOH exacta: A_d = e^{A_c τ}, B_d = ∫_0^τ e^{A_c σ} dσ B_c. Describe el péndulo
  físico con fuerza constante durante τ, pero NO el simulador. Difiere de Euler
  en O(τ²).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from cartpole_lab.params import CartPoleParams


def continuous_linearization(params: CartPoleParams) -> tuple[np.ndarray, np.ndarray]:
    """(A_c, B_c) analíticas; B_c de forma (4,)."""
    p = params
    total_mass = p.total_mass
    effective_length = p.length * (4.0 / 3.0 - p.masspole / total_mass)  # D
    a = p.gravity / effective_length
    b = 1.0 / (total_mass * effective_length)
    c = p.polemass_length / total_mass
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -c * a, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, a, 0.0],
        ]
    )
    B = np.array([0.0, 1.0 / total_mass + c * b, 0.0, -b])
    return A, B


def euler_discretization(A_c: np.ndarray, B_c: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Discretización de Euler explícito: la que corresponde al simulador de Gymnasium."""
    return np.eye(A_c.shape[0]) + tau * A_c, tau * B_c


def zoh_discretization(A_c: np.ndarray, B_c: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Discretización exacta con retenedor de orden cero, por el método de Van Loan.

    exp( [[A, B], [0, 0]] τ ) = [[e^{Aτ}, ∫_0^τ e^{Aσ}dσ B], [0, 1]]
    """
    n = A_c.shape[0]
    augmented = np.zeros((n + 1, n + 1))
    augmented[:n, :n] = A_c
    augmented[:n, n] = B_c
    exponential = expm(augmented * tau)
    return exponential[:n, :n], exponential[:n, n]


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """[B, AB, A²B, A³B]. Rango completo ⇔ el sistema lineal es controlable."""
    columns = [np.asarray(B, dtype=np.float64).reshape(-1)]
    for _ in range(A.shape[0] - 1):
        columns.append(A @ columns[-1])
    return np.stack(columns, axis=1)
