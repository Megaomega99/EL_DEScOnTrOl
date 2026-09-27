r"""Regulador lineal cuadrático (LQR) discreto sobre el modelo de Euler del simulador.

PROBLEMA
--------
Minimizar, para el sistema lineal s_{k+1} = A s_k + B F_k,

    J = Σ_{k≥0} ( s_kᵀ Q s_k + R F_k² )

La solución es una realimentación de estado F_k = −K s_k, con

    K = (R + BᵀPB)⁻¹ BᵀPA,
    P = AᵀPA − AᵀPB (R + BᵀPB)⁻¹ BᵀPA + Q        (ecuación algebraica de Riccati discreta)

y J(s_0) = s_0ᵀ P s_0 es el coste óptimo desde s_0.

QUÉ MODELO
----------
(A, B) = linealización de Euler, que es exactamente el mapa del simulador
(linearization.py). La discretización ZOH exacta y el LQR de tiempo continuo se
calculan solo para comparar (scripts/design_lqr.py, docs/04_lqr.md).

Q Y R: REGLA DE BRYSON (Bryson & Ho, 1975)
------------------------------------------
Q_ii = 1 / (máximo valor ACEPTABLE de s_i)²,   R = 1 / (máxima fuerza ACEPTABLE)².
Hace adimensional cada término y fija el compromiso con magnitudes físicas en
lugar de números arbitrarios. Solo importa la escala relativa: multiplicar Q y R
por el mismo factor no cambia K. Los máximos aceptables son ESPECIFICACIONES DE
DISEÑO, más estrictas que los límites de fallo:

* θ: 8.1°, el ángulo hasta el que cos θ ≈ 1 con error ≤ 1 %. Es la zona donde el
  modelo lineal con el que se diseña es fiel. Criterio con base física.
* x: 1.2 m, la mitad del semirraíl (2.4 m), para dejar la otra mitad de margen
  ante perturbaciones. SUPUESTO sin base teórica (el "mitad" es una elección).
* F: 10 N, el límite del actuador: el diseño lineal debería saturar poco.
* ẋ, θ̇: peso 0. Gymnasium no las acota y cualquier escala sería inventada. El
  LQR las usa igualmente en K, porque son necesarias para estabilizar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_continuous_are, solve_discrete_are, solve_discrete_lyapunov

from cartpole_lab.linearization import continuous_linearization, euler_discretization
from cartpole_lab.params import CartPoleParams
from cartpole_lab.rollout import Trajectory

BRYSON_TRACK_FRACTION = 0.5  # x aceptable = 50 % del semirraíl (SUPUESTO)
BRYSON_COS_TOLERANCE = 0.01  # θ aceptable: cos θ ≥ 0.99 (validez del modelo lineal)


def _frozen(array: ArrayLike) -> np.ndarray:
    out = np.array(array, dtype=np.float64)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class LQRDesign:
    A: np.ndarray  # (4, 4) modelo discreto
    B: np.ndarray  # (4,)
    Q: np.ndarray  # (4, 4)
    R: np.ndarray  # (1, 1)
    K: np.ndarray  # (1, 4)  F = −K s
    P: np.ndarray  # (4, 4)  coste óptimo s0ᵀ P s0

    @property
    def closed_loop_eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvals(self.A - self.B[:, None] @ self.K)

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(self.closed_loop_eigenvalues)))

    def dominant_time_constant(self, tau: float) -> float:
        """−τ / ln ρ [s]: tiempo en que el modo más lento se reduce en un factor e."""
        return float(-tau / np.log(self.spectral_radius))


def bryson_weights(
    params: CartPoleParams,
    *,
    track_fraction: float = BRYSON_TRACK_FRACTION,
    cos_tolerance: float = BRYSON_COS_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    max_x = track_fraction * params.x_threshold
    max_theta = float(np.arccos(1.0 - cos_tolerance))
    Q = np.diag([1.0 / max_x**2, 0.0, 1.0 / max_theta**2, 0.0])
    R = np.array([[1.0 / params.force_mag**2]])
    return _frozen(Q), _frozen(R)


def _validate_weights(Q: np.ndarray, R: np.ndarray) -> None:
    if Q.shape != (4, 4) or R.shape != (1, 1):
        raise ValueError(f"Q debe ser (4, 4) y R (1, 1); se recibieron {Q.shape} y {R.shape}")
    if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(R))):
        raise ValueError("Q y R deben ser finitas")
    if not np.allclose(Q, Q.T, rtol=0, atol=1e-12):
        raise ValueError("Q debe ser simétrica")
    if np.linalg.eigvalsh(Q).min() < -1e-12:
        raise ValueError("Q debe ser semidefinida positiva")
    if R[0, 0] <= 0:
        raise ValueError("R debe ser definida positiva")


def solve_dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> LQRDesign:
    Q, R = np.asarray(Q, dtype=np.float64), np.asarray(R, dtype=np.float64)
    _validate_weights(Q, R)
    B_col = np.asarray(B, dtype=np.float64).reshape(-1, 1)
    P = solve_discrete_are(A, B_col, Q, R)
    K = np.linalg.solve(R + B_col.T @ P @ B_col, B_col.T @ P @ A)
    return LQRDesign(A=_frozen(A), B=_frozen(B_col[:, 0]), Q=_frozen(Q), R=_frozen(R), K=_frozen(K), P=_frozen(P))


def solve_clqr(A_c: np.ndarray, B_c: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """K del LQR de tiempo CONTINUO (minimiza ∫ sᵀQs + R F² dt). Solo para comparar.

    Se usan los MISMOS Q, R que en el diseño discreto, sin escalar por τ. La suma
    discreta que aproxima la integral sería Σ (sᵀ(τQ)s + (τR)F²), pero multiplicar Q
    y R por el mismo factor no cambia la K discreta. La comparación es justa sin
    reescalar.
    """
    Q, R = np.asarray(Q, dtype=np.float64), np.asarray(R, dtype=np.float64)
    _validate_weights(Q, R)
    B_col = np.asarray(B_c, dtype=np.float64).reshape(-1, 1)
    P = solve_continuous_are(A_c, B_col, Q, R)
    return np.linalg.solve(R, B_col.T @ P)


def cost_matrix_of_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, K: ArrayLike) -> np.ndarray:
    """P_K tal que J(s0) = s0ᵀ P_K s0 con F = −K s (ecuación de Lyapunov discreta).

    Sirve para medir cuánto peor que el óptimo es una ganancia cualquiera (otra
    discretización, el PID...) sobre la planta real.
    """
    K = np.asarray(K, dtype=np.float64).reshape(1, -1)
    closed_loop = A - np.asarray(B, dtype=np.float64).reshape(-1, 1) @ K
    if np.max(np.abs(np.linalg.eigvals(closed_loop))) >= 1.0:
        raise ValueError("La ganancia no estabiliza el sistema: su coste infinito no está definido")
    # SciPy resuelve a·X·aᵀ − X + q = 0. Con a = A_clᵀ queda X = A_clᵀ·X·A_cl + q, que es
    # la ecuación de Bellman del coste de la política fija F = −K s (fácil de invertir por error).
    return _frozen(solve_discrete_lyapunov(closed_loop.T, Q + K.T @ R @ K))


def lqr_criterion(traj: Trajectory, Q: np.ndarray, R: np.ndarray) -> float:
    """Σ_k (s_kᵀ Q s_k + R F_k²) sobre una trayectoria REAL (no lineal, con saturación).

    Es el criterio del LQR evaluado en la simulación, útil para comparar métodos con
    él. Un episodio fallido no tiene coste infinito definido: se rechaza.
    """
    if traj.terminated:
        raise ValueError(f"El episodio falló ({traj.termination_reason}); el criterio LQR no está definido")
    states = traj.states[:-1]  # s_0 .. s_{T-1}, emparejados con F_0 .. F_{T-1}
    state_cost = np.einsum("ki,ij,kj->k", states, Q, states)
    return float(np.sum(state_cost + R[0, 0] * np.square(traj.forces)))


def design_default_lqr(params: CartPoleParams) -> LQRDesign:
    """El LQR del proyecto: modelo de Euler + pesos de Bryson."""
    A_d, B_d = euler_discretization(*continuous_linearization(params), params.tau)
    return solve_dlqr(A_d, B_d, *bryson_weights(params))


class LQRController:
    """F = sat(−K s). Sin estado interno. La saturación no forma parte del diseño LQR:
    lejos del equilibrio el controlador deja de ser óptimo (y puede dejar de estabilizar)."""

    action_mode = "continuous"

    def __init__(self, K: ArrayLike, params: CartPoleParams) -> None:
        gain = np.asarray(K, dtype=np.float64)
        if gain.shape not in ((4,), (1, 4)) or not np.all(np.isfinite(gain)):
            raise ValueError(f"K debe ser finita y de forma (1, 4) o (4,), no {gain.shape}")
        self.K = _frozen(gain.reshape(1, 4))
        self._force_limit = params.force_mag

    def reset(self) -> None:
        pass

    def __call__(self, state: ArrayLike) -> float:
        s = np.asarray(state, dtype=np.float64)
        if s.shape != (4,):
            raise ValueError(f"El estado debe tener forma (4,), no {s.shape}")
        force = -(self.K @ s).item()
        return min(max(force, -self._force_limit), self._force_limit)
