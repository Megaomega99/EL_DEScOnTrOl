"""Sintonización de los factores de escala del fuzzy con EXACTAMENTE el protocolo del PID.

Mismo criterio (ITAE + esfuerzo), mismas 32 condiciones iniciales, mismo horizonte, mismo
optimizador y presupuesto (3 semillas × 150 generaciones × 100 candidatos). Solo cambian
los parámetros y sus límites. Se sintonizan los 3 factores de escala (E, D, U) y, como
hizo el PID, las 2 ganancias del lazo externo. Reutilizar las del PID pondría al fuzzy
en desventaja, porque se sintonizaron junto con OTRO lazo interno.
La tabla de reglas y la forma de las membresías NO se sintonizan: son la parte
"explicable" y se escriben desde la física.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from cartpole_lab.controllers.batch import BatchRollout, run_differential_evolution, simulate_batch
from cartpole_lab.controllers.fuzzy import FuzzyGains, fuzzy_force
from cartpole_lab.controllers.pid_tuning import TuningConfig, TuningResult, TuningRun, sample_tuning_initial_states
from cartpole_lab.dynamics import euler_step, finite_difference_jacobian
from cartpole_lab.params import CartPoleParams

FUZZY_GAIN_NAMES = ("theta_scale", "theta_dot_scale", "force_scale", "kp_x", "kd_x")
FUZZY_GAIN_BOUNDS = (
    (0.01, 0.21),  # E: "error grande" entre 0.6° y 12° (el ángulo de fallo)
    (0.05, 3.0),  # D: "velocidad grande" [rad/s]; holgado
    (0.5, 10.0),  # U: "empujar fuerte" hasta el máximo del actuador (10 N)
    (0.0, 0.5),  # kp_x: mismos límites que el PID
    (0.0, 0.5),  # kd_x
)


@dataclass(frozen=True)
class FuzzyTuningConfig(TuningConfig):
    bounds: tuple[tuple[float, float], ...] = FUZZY_GAIN_BOUNDS


def simulate_fuzzy_batch(
    gains: FuzzyGains,
    initial_states: ArrayLike,
    params: CartPoleParams,
    *,
    n_steps: int,
    theta_ref_limit: float,
    record: bool = False,
) -> BatchRollout:
    s0 = np.asarray(initial_states, dtype=np.float64)
    batch_shape = np.broadcast_shapes(np.shape(gains.theta_scale), s0.shape[:-1])

    def law(observation: np.ndarray, memory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        force = fuzzy_force(observation, gains, force_limit=params.force_mag, theta_ref_limit=theta_ref_limit)
        return force, memory  # sin estado interno

    return simulate_batch(law, s0, params, batch_shape=batch_shape, n_steps=n_steps, record=record)


def _objective(
    candidates: np.ndarray, initial_states: np.ndarray, params: CartPoleParams, config: FuzzyTuningConfig
) -> np.ndarray:
    gains = FuzzyGains(*(row[:, None] for row in candidates))
    batch = simulate_fuzzy_batch(
        gains, initial_states, params, n_steps=config.n_steps, theta_ref_limit=config.theta_ref_limit(params)
    )
    return np.mean(batch.itae + config.effort_weight * batch.effort, axis=-1)


def tune_fuzzy(config: FuzzyTuningConfig, params: CartPoleParams) -> TuningResult:
    initial_states = sample_tuning_initial_states(config)
    results = run_differential_evolution(
        _objective, config.bounds, (initial_states, params, config),
        de_seeds=config.de_seeds, popsize=config.popsize, maxiter=config.maxiter, tol=config.tol,
    )
    return TuningResult(runs=tuple(
        TuningRun(de_seed=seed, gains=FuzzyGains(*(float(v) for v in r.x)), cost=float(r.fun), nit=int(r.nit),
                  nfev=int(r.nfev), converged=bool(r.success), message=str(r.message))
        for seed, r in results
    ))


def fuzzy_closed_loop_spectral_radius(gains: FuzzyGains, params: CartPoleParams, theta_ref_limit: float) -> float:
    """Radio espectral del lazo cerrado linealizado en el equilibrio (análisis LOCAL)."""

    def closed_loop(s: np.ndarray) -> np.ndarray:
        force = fuzzy_force(s, gains, force_limit=params.force_mag, theta_ref_limit=theta_ref_limit)
        return euler_step(s, force, params)

    return float(np.max(np.abs(np.linalg.eigvals(finite_difference_jacobian(closed_loop, np.zeros(4))))))
