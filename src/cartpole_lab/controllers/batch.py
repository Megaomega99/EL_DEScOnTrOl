"""Simulación por lotes y optimización compartidas por los controladores que se sintonizan
por simulación (PID, fuzzy). Evita duplicar la parte delicada: la réplica exacta del
rollout de Gymnasium y el criterio ITAE con imputación de fallos.

Una ley de control por lotes tiene la firma

    law(observation (..., 4), memory (...)) -> (force (...), new_memory (...))

`memory` es el estado interno del controlador (la integral del PID); un controlador
sin memoria (el fuzzy) la ignora y la devuelve tal cual.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, differential_evolution

from cartpole_lab.dynamics import euler_step
from cartpole_lab.params import CartPoleParams

BatchLaw = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class Gains(Protocol):
    """Lo mínimo que se exige a las ganancias sintonizadas (PIDGains, FuzzyGains)."""

    def to_dict(self) -> dict[str, float]: ...

# Error normalizado imputado en cada paso tras un fallo (1 por término, 2 términos).
FAILURE_ERROR = 2.0


@dataclass(frozen=True)
class BatchRollout:
    itae: np.ndarray  # (...) ITAE por episodio
    effort: np.ndarray  # (...) Σ (F/F_max)² τ por episodio
    alive: np.ndarray  # (...) True si sobrevivió todo el horizonte
    steps_alive: np.ndarray  # (...) pasos ejecutados antes de terminar
    states: np.ndarray | None = None  # (T+1, ..., 4) solo con record=True (estado float64 interno)
    forces: np.ndarray | None = None  # (T, ...)


def simulate_batch(
    law: BatchLaw,
    initial_states: ArrayLike,
    params: CartPoleParams,
    *,
    batch_shape: tuple[int, ...],
    n_steps: int,
    record: bool = False,
) -> BatchRollout:
    """Simula en paralelo (candidatos × condiciones iniciales) un lazo cerrado con CartPole.

    Réplica exacta de `run_episode(CartPoleTask(...), controlador)`: misma física (euler_step,
    idéntico bit a bit a Gymnasium), el controlador ve el estado redondeado a float32 (como
    la observación oficial) y un episodio terminado se congela (ya no acumula esfuerzo).
    """
    s0 = np.asarray(initial_states, dtype=np.float64)
    state = np.broadcast_to(s0, batch_shape + (4,)).copy()
    memory = np.zeros(batch_shape)
    alive = np.ones(batch_shape, dtype=bool)
    steps_alive = np.zeros(batch_shape, dtype=np.int64)
    itae = np.zeros(batch_shape)
    effort = np.zeros(batch_shape)
    p = params
    states, forces = ([state.copy()], []) if record else (None, None)

    for k in range(n_steps):
        observation = state.astype(np.float32).astype(np.float64)
        force, new_memory = law(observation, memory)
        force = np.where(alive, force, 0.0)
        state = np.where(alive[..., None], euler_step(state, force, p), state)
        memory = np.where(alive, new_memory, memory)
        effort += np.where(alive, np.square(force / p.force_mag), 0.0) * p.tau
        steps_alive += alive

        alive &= (np.abs(state[..., 0]) <= p.x_threshold) & (np.abs(state[..., 2]) <= p.theta_threshold_radians)
        error = np.abs(state[..., 2]) / p.theta_threshold_radians + np.abs(state[..., 0]) / p.x_threshold
        itae += (k + 1) * p.tau * np.where(alive, error, FAILURE_ERROR) * p.tau
        if record:
            states.append(state.copy())
            forces.append(force)

    return BatchRollout(
        itae=itae,
        effort=effort,
        alive=alive,
        steps_alive=steps_alive,
        states=np.stack(states) if record else None,
        forces=np.stack(forces) if record else None,
    )


def run_differential_evolution(
    objective: Callable[..., np.ndarray],
    bounds: Sequence[tuple[float, float]],
    args: tuple,
    *,
    de_seeds: Sequence[int],
    popsize: int,
    maxiter: int,
    tol: float,
) -> list[tuple[int, OptimizeResult]]:
    """Una ejecución de evolución diferencial (vectorizada) por semilla; se devuelven todas."""
    results = []
    for de_seed in de_seeds:
        result = differential_evolution(
            objective,
            bounds=bounds,
            args=args,
            seed=de_seed,
            popsize=popsize,
            maxiter=maxiter,
            tol=tol,
            polish=False,  # el criterio no es diferenciable (saturación, fallos): sin L-BFGS final
            vectorized=True,
            updating="deferred",  # obligatorio con vectorized=True
        )
        results.append((de_seed, result))
    return results
