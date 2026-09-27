"""Protocolo de sanity check COMÚN a todos los controladores.

Se aplica tras implementar cada método (requisito del proyecto), siempre con las
mismas condiciones, para que los resultados sean comparables desde el principio:

* Positivo: N episodios desde la distribución oficial de CartPole-v1 (semillas
  1000–1049, disjuntas de cualquier semilla de sintonización o entrenamiento), más
  un estado inicial difícil de referencia.
* Negativo: estados físicamente irrecuperables con |F| ≤ 10 N. El método debe
  fallar de forma EXPLÍCITA (motivo de terminación registrado, sin NaN).

"Estabilizado" es un criterio del proyecto, no un estándar: sobrevivir los 500
pasos y, en el último segundo, |θ| < 0.5° y |x| < 5 cm. La definición formal del
tiempo de asentamiento llegará con las métricas de la Fase 3.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import differential_evolution

from cartpole_lab.dynamics import euler_step
from cartpole_lab.env import CartPoleTask
from cartpole_lab.params import CartPoleParams
from cartpole_lab.rollout import Controller, Trajectory, run_episode

SANITY_SEEDS = range(1000, 1050)

# Esquina de la caja de condiciones iniciales usada para sintonizar el PID
# (x, ẋ, θ, θ̇). Se reutiliza para todos los métodos para que la figura sea comparable.
REFERENCE_HARD_INITIAL_STATE = (0.5, 0.2, 0.1, 0.2)

# Irrecuperables con |F| ≤ 10 N. Evidencia (tests/test_sanity.py):
# * `best_rescue_violation`: una búsqueda global de la mejor secuencia de fuerzas no encuentra
#   ninguna que mantenga x y θ dentro de límites (violación mínima hallada ≈ 1.73 y ≈ 1.45).
# * Para el carro, además, una cota física: `momentum_bound_on_cart_excursion` ≥ 2.66 m > 2.4 m.
# Criterio de elección: margen AMPLIO, no casos frontera. Un estado anterior del carro
# (x = 2.2, ẋ = 2) solo tenía ~1 cm de margen por el lado del carro, y se descartó.
UNRECOVERABLE_INITIAL_STATES = {
    "poste_cayendo_rapido": ((0.0, 0.0, 0.15, 2.0), frozenset({"pole_angle_limit"})),
    "carro_lanzado_al_borde": ((2.0, 3.5, 0.0, 0.0), frozenset({"cart_position_limit", "pole_angle_limit"})),
}
RESCUE_SEARCH_HORIZON = 25  # pasos (0.5 s): en ambos casos el fallo ocurre mucho antes

TAIL_SECONDS = 1.0
THETA_TOLERANCE_RAD = float(np.radians(0.5))
X_TOLERANCE_M = 0.05


def _tail(traj: Trajectory, params: CartPoleParams) -> np.ndarray:
    return traj.states[-int(round(TAIL_SECONDS / params.tau)) :]


def is_stabilized(traj: Trajectory, params: CartPoleParams) -> bool:
    if traj.termination_reason != "time_limit":
        return False
    tail = _tail(traj, params)
    return bool(
        np.max(np.abs(tail[:, 2])) < THETA_TOLERANCE_RAD and np.max(np.abs(tail[:, 0])) < X_TOLERANCE_M
    )


def _summary(values: list[float]) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0, "max": float(v.max())}


def sanity_statistics(
    controller: Controller, params: CartPoleParams, seeds: range = SANITY_SEEDS
) -> dict:
    """Estadísticos sobre N episodios en el CartPole-v1 real, con media ± desviación típica."""
    task = CartPoleTask(action_mode=controller.action_mode)
    trajectories = []
    try:
        for seed in seeds:
            trajectories.append(run_episode(task, controller, seed=seed))
    finally:
        task.close()
    tails = [_tail(t, params) for t in trajectories]
    return {
        "n_episodes": len(trajectories),
        "seeds": [seeds.start, seeds.stop - 1],
        "survival_rate": float(np.mean([t.termination_reason == "time_limit" for t in trajectories])),
        "stabilized_rate": float(np.mean([is_stabilized(t, params) for t in trajectories])),
        "termination_reasons": dict(Counter(t.termination_reason for t in trajectories)),
        "last_second_max_abs_theta_deg": _summary([np.degrees(np.max(np.abs(s[:, 2]))) for s in tails]),
        "last_second_max_abs_x_m": _summary([np.max(np.abs(s[:, 0])) for s in tails]),
        "fraction_of_steps_at_force_limit": _summary([t.fraction_at_force_limit for t in trajectories]),
        "control_effort_abs_Ns": _summary([t.control_effort_abs for t in trajectories]),
    }


def momentum_bound_on_cart_excursion(state: ArrayLike, params: CartPoleParams) -> float:
    r"""Cota inferior de la excursión máxima del carro en su dirección de avance [m].

    Argumento físico (tiempo continuo): la única fuerza horizontal externa es F, así
    que el centro de masas del sistema, x_c = x + (m l / M) sin θ, cumple M ẍ_c = F
    (se puede comprobar con las ecuaciones de Gymnasium). Con |F| ≤ F_max, frena como
    mucho a F_max/M y recorre al menos v_c² M / (2 F_max) antes de parar. Mientras el
    episodio sigue vivo, |θ| ≤ θ_max, así que x ≥ x_c − (m l / M) sin θ_max.

    Si la cota supera x_threshold, ningún controlador admisible evita el fallo. El
    simulador es de Euler discreto, así que la cota es aproximada en O(τ): solo se
    usa con márgenes amplios.
    """
    x, x_dot, theta, theta_dot = np.asarray(state, dtype=np.float64)
    ratio = params.polemass_length / params.total_mass
    x_c = x + ratio * np.sin(theta)
    v_c = x_dot + ratio * np.cos(theta) * theta_dot
    direction = np.sign(v_c) if v_c != 0 else 1.0
    stopping_distance = v_c**2 * params.total_mass / (2.0 * params.force_mag)
    return float(direction * x_c + stopping_distance - ratio * np.sin(params.theta_threshold_radians))


def best_rescue_violation(
    initial_state: ArrayLike,
    params: CartPoleParams,
    *,
    horizon: int = RESCUE_SEARCH_HORIZON,
    seed: int = 0,
    maxiter: int = 300,
) -> float:
    """Búsqueda global de la secuencia de fuerzas |F| ≤ F_max que mejor evita el fallo.

    Minimiza, con evolución diferencial y sobre el modelo exacto del simulador,
    v = max_k max(|x_k|/x_max, |θ_k|/θ_max) en `horizon` pasos. Si v > 1, TODAS las
    secuencias evaluadas terminan el episodio.

    Límite de la afirmación: la evolución diferencial es heurística. El valor devuelto
    es una cota SUPERIOR del mínimo verdadero: es evidencia numérica, no una
    demostración. Por eso se exige un margen amplio (v ≫ 1) y se valida la
    herramienta con un control positivo, un estado recuperable donde debe dar v < 1.
    """
    s0 = np.asarray(initial_state, dtype=np.float64)
    limits = np.array([params.x_threshold, params.theta_threshold_radians])

    def peak_violation(forces: np.ndarray) -> np.ndarray:  # forces: (horizon, S)
        state = np.broadcast_to(s0, (forces.shape[1], 4)).copy()
        worst = np.full(forces.shape[1], np.max(np.abs(s0[[0, 2]]) / limits))
        for k in range(horizon):
            state = euler_step(state, forces[k], params)
            worst = np.maximum(worst, np.max(np.abs(state[:, [0, 2]]) / limits, axis=1))
        return worst

    result = differential_evolution(
        peak_violation,
        bounds=[(-params.force_mag, params.force_mag)] * horizon,
        seed=seed,
        maxiter=maxiter,
        popsize=15,
        tol=1e-12,
        polish=False,
        vectorized=True,
        updating="deferred",
    )
    return float(result.fun)
