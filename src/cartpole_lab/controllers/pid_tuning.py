r"""Sintonización del PID en cascada por optimización numérica, y análisis de estabilidad.

MÉTODO ELEGIDO: optimización global sin derivadas (evolución diferencial de SciPy)
de un criterio ITAE evaluado por simulación. Por qué, frente a las alternativas:

* Ziegler–Nichols: INAPLICABLE a este sistema. El método de lazo cerrado busca la
  "ganancia última" Ku en la que un control P pasa de estable a oscilación
  sostenida. En el péndulo invertido esa transición no existe: con P puro sobre θ,
  si Kp < a/b (≈ 10.8 N/rad) el poste cae exponencialmente, y si Kp > a/b es un
  oscilador sin amortiguamiento que el Euler explícito del simulador amplifica
  (|λ| = √(1 + ω²τ²) > 1). Radio espectral ≥ 1 para TODA Kp: ver
  `p_only_angle_spectral_radius` y su test. El método de la curva de reacción
  (lazo abierto) exige una planta estable, y esta no lo es.
* Búsqueda manual: no es reproducible ni defendible ante una audiencia.
* Optimización: reproducible (semilla fija), con el criterio escrito
  explícitamente, y usa el simulador como caja negra, como se sintonizaría un PID
  en la planta real, sin usar el modelo linealizado. El modelo linealizado queda
  para el LQR, y así los dos métodos se diferencian de verdad. Por velocidad se
  simula con la copia vectorizada del simulador (`euler_step`), verificada bit a
  bit contra Gymnasium, y se reproduce también la cuantización float32 de la
  observación.

CRITERIO (media sobre N condiciones iniciales):

    J = Σ_k t_k·(|θ_k|/θ_max + |x_k|/x_max)·τ  +  w_u·Σ_k (F_k/F_max)²·τ
        └────────────── ITAE ──────────────┘     └──── esfuerzo ────┘

* ITAE (Graham & Lathrop, 1953) es un criterio clásico de sintonización de PID:
  el factor t_k castiga el error que persiste y favorece un asentamiento rápido
  sin oscilación residual.
* Los errores se normalizan por los límites de fallo para sumar magnitudes
  adimensionales. θ y x pesan igual: la jerarquía θ > x ya la impone la
  estructura en cascada, no los pesos.
* Tras un fallo se imputa el error máximo (1 por término, es decir 2) hasta el
  final del horizonte. Así un fallo tardío cuesta menos que uno temprano, sin
  inventar una constante de penalización.
* Es DISTINTO del coste de evaluación común (cuadrático, `cost.py`), para no
  sintonizar el PID a medida de la métrica con la que luego se compara.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import differential_evolution

from cartpole_lab.controllers.pid import DEFAULT_THETA_REF_FRACTION, PIDGains, pid_force
from cartpole_lab.dynamics import euler_step, finite_difference_jacobian
from cartpole_lab.params import CartPoleParams
from cartpole_lab.rollout import Trajectory

GAIN_NAMES = ("kp_theta", "ki_theta", "kd_theta", "kp_x", "kd_x")

# Límites de búsqueda. Donde hay un argumento físico se da; el resto son holgados a propósito.
GAIN_BOUNDS = (
    (0.0, 200.0),  # kp_theta: con 200 N/rad, 0.05 rad (2.9°) ya satura el actuador; más sería un relé
    (0.0, 200.0),  # ki_theta: holgado (sin argumento físico previo)
    # kd_theta: aislando el lazo de θ̇ (aproximación desacoplada, NO una prueba para el
    # lazo completo), θ̇_{k+1} ≈ (1 − τ·b·Kd)·θ̇_k diverge si τ·b·Kd > 2, es decir Kd > 68.
    # Solo acota la búsqueda con margen; la estabilidad real del resultado la verifica
    # closed_loop_spectral_radius.
    (0.0, 40.0),
    (0.0, 0.5),  # kp_x: con 0.5 rad/m, 10 cm de error ya piden el límite de 3°
    (0.0, 0.5),  # kd_x: ídem por m/s
)

# Error normalizado imputado en cada paso tras un fallo (1 por término, 2 términos).
FAILURE_ERROR = 2.0


@dataclass(frozen=True)
class TuningConfig:
    """Todo lo que determina el resultado de la sintonización. Se guarda junto a las ganancias."""

    n_initial_states: int = 32
    # SUPUESTO (convención del proyecto): caja de condiciones iniciales ±(x, ẋ, θ, θ̇).
    # Contiene la distribución oficial U(±0.05) y la amplía para que ambos lazos
    # trabajen: 0.5 m de desplazamiento obliga al lazo de posición a actuar, y
    # 0.1 rad (5.7°) es la mitad del ángulo de fallo.
    initial_state_box: tuple[float, float, float, float] = (0.5, 0.2, 0.1, 0.2)
    initial_state_seed: int = 2026
    n_steps: int = 500  # horizonte = episodio completo de CartPole-v1 (10 s)
    # SUPUESTO SIN BASE TEÓRICA: peso del esfuerzo. Evita soluciones de ganancia enorme que
    # "tiritan" saturadas (casi gratis en ITAE), sin dominar el criterio.
    effort_weight: float = 0.1
    theta_ref_fraction: float = DEFAULT_THETA_REF_FRACTION
    bounds: tuple[tuple[float, float], ...] = GAIN_BOUNDS
    # Tres ejecuciones independientes del optimizador: si coinciden, el óptimo no es
    # un accidente de la semilla. Se reportan todas.
    de_seeds: tuple[int, ...] = (0, 1, 2)
    popsize: int = 20  # población = popsize × 5 ganancias = 100 candidatos por generación
    maxiter: int = 150
    tol: float = 1e-8

    def theta_ref_limit(self, params: CartPoleParams) -> float:
        return self.theta_ref_fraction * params.theta_threshold_radians

    def to_dict(self) -> dict:
        """Versión JSON (tuplas -> listas), comparable con la guardada en disco."""
        return json.loads(json.dumps(asdict(self)))


@dataclass(frozen=True)
class BatchRollout:
    itae: np.ndarray  # (...) ITAE por episodio
    effort: np.ndarray  # (...) Σ (F/F_max)² τ por episodio
    alive: np.ndarray  # (...) True si sobrevivió todo el horizonte
    steps_alive: np.ndarray  # (...) pasos ejecutados antes de terminar
    states: np.ndarray | None = None  # (T+1, ..., 4) solo con record=True (estado float64 interno)
    forces: np.ndarray | None = None  # (T, ...)


@dataclass(frozen=True)
class TuningRun:
    de_seed: int
    gains: PIDGains
    cost: float
    nit: int
    nfev: int
    converged: bool  # criterio de convergencia de la ED alcanzado antes de maxiter
    message: str


@dataclass(frozen=True)
class TuningResult:
    runs: tuple[TuningRun, ...]

    @property
    def best(self) -> TuningRun:
        return min(self.runs, key=lambda run: run.cost)


def sample_tuning_initial_states(config: TuningConfig) -> np.ndarray:
    rng = np.random.default_rng(config.initial_state_seed)
    box = np.asarray(config.initial_state_box, dtype=np.float64)
    return rng.uniform(-box, box, size=(config.n_initial_states, 4))


def simulate_pid_batch(
    gains: PIDGains,
    initial_states: ArrayLike,
    params: CartPoleParams,
    *,
    n_steps: int,
    theta_ref_limit: float,
    record: bool = False,
) -> BatchRollout:
    """Simula en paralelo (candidatos × condiciones iniciales) el lazo cerrado PID + CartPole.

    Réplica exacta de `run_episode(CartPoleTask(...), CascadePID(...))`: misma física,
    el controlador ve el estado redondeado a float32 (como la observación oficial) y
    un episodio terminado se congela (ya no acumula esfuerzo).
    """
    s0 = np.asarray(initial_states, dtype=np.float64)
    batch_shape = np.broadcast_shapes(np.shape(gains.kp_theta), s0.shape[:-1])
    state = np.broadcast_to(s0, batch_shape + (4,)).copy()
    integral = np.zeros(batch_shape)
    alive = np.ones(batch_shape, dtype=bool)
    steps_alive = np.zeros(batch_shape, dtype=np.int64)
    itae = np.zeros(batch_shape)
    effort = np.zeros(batch_shape)
    p = params
    states, forces = ([state.copy()], []) if record else (None, None)

    for k in range(n_steps):
        observation = state.astype(np.float32).astype(np.float64)
        force, new_integral = pid_force(
            observation, integral, gains, dt=p.tau, force_limit=p.force_mag, theta_ref_limit=theta_ref_limit
        )
        force = np.where(alive, force, 0.0)
        state = np.where(alive[..., None], euler_step(state, force, p), state)
        integral = np.where(alive, new_integral, integral)
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


def itae_cost(traj: Trajectory, params: CartPoleParams, config: TuningConfig) -> float:
    """El criterio de sintonización aplicado a UNA trayectoria real (de cualquier método).

    Misma definición que `simulate_pid_batch`: error real mientras vive, error
    imputado de 2 desde el paso en que falla hasta el final del horizonte.
    Sirve para comparar métodos con el criterio del PID.
    """
    if traj.n_steps > config.n_steps:
        raise ValueError(f"La trayectoria ({traj.n_steps} pasos) supera el horizonte ({config.n_steps})")
    if not traj.terminated and traj.n_steps != config.n_steps:
        # Sin esta guarda, un episodio truncado antes del horizonte (otro max_episode_steps) se
        # rellenaría con "error de fallo" como si hubiera fallado.
        raise ValueError(
            f"Episodio no fallido de {traj.n_steps} pasos con horizonte {config.n_steps}: horizontes incompatibles"
        )
    p = params
    after_step = traj.states[1:]
    error = np.abs(after_step[:, 2]) / p.theta_threshold_radians + np.abs(after_step[:, 0]) / p.x_threshold
    if traj.terminated:
        error[-1] = FAILURE_ERROR  # el estado que viola el límite ya cuenta como fallo
    error = np.concatenate([error, np.full(config.n_steps - traj.n_steps, FAILURE_ERROR)])
    times = np.arange(1, config.n_steps + 1) * p.tau
    itae = float(np.sum(times * error) * p.tau)
    effort = float(np.sum(np.square(traj.forces / p.force_mag)) * p.tau)
    return itae + config.effort_weight * effort


def _objective(
    candidates: np.ndarray, initial_states: np.ndarray, params: CartPoleParams, config: TuningConfig
) -> np.ndarray:
    """J para S candidatos a la vez: `candidates` tiene forma (5, S) (API vectorizada de SciPy)."""
    gains = PIDGains(*(row[:, None] for row in candidates))  # (S, 1) difunde con (N,) -> (S, N)
    batch = simulate_pid_batch(
        gains, initial_states, params, n_steps=config.n_steps, theta_ref_limit=config.theta_ref_limit(params)
    )
    return np.mean(batch.itae + config.effort_weight * batch.effort, axis=-1)


def tune_pid(config: TuningConfig, params: CartPoleParams) -> TuningResult:
    """Una ejecución de evolución diferencial por cada semilla de `config.de_seeds`."""
    initial_states = sample_tuning_initial_states(config)
    runs = []
    for de_seed in config.de_seeds:
        result = differential_evolution(
            _objective,
            bounds=config.bounds,
            args=(initial_states, params, config),
            seed=de_seed,
            popsize=config.popsize,
            maxiter=config.maxiter,
            tol=config.tol,
            polish=False,  # el criterio no es diferenciable (saturación, fallos): sin L-BFGS final
            vectorized=True,
            updating="deferred",  # obligatorio con vectorized=True
        )
        runs.append(
            TuningRun(
                de_seed=de_seed,
                gains=PIDGains(*(float(v) for v in result.x)),
                cost=float(result.fun),
                nit=int(result.nit),
                nfev=int(result.nfev),
                converged=bool(result.success),
                message=str(result.message),
            )
        )
    return TuningResult(runs=tuple(runs))


def closed_loop_eigenvalues(
    gains: PIDGains, params: CartPoleParams, theta_ref_limit: float, *, include_integral: bool = True
) -> np.ndarray:
    """Autovalores del lazo cerrado linealizado en el equilibrio (mapa discreto del simulador).

    include_integral=True: sistema completo de 5 estados (s, ∫e).
    include_integral=False: solo los 4 modos físicos con Ki = 0 (la parte PD).

    Por qué hacen falta los dos: si Ki ≈ 0, el integrador casi no realimenta y su
    modo queda en λ ≈ 1. El radio espectral del sistema completo sale entonces ≈ 1
    aunque el poste y el carro converjan con margen. Ese número por sí solo
    engañaría; los modos físicos dicen la verdad sobre la convergencia del estado.
    Es un análisis LOCAL: no dice nada sobre condiciones iniciales grandes (saturación).
    """
    if include_integral:

        def full_step(z: np.ndarray) -> np.ndarray:
            return _closed_loop_step(z, gains, params, theta_ref_limit)

        return np.linalg.eigvals(finite_difference_jacobian(full_step, np.zeros(5)))

    pd_gains = PIDGains(gains.kp_theta, 0.0, gains.kd_theta, gains.kp_x, gains.kd_x)

    def physical_step(s: np.ndarray) -> np.ndarray:
        return _closed_loop_step(np.append(s, 0.0), pd_gains, params, theta_ref_limit)[:4]

    return np.linalg.eigvals(finite_difference_jacobian(physical_step, np.zeros(4)))


def _closed_loop_step(z: np.ndarray, gains: PIDGains, params: CartPoleParams, theta_ref_limit: float) -> np.ndarray:
    """z_{k+1} para z = [x, ẋ, θ, θ̇, ∫e]: un paso de simulador + controlador."""
    state, integral = z[:4], z[4]
    force, new_integral = pid_force(
        state, integral, gains, dt=params.tau, force_limit=params.force_mag, theta_ref_limit=theta_ref_limit
    )
    return np.append(euler_step(state, force, params), new_integral)


def closed_loop_spectral_radius(
    gains: PIDGains, params: CartPoleParams, theta_ref_limit: float, *, include_integral: bool = True
) -> float:
    """max |λ|. < 1: estabilidad asintótica local. Ver `closed_loop_eigenvalues`."""
    eigenvalues = closed_loop_eigenvalues(gains, params, theta_ref_limit, include_integral=include_integral)
    return float(np.max(np.abs(eigenvalues)))


def p_only_angle_spectral_radius(kp: float, params: CartPoleParams) -> float:
    """Radio espectral del subsistema (θ, θ̇) con control P puro F = Kp·θ.

    Basta el bloque (θ, θ̇): en la linealización, θ no depende de (x, ẋ), así que la
    matriz es triangular por bloques y el bloque (x, ẋ) aporta un autovalor doble en 1.
    """
    jacobian = finite_difference_jacobian(lambda s: euler_step(s, kp * s[2], params), np.zeros(4))
    return float(np.max(np.abs(np.linalg.eigvals(jacobian[2:, 2:]))))
