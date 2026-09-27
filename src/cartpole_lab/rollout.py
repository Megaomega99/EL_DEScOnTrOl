"""Bucle de simulación común: el MISMO código ejecuta PID, LQR, MPC, fuzzy y agentes RL.

Un único bucle garantiza que ningún método tenga ventajas ocultas (otro orden de
observación/acción, otro criterio de parada, otra forma de medir el esfuerzo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from cartpole_lab.env import CartPoleTask

FORCE_LIMIT_RTOL = 1e-6  # tolerancia para considerar que el actuador está en su límite


class Controller(Protocol):
    """Contrato mínimo de cualquier controlador o agente entrenado."""

    action_mode: str  # "continuous" (devuelve F en N) o "discrete" (devuelve 0/1)

    def reset(self) -> None:
        """Borra el estado interno (integrador del PID, warm start del MPC, ...)."""
        ...

    def __call__(self, state: np.ndarray) -> float | int:
        """Acción para el estado s_k = [x, ẋ, θ, θ̇]."""
        ...


@dataclass(frozen=True)
class Trajectory:
    """Registro de un episodio. Arrays de solo lectura (inmutables tras crearse)."""

    times: np.ndarray  # (T+1,) t_k [s]
    states: np.ndarray  # (T+1, 4) s_k, incluido el estado inicial
    forces: np.ndarray  # (T,) fuerza del actuador tras saturar [N]
    disturbance_forces: np.ndarray  # (T,) fuerza externa [N]
    rewards: np.ndarray  # (T,) recompensa de Gymnasium
    costs: np.ndarray  # (T,) coste de etapa común
    saturated: np.ndarray  # (T,) bool: el ACTUADOR recortó la petición (ver nota abajo)
    terminated: bool
    truncated: bool
    termination_reason: str
    dt: float
    force_limit: float  # F_max del actuador [N]

    # Nota sobre `saturated`: solo es True si el controlador pidió |F| > F_max. Un
    # controlador que ya recorta internamente (PID con anti-windup, LQR) o un agente
    # discreto (±10 N exactos) nunca lo activa aunque trabaje en el límite. Para
    # comparar métodos se usa `fraction_at_force_limit`, que es una medida física.

    @property
    def fraction_at_force_limit(self) -> float:
        """Fracción de pasos con el actuador en su límite |F| = F_max, lo recorte quien lo recorte.

        Con tolerancia relativa de 1e-6: un solver de optimización (MPC) entrega 9.9999999 N
        cuando la restricción |F| ≤ 10 N está activa, y eso también es "en el límite".
        """
        if self.n_steps == 0:
            return 0.0
        return float(np.mean(np.abs(self.forces) >= self.force_limit * (1.0 - FORCE_LIMIT_RTOL)))

    @property
    def n_steps(self) -> int:
        return len(self.forces)

    @property
    def duration(self) -> float:
        return self.n_steps * self.dt

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())

    @property
    def total_cost(self) -> float:
        return float(self.costs.sum())

    @property
    def control_effort_abs(self) -> float:
        """Σ |F_k| τ [N·s]: impulso total entregado por el actuador."""
        return float(np.abs(self.forces).sum() * self.dt)

    @property
    def control_effort_sq(self) -> float:
        """Σ F_k² τ [N²·s]: esfuerzo "energético", el que penaliza un coste cuadrático."""
        return float(np.square(self.forces).sum() * self.dt)


def _frozen(values, dtype) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


def run_episode(
    task: CartPoleTask,
    controller: Controller,
    *,
    seed: int | None = None,
    initial_state: ArrayLike | None = None,
) -> Trajectory:
    """Simula un episodio completo y devuelve la trayectoria registrada."""
    if controller.action_mode != task.action_mode:
        raise ValueError(
            f"El controlador es {controller.action_mode!r} pero la tarea es {task.action_mode!r}"
        )
    options = None if initial_state is None else {"initial_state": initial_state}
    state, _ = task.reset(seed=seed, options=options)
    controller.reset()

    states, forces, disturbances, rewards, costs, saturated = [state], [], [], [], [], []
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        state, reward, terminated, truncated, info = task.step(controller(state))
        states.append(state)
        forces.append(info["force"])
        disturbances.append(info["disturbance_force"])
        rewards.append(reward)
        costs.append(info["cost"])
        saturated.append(info["saturated"])

    n = len(forces)
    return Trajectory(
        times=_frozen(np.arange(n + 1) * task.dt, np.float64),
        states=_frozen(states, np.float64),
        forces=_frozen(forces, np.float64),
        disturbance_forces=_frozen(disturbances, np.float64),
        rewards=_frozen(rewards, np.float64),
        costs=_frozen(costs, np.float64),
        saturated=_frozen(saturated, bool),
        terminated=terminated,
        truncated=truncated,
        termination_reason=info["termination_reason"],
        dt=task.dt,
        force_limit=task.params.force_mag,
    )
