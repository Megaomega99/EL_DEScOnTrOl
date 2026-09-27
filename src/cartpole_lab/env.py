"""`CartPoleTask`: la interfaz común que usan TODOS los controladores (clásicos y RL).

Qué garantiza (y los tests verifican) para que las comparaciones sean válidas:

* Misma física: el wrapper envuelve el `CartPole-v1` oficial y deja que su
  `step()` integre la dinámica. No hay una segunda implementación del simulador.
* Mismo estado: s = [x, ẋ, θ, θ̇] en float64, tomado de la observación oficial.
* Misma señal de recompensa: la de CartPole-v1 sin modificar (+1 por paso,
  incluido el terminal). Útil para RL y para comparar con la literatura.
* Mismo coste de evaluación: `QuadraticCost` (ver cost.py), idéntico para todos.
* Misma terminación: la de CartPole-v1 (|θ| > 12°, |x| > 2.4 m, 500 pasos),
  con el MOTIVO explícito en `info["termination_reason"]`.

La única diferencia permitida entre métodos es el actuador (`action_mode`):

* "discrete": acción ∈ {0, 1} -> F = ±10 N. Es el CartPole-v1 canónico
  (el que usan Q-learning, SARSA y DQN en la literatura).
* "continuous": acción = F ∈ [-10, 10] N. Lo que producen de forma natural
  PID, LQR y MPC. El límite ±10 N es el mismo F_max del entorno original.

Cómo se aplica una fuerza continua sin reimplementar la física (decisión clave):
CartPoleEnv calcula `force = force_mag if action == 1 else -force_mag`. Si antes
de cada paso se fija `force_mag = |F|` y `action = 1 si F >= 0 si no 0`, el
propio código de Gymnasium aplica exactamente F. Se restaura `force_mag` al
terminar el paso. `force_mag` solo se usa en `step()` (verificado en el código
fuente de la v1.3.0), y los tests comprueban igualdad bit a bit con Gymnasium.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import ArrayLike

from cartpole_lab.cost import QuadraticCost
from cartpole_lab.disturbances import Disturbance
from cartpole_lab.params import ENV_ID, CartPoleParams

ActionMode = Literal["continuous", "discrete"]
ACTION_MODES: tuple[str, ...] = ("continuous", "discrete")


class CartPoleTask(gym.Wrapper):
    """CartPole-v1 con actuador configurable, perturbaciones y métricas comunes.

    Claves añadidas a `info` en cada `step()`:
        force_requested  fuerza pedida por el controlador [N] (antes de saturar)
        force            fuerza del actuador tras saturar a ±F_max [N]
        saturated        True si se recortó la fuerza pedida (solo ocurre si el controlador pide
                         |F| > F_max; para medir tiempo en el límite usar
                         Trajectory.fraction_at_force_limit)
        disturbance_force  fuerza externa sobre el carro [N] (no se satura)
        total_force      force + disturbance_force [N], lo que integra el simulador
        cost             coste de etapa c(s_k, F_k) con el estado ANTES del paso
        step, time       índice k+1 y tiempo t_{k+1} = (k+1)·τ [s] tras el paso
        termination_reason  None | "pole_angle_limit" | "cart_position_limit" | "time_limit"
    """

    def __init__(
        self,
        action_mode: ActionMode = "continuous",
        *,
        max_episode_steps: int | None = None,
        render_mode: str | None = None,
        cost: QuadraticCost | None = None,
        disturbance: Disturbance | None = None,
    ) -> None:
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode debe ser uno de {ACTION_MODES}, no {action_mode!r}")
        # max_episode_steps=None conserva el límite oficial de CartPole-v1 (500 pasos = 10 s).
        super().__init__(gym.make(ENV_ID, render_mode=render_mode, max_episode_steps=max_episode_steps))

        self.params = CartPoleParams.from_env(self.env)
        self.action_mode = action_mode
        self.cost = cost if cost is not None else QuadraticCost.normalized(self.params)
        self.disturbance = disturbance

        f_max = self.params.force_mag
        if action_mode == "continuous":
            self.action_space = spaces.Box(-f_max, f_max, shape=(1,), dtype=np.float64)
        # Mismos límites que CartPole-v1, pero en float64 (el formato común de estado).
        high = self.env.observation_space.high.astype(np.float64)
        self.observation_space = spaces.Box(-high, high, dtype=np.float64)

        self._step_index = 0
        self._needs_reset = True

    @property
    def dt(self) -> float:
        """Periodo de muestreo/control τ [s]."""
        return self.params.tau

    # ------------------------------------------------------------------ reset
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reinicia el episodio.

        options["initial_state"]: estado inicial exacto [x, ẋ, θ, θ̇]. Gymnasium solo
        admite límites escalares ("low"/"high") para su muestreo uniforme, así que la
        condición inicial exacta se impone sobrescribiendo `unwrapped.state` DESPUÉS
        del reset oficial (que ya reinició el RNG, el contador de tiempo y el
        indicador de terminación). Sin "initial_state" se usa la distribución
        oficial U(-0.05, 0.05)^4, o la que indiquen "low"/"high".
        """
        options = dict(options) if options else {}
        initial_state = options.pop("initial_state", None)
        if initial_state is not None and options:
            raise ValueError(
                f"'initial_state' es incompatible con otras opciones de reset: {sorted(options)}"
            )

        obs, info = self.env.reset(seed=seed, options=options or None)
        if initial_state is None:
            source = "gymnasium_uniform"
        else:
            state = self._validate_initial_state(initial_state)
            self.env.unwrapped.state = state
            obs = np.array(state, dtype=np.float32)  # mismo formato que la observación oficial
            source = "user"
            if self.render_mode == "human":
                self.env.render()

        self._step_index = 0
        self._needs_reset = False
        return obs.astype(np.float64), {**info, "initial_state_source": source}

    def _validate_initial_state(self, initial_state: ArrayLike) -> np.ndarray:
        state = np.asarray(initial_state, dtype=np.float64)
        if state.shape != (4,):
            raise ValueError(f"initial_state debe tener forma (4,), no {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError(f"initial_state contiene valores no finitos: {state}")
        p = self.params
        if abs(state[0]) > p.x_threshold or abs(state[2]) > p.theta_threshold_radians:
            # Fallo explícito: Gymnasium no se quejaría, pero el episodio terminaría en el
            # primer paso y el resultado ("falla en 1 paso") se confundiría con un mal control.
            raise ValueError(
                f"initial_state {state} está fuera de la región no terminal "
                f"(|x| <= {p.x_threshold} m, |θ| <= {p.theta_threshold_radians:.4f} rad)"
            )
        return state.copy()

    # ------------------------------------------------------------------- step
    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            # Gymnasium solo emitiría un warning; aquí se prefiere un error para que un
            # bucle mal escrito no siga simulando un episodio ya terminado.
            raise RuntimeError("Hay que llamar a reset() antes de step() y tras cada fin de episodio")

        f_max = self.params.force_mag
        requested = self._requested_force(action)
        force = min(max(requested, -f_max), f_max)
        disturbance_force = float(self.disturbance(self._step_index)) if self.disturbance else 0.0
        if not math.isfinite(disturbance_force):
            # Se comprueba ANTES de tocar el simulador: un NaN no hace terminar el episodio
            # (toda comparación con NaN es False) y corrompería la trayectoria en silencio.
            raise ValueError(f"La perturbación devolvió una fuerza no finita en el paso {self._step_index}: {disturbance_force}")
        total_force = force + disturbance_force
        state_before = np.array(self.env.unwrapped.state, dtype=np.float64)

        try:
            obs, reward, terminated, truncated, info = self._step_with_force(total_force)
        except Exception:
            # Si Gymnasium falla a mitad de step(), el estado interno puede haber cambiado ya:
            # el episodio deja de ser fiable y se exige reset() antes de continuar.
            self._needs_reset = True
            raise

        self._step_index += 1
        self._needs_reset = terminated or truncated
        info = {
            **info,
            "force_requested": requested,
            "force": force,
            "saturated": force != requested,
            "disturbance_force": disturbance_force,
            "total_force": total_force,
            "cost": self.cost(state_before, force),
            "step": self._step_index,
            "time": self._step_index * self.params.tau,
            "termination_reason": self._termination_reason(terminated, truncated),
        }
        return obs.astype(np.float64), float(reward), terminated, truncated, info

    def _requested_force(self, action: Any) -> float:
        """Traduce la acción del controlador a una fuerza pedida [N], validándola."""
        if self.action_mode == "discrete":
            if isinstance(action, np.ndarray) and action.shape == () and np.issubdtype(action.dtype, np.integer):
                action = action.item()
            # bool es subclase de int en Python: se rechaza para no aceptar True como "1".
            if isinstance(action, (bool, np.bool_)) or not isinstance(action, numbers.Integral):
                raise TypeError(f"En modo discreto la acción debe ser un entero 0 o 1, no {action!r}")
            if action not in (0, 1):
                raise ValueError(f"En modo discreto la acción debe ser 0 o 1, no {action}")
            return self.params.force_mag if action == 1 else -self.params.force_mag

        if isinstance(action, (str, bytes)) or np.asarray(action).dtype == np.bool_:
            # Un booleano (escalar o array) se convertiría en 0/1 N sin avisar: se rechaza.
            raise TypeError(f"En modo continuo la acción debe ser una fuerza numérica, no {action!r}")
        value = np.asarray(action, dtype=np.float64)
        if value.size != 1:
            raise ValueError(f"En modo continuo la acción debe ser escalar, forma recibida {value.shape}")
        force = float(value.reshape(()))
        if not math.isfinite(force):
            raise ValueError(f"Fuerza no finita: {force}")
        return force

    def _step_with_force(self, total_force: float):
        """Aplica una fuerza arbitraria usando el propio `step()` de Gymnasium (ver docstring del módulo)."""
        base = self.env.unwrapped
        base.force_mag = abs(total_force)
        try:
            return self.env.step(1 if total_force >= 0.0 else 0)
        finally:
            base.force_mag = self.params.force_mag

    def _termination_reason(self, terminated: bool, truncated: bool) -> str | None:
        if terminated:
            x, _, theta, _ = self.env.unwrapped.state
            reasons = []
            if abs(theta) > self.params.theta_threshold_radians:
                reasons.append("pole_angle_limit")
            if abs(x) > self.params.x_threshold:
                reasons.append("cart_position_limit")
            if not reasons:
                raise RuntimeError(f"Gymnasium terminó el episodio sin violar ningún límite: estado {x=}, {theta=}")
            return "+".join(reasons)
        return "time_limit" if truncated else None
