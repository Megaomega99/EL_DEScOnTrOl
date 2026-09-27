"""Parámetros físicos de CartPole-v1, LEÍDOS de Gymnasium (nunca copiados a mano).

Por qué se leen del objeto del entorno en lugar de escribir 9.8, 1.0, 0.1, ...:
el requisito es que el modelo interno de LQR/MPC coincida EXACTAMENTE con el
simulador. Un valor copiado a mano podría divergir en silencio si Gymnasium
cambiara una constante; leído en tiempo de ejecución, el modelo sigue siempre
al simulador, y `tests/test_gymnasium_conventions.py` avisa del cambio.

Los nombres de los campos son idénticos a los atributos de
`gymnasium.envs.classic_control.CartPoleEnv` a propósito: así la correspondencia
con el código fuente de Gymnasium es inmediata y no hay errores de traducción.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import gymnasium as gym
from gymnasium.envs.classic_control import CartPoleEnv

ENV_ID = "CartPole-v1"


@dataclass(frozen=True)
class CartPoleParams:
    """Constantes que usa `CartPoleEnv.step()` (Gymnasium 1.3.0, cartpole.py).

    Unidades SI. Valores en CartPole-v1 entre corchetes.
    """

    gravity: float  # g [9.8 m/s²]
    masscart: float  # M, masa del carro [1.0 kg]
    masspole: float  # m, masa del poste [0.1 kg]
    length: float  # l [0.5 m] — ¡MITAD de la longitud del poste! (pivote → centro de masa)
    force_mag: float  # F_max [10 N] — la acción discreta aplica ±F_max
    tau: float  # Δt [0.02 s] — paso de integración = periodo de control (50 Hz)
    x_threshold: float  # [2.4 m] — |x| > x_threshold termina el episodio
    theta_threshold_radians: float  # [12° ≈ 0.2094 rad] — |θ| > esto termina el episodio
    kinematics_integrator: str  # ["euler"] — Euler explícito (ver dynamics.py)

    @property
    def total_mass(self) -> float:
        """M + m. Misma operación que Gymnasium (masspole + masscart) -> mismo float."""
        return self.masspole + self.masscart

    @property
    def polemass_length(self) -> float:
        """m·l, el producto que aparece en las ecuaciones de movimiento."""
        return self.masspole * self.length

    @classmethod
    def from_env(cls, env: gym.Env) -> CartPoleParams:
        """Extrae los parámetros de un entorno CartPole (envuelto o no)."""
        base = env.unwrapped
        if not isinstance(base, CartPoleEnv):
            raise TypeError(f"Se esperaba un CartPoleEnv, se recibió {type(base).__name__}")
        return cls(
            gravity=base.gravity,
            masscart=base.masscart,
            masspole=base.masspole,
            length=base.length,
            force_mag=base.force_mag,
            tau=base.tau,
            x_threshold=base.x_threshold,
            theta_threshold_radians=base.theta_threshold_radians,
            kinematics_integrator=base.kinematics_integrator,
        )


@lru_cache(maxsize=1)
def load_params() -> CartPoleParams:
    """Parámetros de un CartPole-v1 recién creado. Cacheado: la dataclass es inmutable."""
    env = gym.make(ENV_ID)
    try:
        return CartPoleParams.from_env(env)
    finally:
        env.close()
