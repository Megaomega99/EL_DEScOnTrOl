"""Coste de etapa común para EVALUAR (no para diseñar) todos los controladores.

    c(s, F) = Σ_i q_i s_i² + r F²

Por qué existe además de la recompensa de Gymnasium: la recompensa de CartPole-v1
(+1 por paso vivo) solo mide supervivencia; no distingue un péndulo quieto en el
centro de uno oscilando al borde del fallo. El coste cuadrático sí.

Advertencia metodológica: este coste NO debe coincidir con las matrices Q, R de
diseño del LQR/MPC. Si coincidieran, el LQR sería óptimo "por construcción" para
la métrica de evaluación y la comparación estaría sesgada a su favor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from cartpole_lab.params import CartPoleParams

# SUPUESTO SIN BASE TEÓRICA (convención del proyecto, no estándar de la literatura):
# con 0.1, saturar el actuador (|F| = F_max) cuesta el 10 % de estar en el límite
# de fallo en x o en θ. Es un valor elegido para que el esfuerzo pese, pero menos
# que el error de estado. Se reporta siempre junto a los resultados.
DEFAULT_EFFORT_WEIGHT = 0.1


@dataclass(frozen=True)
class QuadraticCost:
    state_weights: tuple[float, float, float, float]  # q_x, q_ẋ, q_θ, q_θ̇
    force_weight: float  # r

    def __post_init__(self) -> None:
        weights = np.asarray(self.state_weights, dtype=np.float64)
        if weights.shape != (4,):
            raise ValueError(f"Se necesitan 4 pesos de estado, se recibieron {weights.shape}")
        if not (np.all(np.isfinite(weights)) and np.all(weights >= 0)):
            raise ValueError("Los pesos de estado deben ser finitos y >= 0")
        if not (np.isfinite(self.force_weight) and self.force_weight >= 0):
            raise ValueError("El peso de la fuerza debe ser finito y >= 0")

    @classmethod
    def normalized(
        cls, params: CartPoleParams, effort_weight: float = DEFAULT_EFFORT_WEIGHT
    ) -> QuadraticCost:
        """Pesos adimensionales: cada término vale 1 justo en el límite de fallo.

        q_x = 1/x_max², q_θ = 1/θ_max², r = effort_weight / F_max².
        Velocidades con peso 0: Gymnasium no define límites para ẋ ni θ̇, así que
        cualquier escala de normalización sería arbitraria (se prefiere no inventarla).
        """
        return cls(
            state_weights=(
                1.0 / params.x_threshold**2,
                0.0,
                1.0 / params.theta_threshold_radians**2,
                0.0,
            ),
            force_weight=effort_weight / params.force_mag**2,
        )

    def __call__(self, state: ArrayLike, force: float) -> float:
        s = np.asarray(state, dtype=np.float64)
        return float(np.dot(self.state_weights, s**2) + self.force_weight * force**2)
