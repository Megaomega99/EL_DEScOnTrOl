"""Perturbaciones externas: fuerza adicional sobre el carro, ajena al controlador.

Canal elegido: una fuerza horizontal sobre el CARRO que se suma a la del
actuador. Así la perturbación entra por la misma física de Gymnasium (no hay
que tocar el estado a mano) y su magnitud tiene unidades claras: impulso
J = F · duración [N·s]. La perturbación NO pasa por la saturación del
actuador (es un empujón externo, no una orden del controlador).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class Disturbance(Protocol):
    def __call__(self, step: int) -> float:
        """Fuerza externa [N] aplicada durante el paso k (de t_k a t_{k+1}).

        Debe ser finita: `CartPoleTask.step()` lanza ValueError si no lo es.
        """
        ...


@dataclass(frozen=True)
class ForcePulse:
    """Pulso rectangular de fuerza: `force` N durante los pasos [start, start + duration)."""

    start_step: int
    duration_steps: int
    force: float

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError("start_step debe ser >= 0")
        if self.duration_steps < 1:
            raise ValueError("duration_steps debe ser >= 1")
        if not math.isfinite(self.force):
            raise ValueError("force debe ser finita")

    @classmethod
    def at_time(cls, t_start: float, duration: float, force: float, dt: float) -> ForcePulse:
        """Construye el pulso en segundos; exige que los tiempos caigan en la malla k·dt.

        Se prohíben tiempos fuera de la malla (p. ej. 5.01 s con dt = 0.02 s) en lugar
        de redondearlos en silencio: un redondeo oculto cambiaría el impulso aplicado.
        """
        start, n = t_start / dt, duration / dt
        if not (math.isclose(start, round(start), abs_tol=1e-9) and math.isclose(n, round(n), abs_tol=1e-9)):
            raise ValueError(f"t_start={t_start} y duration={duration} deben ser múltiplos de dt={dt}")
        return cls(start_step=round(start), duration_steps=round(n), force=force)

    def impulse(self, dt: float) -> float:
        """J = F · duración [N·s]."""
        return self.force * self.duration_steps * dt

    def __call__(self, step: int) -> float:
        in_pulse = self.start_step <= step < self.start_step + self.duration_steps
        return self.force if in_pulse else 0.0
