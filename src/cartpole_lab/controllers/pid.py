r"""PID en cascada: el ángulo es la variable controlada; la posición, un objetivo secundario.

ESTRUCTURA (dos lazos anidados)
-------------------------------
Un solo PID sobre θ mantiene el poste vertical, pero deja el carro a la deriva
hasta que sale del raíl (|x| > 2.4 m). Por eso se añade un lazo externo lento:

    lazo externo (PD sobre x):   θ_ref = sat_{±θ_lim}( −(Kp_x·x + Kd_x·ẋ) )
    lazo interno (PID sobre θ):  e = θ − θ_ref
                                 F = sat_{±F_max}( Kp_θ·e + Ki_θ·∫e dt + Kd_θ·θ̇ )

* Jerarquía: la posición solo puede pedir una inclinación de hasta ±θ_lim (por
  defecto el 25 % del límite de fallo, 3°). La estructura garantiza que el
  objetivo secundario nunca comprometa al primario; no se confía en los pesos.
* Signo del lazo externo: si el carro está a la derecha (x > 0), se pide
  inclinar el poste hacia la izquierda (θ_ref < 0). Para conseguirlo, el lazo
  interno empuja primero hacia la DERECHA. Es la respuesta de fase no mínima del
  carro-péndulo: para ir a un lado, primero hay que moverse hacia el otro.
* Error "de acción inversa" e = θ − θ_ref (y no r − y): la planta tiene ganancia
  negativa (F > 0 hace θ̈ < 0), así que con esta definición todas las ganancias
  son positivas y cada término se lee físicamente ("si cae a la derecha, empuja
  a la derecha").
* Derivada sobre la MEDIDA (Kd_θ·θ̇, no d e/dt): CartPole entrega θ̇ exacta, así
  que no hace falta derivar numéricamente, y se evita el "derivative kick"
  cuando θ_ref cambia. En un sistema real con encoder ruidoso habría que filtrar.
* Anti-windup por integración condicional: si el actuador está saturado y el
  error empuja en la misma dirección, el integrador se congela. Si no, el
  integrador acumularía error que no puede corregirse y provocaría sobreoscilación.

Con Ki_θ = 0 y sin saturaciones, la cascada es una realimentación lineal de estado:
F = Kp_θKp_x·x + Kp_θKd_x·ẋ + Kp_θ·θ + Kd_θ·θ̇. Es la misma FORMA que el LQR
(paso 4); la diferencia está en cómo se eligen las ganancias.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from cartpole_lab.params import CartPoleParams
from cartpole_lab.paths import TUNING_DIR

TUNED_GAINS_PATH = TUNING_DIR / "pid.json"

# SUPUESTO DE DISEÑO (no estándar): el lazo de posición puede pedir como mucho una
# inclinación del 25 % del ángulo de fallo (3° de 12°). Deja margen al lazo
# interno y mantiene el poste en la zona casi lineal (sin θ ≈ θ con error < 0.05 %).
DEFAULT_THETA_REF_FRACTION = 0.25


Gain = float | np.ndarray  # escalar para el controlador; array para evaluar lotes de candidatos


@dataclass(frozen=True)
class PIDGains:
    """Ganancias de la cascada. Pueden ser escalares o arrays (para evaluar lotes)."""

    kp_theta: Gain  # [N/rad]
    ki_theta: Gain  # [N/(rad·s)]
    kd_theta: Gain  # [N·s/rad]
    kp_x: Gain  # [rad/m]      ángulo de referencia pedido por metro de error
    kd_x: Gain  # [rad·s/m]    ídem por m/s de velocidad del carro

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            array = np.asarray(value, dtype=np.float64)
            if not (np.all(np.isfinite(array)) and np.all(array >= 0.0)):
                raise ValueError(f"La ganancia {name} debe ser finita y >= 0, no {value}")

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


def pid_force(
    state: ArrayLike,
    integral: ArrayLike,
    gains: PIDGains,
    *,
    dt: float,
    force_limit: float,
    theta_ref_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ley de control pura (sin estado oculto): devuelve (F_k, ∫e hasta k+1).

    Vectorizada: `state` de forma (..., 4), `integral` y las ganancias difundibles
    con (...). La misma función sirve para el controlador del rollout y para
    evaluar miles de candidatos durante la sintonización (una sola fuente de verdad).
    """
    s = np.asarray(state, dtype=np.float64)
    x, x_dot, theta, theta_dot = s[..., 0], s[..., 1], s[..., 2], s[..., 3]

    theta_ref = np.clip(-(gains.kp_x * x + gains.kd_x * x_dot), -theta_ref_limit, theta_ref_limit)
    error = theta - theta_ref
    unsaturated = gains.kp_theta * error + gains.ki_theta * integral + gains.kd_theta * theta_dot
    force = np.clip(unsaturated, -force_limit, force_limit)

    winding_up = (unsaturated != force) & (np.sign(error) == np.sign(unsaturated))
    new_integral = np.where(winding_up, integral, integral + error * dt)
    return force, new_integral


class CascadePID:
    """Controlador para `run_episode`. Su único estado interno es la integral del error."""

    action_mode = "continuous"

    def __init__(
        self, gains: PIDGains, params: CartPoleParams, *, theta_ref_limit: float | None = None
    ) -> None:
        if theta_ref_limit is None:
            theta_ref_limit = DEFAULT_THETA_REF_FRACTION * params.theta_threshold_radians
        if not 0.0 < theta_ref_limit < params.theta_threshold_radians:
            # Un límite >= 12° permitiría al lazo externo pedir un ángulo que termina el episodio.
            raise ValueError(
                f"theta_ref_limit debe estar en (0, {params.theta_threshold_radians:.4f}) rad, "
                f"no {theta_ref_limit}"
            )
        self.gains = gains
        self.theta_ref_limit = theta_ref_limit
        self._dt = params.tau
        self._force_limit = params.force_mag
        self._integral = 0.0

    def reset(self) -> None:
        self._integral = 0.0

    def __call__(self, state: np.ndarray) -> float:
        force, self._integral = pid_force(
            state,
            self._integral,
            self.gains,
            dt=self._dt,
            force_limit=self._force_limit,
            theta_ref_limit=self.theta_ref_limit,
        )
        return float(force)


def load_tuning_record(path: Path = TUNED_GAINS_PATH) -> dict[str, Any]:
    """Lee el resultado de `scripts/tune_pid.py` (ganancias + configuración + metadatos)."""
    if not path.exists():
        raise FileNotFoundError(f"No hay ganancias sintonizadas en {path}: ejecutar scripts/tune_pid.py")
    return json.loads(path.read_text(encoding="utf-8"))


def load_tuned_gains(path: Path = TUNED_GAINS_PATH) -> PIDGains:
    return PIDGains(**load_tuning_record(path)["gains"])
