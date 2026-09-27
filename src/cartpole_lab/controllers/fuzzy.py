r"""Controlador fuzzy Takagi-Sugeno de orden 0, en la misma cascada que el PID.

IDEA
----
En lugar de una fórmula, el controlador se escribe con REGLAS en lenguaje natural:

    "SI el poste está muy inclinado a la derecha Y cae deprisa a la derecha
     ENTONCES empuja fuerte a la derecha"

y un mecanismo de inferencia que mezcla las reglas cuando la situación está "a medio
camino" entre varias.

ESTRUCTURA (idéntica a la del PID, sustituyendo solo el lazo interno)
--------------------------------------------------------------------
    lazo externo:  θ_ref = sat_{±3°}( −(Kp_x·x + Kd_x·ẋ) )           (el mismo que el PID)
    lazo interno:  e = θ − θ_ref,  θ̇   →   reglas fuzzy   →   F

1) FUZZIFICACIÓN. Cada entrada se normaliza por su factor de escala (e/E, θ̇/D) y se
   describe con 5 etiquetas: NG, NP, Z, PP, PG (negativo grande, negativo pequeño,
   cero, positivo pequeño, positivo grande). Las funciones de membresía son
   triangulares, centradas en −1, −0.5, 0, 0.5, 1, con "hombros" en los extremos
   (todo lo que supera "grande" sigue siendo grande).
   Por qué triangulares y equiespaciadas: son las más simples de explicar, y forman
   una PARTICIÓN DE LA UNIDAD (en cada punto las pertenencias suman 1). Así ningún
   estado queda sin regla y ninguno se cuenta dos veces.

2) REGLAS (tabla 5×5, escrita a mano desde la física y NUNCA optimizada). La salida
   de cada regla es un NÚMERO, no un conjunto: esto es "orden 0" de Sugeno. Es la
   tabla clásica de MacVicar-Whelan (1981): consecuente = sat(i + j), con i, j ∈ {−2..2}
   los índices de las etiquetas de e y θ̇. Se lee: "la fuerza sigue a hacia dónde cae
   el poste, más lo deprisa que cae". Si ambos van al mismo lado, fuerte. Si se
   compensan (inclinado a la derecha pero volviendo rápido), no hagas nada.

3) INFERENCIA Y SALIDA. Activación de cada regla = μ_e,i · μ_θ̇,j (el "Y" como
   producto). Salida = media de los consecuentes ponderada por las activaciones,
   multiplicada por U [N]:

       F = U · Σ_ij μ_e,i μ_θ̇,j c_ij / Σ_ij μ_e,i μ_θ̇,j

   Es un Mamdani cuyos conjuntos de salida son picos (singletons): no hace falta
   calcular el centroide.

Consecuencia verificable (test): con estas elecciones, el controlador INTERPOLA
bilinealmente la tabla. En la celda central (|e/E|, |θ̇/D| ≤ 0.5) es exactamente un PD,
F = U·(e/E + θ̇/D), con Kp = U/E y Kd = U/D, y fuera de ella satura suavemente
hasta ±U. Las reglas no inventan una ley exótica: son una forma de ESCRIBIR con
palabras una ley no lineal.

Sin término integral: el PID sintonizado ya mostró que en este sistema la I no
aporta (docs/03_pid.md).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from cartpole_lab.controllers.pid import DEFAULT_THETA_REF_FRACTION
from cartpole_lab.params import CartPoleParams
from cartpole_lab.paths import TUNING_DIR

TUNED_FUZZY_PATH = TUNING_DIR / "fuzzy.json"

LABELS = ("NG", "NP", "Z", "PP", "PG")
_CENTERS = np.linspace(-1.0, 1.0, len(LABELS))
_HALF_WIDTH = 0.5  # = separación entre centros -> partición de la unidad

# Consecuentes en unidades de U (−1 = "empuja fuerte a la izquierda", +1 = "fuerte a la derecha").
#                     θ̇:  NG    NP     Z    PP    PG
_RULES = np.array(
    [
        [-1.0, -1.0, -1.0, -0.5, 0.0],  # e = NG  (muy inclinado a la izquierda)
        [-1.0, -1.0, -0.5, 0.0, 0.5],  # e = NP
        [-1.0, -0.5, 0.0, 0.5, 1.0],  # e = Z   (vertical: solo cuenta hacia dónde gira)
        [-0.5, 0.0, 0.5, 1.0, 1.0],  # e = PP
        [0.0, 0.5, 1.0, 1.0, 1.0],  # e = PG  (muy inclinado a la derecha)
    ]
)
_RULES.setflags(write=False)
RULE_TABLE = _RULES


def memberships(z: ArrayLike) -> np.ndarray:
    """Pertenencia de z (ya normalizado) a cada etiqueta: forma (..., 5), suma 1."""
    clipped = np.clip(np.asarray(z, dtype=np.float64), -1.0, 1.0)  # hombros en ±1
    return np.maximum(0.0, 1.0 - np.abs(clipped[..., None] - _CENTERS) / _HALF_WIDTH)


Scale = float | np.ndarray  # escalar para el controlador; array para evaluar lotes de candidatos


@dataclass(frozen=True)
class FuzzyGains:
    """Parámetros sintonizables. La tabla de reglas y la forma de las membresías NO lo son."""

    theta_scale: Scale  # E [rad]: error de ángulo que se considera "grande"
    theta_dot_scale: Scale  # D [rad/s]: velocidad angular que se considera "grande"
    force_scale: Scale  # U [N]: fuerza de "empujar fuerte"
    kp_x: Scale  # [rad/m]    lazo externo, como en el PID
    kd_x: Scale  # [rad·s/m]

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            array = np.asarray(value, dtype=np.float64)
            strictly_positive = name in ("theta_scale", "theta_dot_scale", "force_scale")
            valid = np.all(np.isfinite(array)) and np.all(array > 0 if strictly_positive else array >= 0)
            if not valid:
                raise ValueError(f"{name} debe ser finito y {'> 0' if strictly_positive else '>= 0'}, no {value}")

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}

    def equivalent_pd(self) -> dict[str, float]:
        """Ganancias del PD al que equivale el fuzzy en la celda central (ver docstring del módulo)."""
        kp, kd = self.force_scale / self.theta_scale, self.force_scale / self.theta_dot_scale
        return {"kp_theta": float(kp), "kd_theta": float(kd),
                "state_feedback_K": [float(kp * self.kp_x), float(kp * self.kd_x), float(kp), float(kd)]}


def fuzzy_force(
    state: ArrayLike, gains: FuzzyGains, *, force_limit: float, theta_ref_limit: float
) -> np.ndarray:
    """Ley de control pura y vectorizada: `state` (..., 4) -> F (...)."""
    s = np.asarray(state, dtype=np.float64)
    x, x_dot, theta, theta_dot = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    theta_ref = np.clip(-(gains.kp_x * x + gains.kd_x * x_dot), -theta_ref_limit, theta_ref_limit)

    mu_error = memberships((theta - theta_ref) / gains.theta_scale)  # 1) fuzzificación
    mu_rate = memberships(theta_dot / gains.theta_dot_scale)
    numerator = np.einsum("...i,...j,ij->...", mu_error, mu_rate, RULE_TABLE)  # 2)+3) reglas
    activation = np.einsum("...i,...j->...", mu_error, mu_rate)  # = 1 (partición de la unidad)
    return np.clip(gains.force_scale * numerator / activation, -force_limit, force_limit)


class CascadeFuzzy:
    """Controlador para `run_episode`. Sin estado interno (no hay término integral)."""

    action_mode = "continuous"

    def __init__(self, gains: FuzzyGains, params: CartPoleParams, *, theta_ref_limit: float | None = None) -> None:
        if theta_ref_limit is None:
            theta_ref_limit = DEFAULT_THETA_REF_FRACTION * params.theta_threshold_radians
        if not 0.0 < theta_ref_limit < params.theta_threshold_radians:
            raise ValueError(f"theta_ref_limit debe estar en (0, {params.theta_threshold_radians:.4f}) rad")
        self.gains = gains
        self.theta_ref_limit = theta_ref_limit
        self._force_limit = params.force_mag

    def reset(self) -> None:
        pass

    def __call__(self, state: np.ndarray) -> float:
        return float(fuzzy_force(state, self.gains, force_limit=self._force_limit, theta_ref_limit=self.theta_ref_limit))


def load_tuning_record(path: Path = TUNED_FUZZY_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No hay parámetros fuzzy sintonizados en {path}: ejecutar scripts/tune_fuzzy.py")
    return json.loads(path.read_text(encoding="utf-8"))


def load_tuned_fuzzy_gains(path: Path = TUNED_FUZZY_PATH) -> FuzzyGains:
    return FuzzyGains(**load_tuning_record(path)["gains"])
