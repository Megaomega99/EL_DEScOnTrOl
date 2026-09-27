r"""Control predictivo (MPC) lineal: el LQR + un horizonte finito + restricciones explícitas.

PROBLEMA (se resuelve en CADA paso de control, cada 20 ms)
----------------------------------------------------------
Dado el estado medido s_0:

    min   Σ_{i=0}^{N-1} ( s_iᵀ Q s_i + R F_i² )  +  s_Nᵀ P s_N  +  w Σ_i (σ_x,i + σ_θ,i)
    s.a.  s_{i+1} = A s_i + B F_i                 (modelo de Euler, el mismo del LQR)
          |F_i| ≤ F_max                            (actuador: restricción DURA)
          |x_i| ≤ x_lim + σ_x,i,  |θ_i| ≤ θ_lim + σ_θ,i,  σ ≥ 0   (estado: restricciones BLANDAS)

Se aplica solo F_0 y se repite en el paso siguiente con la nueva medida (horizonte deslizante).

DECISIONES Y POR QUÉ
--------------------
* Q, R, P heredados del LQR (paso 4), sin retocarlos. Así la ÚNICA diferencia con el
  LQR son las restricciones y el horizonte, y la comparación es limpia. Con coste
  terminal P (coste óptimo del LQR), si ninguna restricción está activa, F_0 = −K s_0
  EXACTAMENTE (principio de Bellman; lo verifica un test).
* θ_lim = 8.1°, y no 12°: margen de seguridad (constraint tightening). Con la
  restricción EXACTAMENTE en el límite de fallo, el plan roza 12° y el pequeño error del
  modelo lineal (~0.005°) lo cruza. En los 12 estados cerca de la frontera, el
  resultado es 6/12, igual que el LQR. En esta simulación ideal basta 0.01° de margen
  para 12/12. Se usan 8.1° (conservador a propósito) por coherencia con el criterio
  de validez del modelo lineal del LQR (cos θ ≥ 0.99), y por robustez ante
  perturbaciones y errores de modelo mayores que los de una simulación perfecta. El
  valor no se ajustó a los resultados.
* x_lim = 2.4 m, sin margen: el error del modelo en x es de segundo orden si θ se
  mantiene en la zona lineal. SUPUESTO (funcionó en los ensayos; no se ha demostrado).
* Restricciones de estado BLANDAS, con penalización L1 de peso w = 10⁴. Duras podrían
  hacer el problema infactible, por ejemplo si el estado inicial ya viola 8.1°, y el
  controlador no tendría respuesta. Con L1 y w mayor que los multiplicadores de Lagrange,
  la penalización es EXACTA: cuando el problema duro es factible, la solución es
  idéntica (Kerrigan & Maciejowski, 2000). Los tests comprueban ambas cosas.
* Horizonte N = 25 (0.5 s): el menor de {3, 5, 10, 25, 50} cuyo coste en lazo cerrado,
  en el peor caso de los estados de prueba, queda a menos del 1 % del de N = 50
  (docs/05_mpc.md). El criterio del 1 % es una elección.
* Solver: CVXPY (modelado) + Clarabel (punto interior). OSQP (ADMM), el habitual en
  MPC embebido por su warm start, dio 'optimal_inaccurate' o agotó iteraciones en
  estados difíciles, con un peor caso de 187–462 ms frente a 7.8 ms de Clarabel.
  En tiempo real importa el peor caso, no la media.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike

from cartpole_lab.controllers.lqr import BRYSON_COS_TOLERANCE, LQRDesign, design_default_lqr
from cartpole_lab.params import CartPoleParams


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 25
    theta_constraint_rad: float | None = None  # None -> arccos(1 − tolerancia de Bryson) = 8.1°
    x_constraint_m: float | None = None  # None -> x_threshold (2.4 m)
    slack_weight: float | None = 1e4  # None -> restricciones de estado DURAS (solo para tests)
    solver: str = "CLARABEL"
    solver_options: tuple[tuple[str, Any], ...] = ()  # p. ej. (("eps_abs", 1e-7),) para OSQP

    def theta_constraint(self, params: CartPoleParams) -> float:
        if self.theta_constraint_rad is not None:
            return self.theta_constraint_rad
        return float(np.arccos(1.0 - BRYSON_COS_TOLERANCE))

    def x_constraint(self, params: CartPoleParams) -> float:
        return params.x_threshold if self.x_constraint_m is None else self.x_constraint_m

    def validate(self, params: CartPoleParams) -> None:
        if self.horizon < 1:
            raise ValueError(f"El horizonte debe ser >= 1, no {self.horizon}")
        if not 0.0 < self.theta_constraint(params) <= params.theta_threshold_radians:
            raise ValueError("La restricción de θ debe estar en (0, θ_threshold]")
        if not 0.0 < self.x_constraint(params) <= params.x_threshold:
            raise ValueError("La restricción de x debe estar en (0, x_threshold]")
        if self.slack_weight is not None and not self.slack_weight > 0.0:
            raise ValueError("El peso de las holguras debe ser > 0 (o None para restricciones duras)")

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))


class LinearMPC:
    """Controlador para `run_episode`. Construye el problema UNA vez (parámetro s_0, CVXPY/DPP)
    y en cada llamada solo cambia el valor de s_0 y re-resuelve."""

    action_mode = "continuous"

    def __init__(self, params: CartPoleParams, config: MPCConfig | None = None) -> None:
        self.config = config if config is not None else MPCConfig()
        self.config.validate(params)
        self.design: LQRDesign = design_default_lqr(params)
        self._force_limit = params.force_mag
        self._build(params)
        self.solve_times: list[float] = []
        self.last_plan: dict[str, Any] | None = None
        self._solve(np.zeros(4))  # precompilación: la 1.ª resolución incluye la canonicalización
        self.reset()

    def _build(self, params: CartPoleParams) -> None:
        d, N = self.design, self.config.horizon
        if np.count_nonzero(d.Q - np.diag(np.diag(d.Q))):
            raise ValueError("Se espera Q diagonal (pesos de Bryson)")
        self._s0 = cp.Parameter(4, name="s0")
        self._forces = cp.Variable(N, name="F")
        self._states = cp.Variable((N + 1, 4), name="s")
        s, u = self._states, self._forces

        stage = cp.sum_squares(s[:N] @ np.diag(np.sqrt(np.diag(d.Q)))) + d.R[0, 0] * cp.sum_squares(u)
        terminal = cp.sum_squares(s[N] @ np.linalg.cholesky(d.P))  # sᵀPs = ‖s L‖², P = L Lᵀ
        constraints = [s[0] == self._s0, cp.abs(u) <= self._force_limit]
        constraints += [s[i + 1] == d.A @ s[i] + d.B * u[i] for i in range(N)]

        limits = (self.config.x_constraint(params), self.config.theta_constraint(params))
        if self.config.slack_weight is None:
            self._slack = None
            self._state_constraints = [cp.abs(s[1:, 0]) <= limits[0], cp.abs(s[1:, 2]) <= limits[1]]
            penalty = 0.0
        else:
            self._slack = cp.Variable((N, 2), nonneg=True, name="sigma")
            self._state_constraints = [
                cp.abs(s[1:, 0]) <= limits[0] + self._slack[:, 0],
                cp.abs(s[1:, 2]) <= limits[1] + self._slack[:, 1],
            ]
            penalty = self.config.slack_weight * cp.sum(self._slack)
        self._problem = cp.Problem(cp.Minimize(stage + terminal + penalty), constraints + self._state_constraints)

    def reset(self) -> None:
        self.solve_times = []
        self.last_plan = None

    def _solve(self, state: np.ndarray) -> float:
        self._s0.value = state
        start = time.perf_counter()
        self._problem.solve(solver=self.config.solver, **dict(self.config.solver_options))
        self.solve_times.append(time.perf_counter() - start)
        if self._problem.status != cp.OPTIMAL:
            # Nunca se devuelve una fuerza de un problema no resuelto: sería silenciosamente arbitraria.
            raise RuntimeError(f"MPC: el solver {self.config.solver} devolvió '{self._problem.status}'")
        duals = [np.max(np.abs(c.dual_value)) for c in self._state_constraints]
        self.last_plan = {
            "states": np.array(self._states.value),
            "forces": np.array(self._forces.value),
            "max_slack": 0.0 if self._slack is None else float(np.max(self._slack.value)),
            "max_state_constraint_dual": float(max(duals)),
        }
        return float(np.clip(self._forces.value[0], -self._force_limit, self._force_limit))

    def __call__(self, state: ArrayLike) -> float:
        s = np.asarray(state, dtype=np.float64)
        if s.shape != (4,):
            raise ValueError(f"El estado debe tener forma (4,), no {s.shape}")
        return self._solve(s)
