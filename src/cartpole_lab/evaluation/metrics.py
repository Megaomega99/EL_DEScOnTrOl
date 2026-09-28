"""Métricas por episodio de la evaluación comparativa (definiciones en docs/12_protocolo_evaluacion.md §3)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cartpole_lab.evaluation.protocol import THETA_BAND_RAD, X_BAND_M
from cartpole_lab.rollout import Trajectory

# Códigos de resultado para las mallas de región de atracción.
FAILED, SURVIVED, STABILIZED = 0, 1, 2


def settling_time(traj: Trajectory, *, theta_band: float = THETA_BAND_RAD, x_band: float | None = X_BAND_M,
                  from_step: int = 0) -> float | None:
    """Menor t (desde `from_step`) a partir del cual el estado queda DENTRO de la banda hasta el final.

    None si el episodio no sobrevive o si su último estado está fuera de la banda ("no asentado").
    `x_band=None` ignora la posición: tiempo de asentamiento del poste solo.
    """
    if traj.termination_reason != "time_limit":
        return None
    states = traj.states[from_step:]
    inside = np.abs(states[:, 2]) <= theta_band
    if x_band is not None:
        inside &= np.abs(states[:, 0]) <= x_band
    if not inside[-1]:
        return None
    outside = np.flatnonzero(~inside)
    first_inside_for_good = 0 if outside.size == 0 else int(outside[-1]) + 1
    return float(traj.times[from_step + first_inside_for_good] - traj.times[from_step])


def episode_metrics(traj: Trajectory, *, impulse_step: int | None = None) -> dict:
    """Todas las métricas de un episodio (nominal o con impulso)."""
    metrics = {
        "survived": traj.termination_reason == "time_limit",
        "reason": traj.termination_reason,
        "settling_time_s": settling_time(traj),
        "pole_settling_time_s": settling_time(traj, x_band=None),
        "effort_abs_Ns": traj.control_effort_abs,
        "effort_sq_N2s": traj.control_effort_sq,
        "fraction_at_force_limit": traj.fraction_at_force_limit,
        "cost": traj.total_cost,
    }
    if impulse_step is not None:
        # El pulso actúa en la transición states[k] -> states[k+1]: states[k] es aún el estado PREVIO al
        # impulso. Los picos se miden desde k+1; el tiempo de recuperación se cuenta desde t_k (inicio del pulso).
        after = traj.states[impulse_step + 1:] if traj.n_steps > impulse_step else traj.states[-1:]
        metrics.update(
            recovery_time_s=settling_time(traj, x_band=None, from_step=impulse_step) if traj.n_steps > impulse_step else None,
            peak_abs_theta_after_deg=float(np.degrees(np.max(np.abs(after[:, 2])))),
            peak_abs_x_after_m=float(np.max(np.abs(after[:, 0]))),
            failed_before_impulse=traj.n_steps <= impulse_step and traj.terminated,
        )
    return metrics


def grid_outcome(traj: Trajectory) -> int:
    if traj.termination_reason != "time_limit":
        return FAILED
    return STABILIZED if settling_time(traj) is not None else SURVIVED


def _mean_std(values: Sequence[float]) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"mean": None, "std": None, "median": None, "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "median": float(np.median(v)), "n": int(v.size)}


def aggregate_episodes(episodes: Sequence[dict]) -> dict:
    """Resumen de un conjunto de episodios de UN controlador (estadística entre condiciones iniciales)."""
    summary = {
        "n_episodes": len(episodes),
        "survival_rate": float(np.mean([e["survived"] for e in episodes])),
        "settled_rate": float(np.mean([e["settling_time_s"] is not None for e in episodes])),
        "pole_settled_rate": float(np.mean([e["pole_settling_time_s"] is not None for e in episodes])),
        "settling_time_s": _mean_std([e["settling_time_s"] for e in episodes]),
        "pole_settling_time_s": _mean_std([e["pole_settling_time_s"] for e in episodes]),
        "effort_abs_Ns": _mean_std([e["effort_abs_Ns"] for e in episodes]),
        "fraction_at_force_limit": _mean_std([e["fraction_at_force_limit"] for e in episodes]),
        "cost": _mean_std([e["cost"] for e in episodes if e["survived"]]),  # coste solo comparable si sobrevive
    }
    if "recovery_time_s" in episodes[0]:
        summary["recovery_time_s"] = _mean_std([e["recovery_time_s"] for e in episodes])
        summary["peak_abs_theta_after_deg"] = _mean_std([e["peak_abs_theta_after_deg"] for e in episodes if e["survived"]])
        summary["failed_before_impulse"] = int(sum(e["failed_before_impulse"] for e in episodes))
    return summary


def across_seeds(per_seed: Sequence[dict], keys: Sequence[str]) -> dict:
    """Media ± desviación típica ENTRE SEMILLAS de las cantidades escalares de cada semilla (RL)."""
    out = {"n_seeds": len(per_seed)}
    for key in keys:
        values = []
        for summary in per_seed:
            value = summary[key]
            values.append(value["mean"] if isinstance(value, dict) else value)
        out[key] = _mean_std(values)
    return out
