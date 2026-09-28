"""¿Por qué algunas políticas de RL sobreviven a impulsos que tumban a PID, LQR y MPC? (docs/13 §3)

Hipótesis: los clásicos (y el actor-crítico) están en el equilibrio cuando llega el impulso, en t = 5 s, y un
impulso de 1.75 N·s los tumba con cualquier signo. Las políticas que OSCILAN, en cambio, reciben el impulso en
una fase cualquiera de su oscilación. Si llega cuando el poste ya gira en el sentido contrario al que lo
empuja el impulso, lo compensa en parte. Si es así, esas supervivencias no son robustez: son el momento en
que llega el impulso.

Comprobación: para cada episodio con J ≥ 1.75 N·s, el estado justo antes del impulso, proyectado sobre su
signo s (±1). Un impulso +x acelera el carro hacia +x y hace girar el poste hacia −θ (θ̈ < 0 con F > 0,
docs/01), así que compensa si s·θ̇ > 0 (el poste giraba hacia +θ) y s·ẋ < 0 (el carro iba hacia −x).

Uso:  python scripts/analyze_impulse_phase.py      (~1 min; escribe results/metrics/impulse_phase.json)
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from cartpole_lab import CartPoleTask, ForcePulse, load_params, run_episode
from cartpole_lab.evaluation.methods import PolicySpec, all_specs, build_controller
from cartpole_lab.evaluation.protocol import IMPULSE_STEP, NOMINAL_SEEDS, impulse_sign
from cartpole_lab.paths import RESULTS_DIR

OUTPUT = RESULTS_DIR / "metrics" / "impulse_phase.json"
LARGE_IMPULSES_NS = (1.75, 2.0)  # por encima del límite de PID/LQR/MPC (entre 1.5 y 1.75 N·s)
# Sin el MPC: en comparison.json su supervivencia, recuperación y picos tras el impulso coinciden con los del
# LQR para todos los J (y tarda ~100 veces más en simular).
METHODS = ("PID", "LQR", "Fuzzy", "Q-learning", "SARSA", "DQN", "Actor-crítico")
SINGLE_THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _episodes(spec: PolicySpec) -> list[dict]:
    """Estado proyectado justo antes del impulso y resultado, para cada CI y cada J grande."""
    params = load_params()
    controller = build_controller(spec, params)
    rows = []
    for J in LARGE_IMPULSES_NS:
        for i, seed in enumerate(NOMINAL_SEEDS):
            sign = impulse_sign(i)
            task = CartPoleTask(action_mode="continuous", disturbance=ForcePulse(IMPULSE_STEP, 1, sign * J / params.tau))
            traj = run_episode(task, controller, seed=seed)
            if traj.n_steps <= IMPULSE_STEP:  # falló antes del impulso: no informa sobre la fase
                continue
            x, x_dot, theta, theta_dot = traj.states[IMPULSE_STEP]
            rows.append({"J": J, "survived": traj.termination_reason == "time_limit", "reason": traj.termination_reason,
                         "s_theta_deg": float(np.degrees(sign * theta)), "s_theta_dot": float(sign * theta_dot),
                         "s_x": float(sign * x), "s_x_dot": float(sign * x_dot)})
    return rows


def _median(rows: list[dict], key: str) -> float | None:
    return float(np.median([r[key] for r in rows])) if rows else None


def _rate(rows: list[dict]) -> float | None:
    return float(np.mean([r["survived"] for r in rows])) if rows else None


def summarize(method: str, rows: list[dict]) -> dict:
    alive = [r for r in rows if r["survived"]]
    dead = [r for r in rows if not r["survived"]]
    compensating = [r for r in rows if r["s_theta_dot"] > 0]  # el impulso frena el giro del poste
    opposing = [r for r in rows if r["s_theta_dot"] <= 0]
    return {
        "method": method, "n_episodes": len(rows), "n_survived": len(alive),
        "failure_reasons": dict(Counter(r["reason"] for r in dead)),
        "median_before_impulse": {k: {"survived": _median(alive, k), "failed": _median(dead, k)}
                                  for k in ("s_theta_deg", "s_theta_dot", "s_x", "s_x_dot")},
        "survival_if_impulse_brakes_pole_rotation": {"n": len(compensating), "rate": _rate(compensating)},
        "survival_otherwise": {"n": len(opposing), "rate": _rate(opposing)},
    }


def main() -> int:
    specs = [s for s in all_specs(include_supplementary=False) if s.method in METHODS]
    os.environ.update(SINGLE_THREAD_ENV)
    with ProcessPoolExecutor(max_workers=12) as pool:
        per_spec = list(pool.map(_episodes, specs, chunksize=1))
    by_method: dict[str, list[dict]] = {}
    for spec, rows in zip(specs, per_spec):
        by_method.setdefault(spec.method, []).extend(rows)
    summary = {m: summarize(m, by_method[m]) for m in METHODS}
    for m, s in summary.items():
        med = s["median_before_impulse"]
        print(f"{m:14s} sobreviven {s['n_survived']:3d}/{s['n_episodes']:3d} | mediana s·θ̇ sob/fall "
              f"{med['s_theta_dot']['survived'] or float('nan'):5.2f}/{med['s_theta_dot']['failed']:5.2f} rad/s | "
              f"supervivencia si el impulso frena el giro: {s['survival_if_impulse_brakes_pole_rotation']['rate'] or 0:.2f}"
              f" (n={s['survival_if_impulse_brakes_pole_rotation']['n']}), si no: {s['survival_otherwise']['rate'] or 0:.2f}"
              f" (n={s['survival_otherwise']['n']}) | fallos {s['failure_reasons']}")
    record = {"impulses_Ns": list(LARGE_IMPULSES_NS), "impulse_step": IMPULSE_STEP,
              "projection": "s·(estado) con s = signo del impulso, justo antes del pulso", "methods": summary}
    OUTPUT.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Guardado {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
