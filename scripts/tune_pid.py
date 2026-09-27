"""Paso 3: sintoniza el PID en cascada, lo verifica en Gymnasium y guarda ganancias y figura.

Uso:
    python scripts/tune_pid.py                 sintoniza (~2 min) y verifica
    python scripts/tune_pid.py --skip-tuning   reutiliza las ganancias guardadas; solo recalcula
                                               diagnósticos, sanity check y figura

Salidas:
    results/tuning/pid.json               ganancias, configuración, ejecuciones del optimizador,
                                          diagnósticos de estabilidad, sanity check y versiones
    results/figures/03_pid_respuesta.png  respuesta desde el estado inicial difícil de referencia
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from cartpole_lab import CartPoleTask, load_params, run_episode
from cartpole_lab.controllers.pid import TUNED_GAINS_PATH, CascadePID, PIDGains, load_tuning_record
from cartpole_lab.controllers.pid_tuning import TuningConfig, closed_loop_eigenvalues, tune_pid
from cartpole_lab.paths import FIGURES_DIR
from cartpole_lab.plotting import METHOD_COLORS, MUTED, TEXT_SECONDARY, apply_style
from cartpole_lab.sanity import REFERENCE_HARD_INITIAL_STATE, sanity_statistics

FIGURE_PATH = FIGURES_DIR / "03_pid_respuesta.png"


def stability_diagnostics(gains: PIDGains, params, limit: float) -> dict:
    """Análisis lineal local en el equilibrio (ver closed_loop_eigenvalues)."""
    full = closed_loop_eigenvalues(gains, params, limit)
    physical = closed_loop_eigenvalues(gains, params, limit, include_integral=False)
    rho_physical = float(np.max(np.abs(physical)))
    return {
        "spectral_radius_full": float(np.max(np.abs(full))),
        "abs_eigenvalues_full": sorted(float(v) for v in np.abs(full)),
        "spectral_radius_physical_modes": rho_physical,
        "dominant_time_constant_s": float(-params.tau / np.log(rho_physical)),
        # Con Ki = 0 y sin saturación, la cascada es F = K·s: misma forma que el LQR.
        "equivalent_state_feedback_K": [
            gains.kp_theta * gains.kp_x, gains.kp_theta * gains.kd_x, gains.kp_theta, gains.kd_theta,
        ],
        "pd_offset_per_newton_of_constant_force_m": float(-1.0 / (gains.kp_theta * gains.kp_x)),
    }


def plot_response(gains: PIDGains, params, limit: float) -> None:
    task = CartPoleTask(action_mode="continuous")
    try:
        traj = run_episode(
            task, CascadePID(gains, params, theta_ref_limit=limit), initial_state=REFERENCE_HARD_INITIAL_STATE
        )
    finally:
        task.close()
    t, s = traj.times, traj.states
    theta_ref = np.clip(-(gains.kp_x * s[:, 0] + gains.kd_x * s[:, 1]), -limit, limit)
    color = METHOD_COLORS["PID"]

    apply_style()
    fig, (ax_x, ax_th, ax_f) = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    ax_x.plot(t, s[:, 0], color=color)
    ax_x.set_ylabel("x [m]")
    ax_x.set_title("Posición del carro (objetivo secundario)", loc="left", fontsize=11)

    ax_th.plot(t, np.degrees(s[:, 2]), color=color, label="θ medido")
    # θ_ref es una referencia interna, no otro método: tinta neutra, no un color categórico.
    ax_th.plot(t, np.degrees(theta_ref), color=TEXT_SECONDARY, linestyle="--", linewidth=1.6,
               label="θ_ref pedido por el lazo de posición")
    for sign in (+1, -1):
        ax_th.axhline(sign * np.degrees(limit), color=MUTED, linewidth=1, linestyle=":")
    ax_th.set_ylabel("θ [°]")
    ax_th.set_title(f"Ángulo del poste (variable controlada); límite de θ_ref = ±{np.degrees(limit):.0f}°",
                    loc="left", fontsize=11)
    ax_th.legend(loc="upper right")

    ax_f.step(t[:-1], traj.forces, where="post", color=color)
    for sign in (+1, -1):
        ax_f.axhline(sign * params.force_mag, color=MUTED, linewidth=1, linestyle=":")
    ax_f.set_ylabel("F [N]")
    ax_f.set_xlabel("t [s]")
    ax_f.set_title("Fuerza del actuador (saturación en ±10 N)", loc="left", fontsize=11)

    x0, v0, th0, w0 = REFERENCE_HARD_INITIAL_STATE
    fig.suptitle(
        "PID en cascada sintonizado (ITAE + evolución diferencial)\n"
        f"CI: x={x0} m, ẋ={v0} m/s, θ={th0} rad, θ̇={w0} rad/s  (esquina más difícil de la caja de sintonización)",
        fontsize=12, color=TEXT_SECONDARY,
    )
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def run_tuning(config: TuningConfig, params) -> tuple[PIDGains, dict]:
    start = time.perf_counter()
    result = tune_pid(config, params)
    elapsed = time.perf_counter() - start
    print(f"Sintonización: {len(result.runs)} ejecuciones de evolución diferencial en {elapsed:.1f} s")
    for run in result.runs:
        print(f"  semilla {run.de_seed}: J={run.cost:.5f} nit={run.nit} nfev={run.nfev} convergió={run.converged} | "
              + ", ".join(f"{k}={v:.4g}" for k, v in run.gains.to_dict().items()))
    best = result.best
    tuning = {
        "method": "differential_evolution (scipy), criterio ITAE + esfuerzo, ver pid_tuning.py",
        "config": config.to_dict(),
        "selected_de_seed": best.de_seed,
        "cost": best.cost,
        "runs": [
            {"de_seed": r.de_seed, "cost": r.cost, "nit": r.nit, "nfev": r.nfev,
             "converged": r.converged, "message": r.message, "gains": r.gains.to_dict()}
            for r in result.runs
        ],
        "wall_time_s": elapsed,
    }
    return best.gains, tuning


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-tuning", action="store_true", help="reutilizar las ganancias de pid.json")
    args = parser.parse_args()

    params = load_params()
    config = TuningConfig()
    limit = config.theta_ref_limit(params)
    if args.skip_tuning:
        previous = load_tuning_record()
        if previous["tuning"]["config"] != config.to_dict():
            raise SystemExit("La configuración cambió desde la última sintonización: ejecutar sin --skip-tuning")
        gains, tuning = PIDGains(**previous["gains"]), previous["tuning"]
    else:
        gains, tuning = run_tuning(config, params)

    stability = stability_diagnostics(gains, params, limit)
    sanity = sanity_statistics(CascadePID(gains, params, theta_ref_limit=limit), params)
    print(json.dumps({"stability": stability, "sanity": sanity}, indent=2, ensure_ascii=False))

    record = {
        "controller": "CascadePID",
        "gains": gains.to_dict(),
        "theta_ref_limit_rad": limit,
        "tuning": tuning,
        "diagnostics": {"stability": stability, "sanity_gymnasium_default_ic": sanity},
        "versions": {"gymnasium": gym.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "generated_by": "scripts/tune_pid.py",
    }
    TUNED_GAINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_GAINS_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_response(gains, params, limit)
    print(f"\nGuardado {TUNED_GAINS_PATH}\nFigura {FIGURE_PATH}")

    ok = (
        stability["spectral_radius_full"] <= 1.0 + 1e-9
        and stability["spectral_radius_physical_modes"] < 1.0
        and sanity["stabilized_rate"] == 1.0
    )
    if not ok:
        print("ATENCIÓN: el PID sintonizado NO supera el sanity check; revisar antes de continuar.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
