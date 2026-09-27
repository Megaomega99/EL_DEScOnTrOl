"""Paso 4: diseña el LQR, compara discretizaciones y criterios, verifica en Gymnasium.

Uso:  python scripts/design_lqr.py      (segundos: el LQR tiene solución cerrada)

Salidas:
    results/tuning/lqr.json               A, B, Q, R, K, P, comparaciones, sanity check, versiones
    results/figures/04_lqr_vs_pid.png     LQR y PID desde el mismo estado inicial difícil
"""

from __future__ import annotations

import json
import sys

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from cartpole_lab import CartPoleTask, load_params, run_episode
from cartpole_lab.controllers.lqr import (
    LQRController,
    cost_matrix_of_gain,
    design_default_lqr,
    lqr_criterion,
    solve_clqr,
    solve_dlqr,
)
from cartpole_lab.controllers.pid import CascadePID, load_tuned_gains
from cartpole_lab.controllers.pid_tuning import TuningConfig, itae_cost
from cartpole_lab.linearization import continuous_linearization, euler_discretization, zoh_discretization
from cartpole_lab.paths import FIGURES_DIR, TUNING_DIR
from cartpole_lab.plotting import METHOD_COLORS, MUTED, TEXT_SECONDARY, apply_style
from cartpole_lab.sanity import REFERENCE_HARD_INITIAL_STATE, is_stabilized, sanity_statistics

RECORD_PATH = TUNING_DIR / "lqr.json"
FIGURE_PATH = FIGURES_DIR / "04_lqr_vs_pid.png"
# CI para cruzar criterios: misma caja que la sintonización del PID pero OTRA semilla,
# de modo que son fuera de muestra tanto para el PID como para el LQR.
CROSSOVER_SEED = 2027
R_SWEEP_FACTORS = (0.1, 1.0, 10.0)
Q_X_SWEEP_FACTORS = (0.25, 1.0, 4.0, 16.0)


def run(controller, initial_state):
    task = CartPoleTask(action_mode="continuous")
    try:
        return run_episode(task, controller, initial_state=initial_state)
    finally:
        task.close()


def discretization_comparison(params, design) -> list[dict]:
    """¿Importa diseñar sobre Euler (el simulador) o sobre la física exacta (ZOH / continuo)?"""
    A_c, B_c = continuous_linearization(params)
    A_z, B_z = zoh_discretization(A_c, B_c, params.tau)
    gains = {
        "euler (simulador)": design.K,
        "zoh exacta": solve_dlqr(A_z, B_z, design.Q, design.R).K,
        "continuo (CARE)": solve_clqr(A_c, B_c, design.Q, design.R),
    }
    optimal_trace = np.trace(design.P)
    rows = []
    for name, K in gains.items():
        P_K = cost_matrix_of_gain(design.A, design.B, design.Q, design.R, K)
        closed_loop = design.A - design.B[:, None] @ K
        traj = run(LQRController(K, params), REFERENCE_HARD_INITIAL_STATE)
        rows.append({
            "model": name,
            "K": K.ravel().tolist(),
            "spectral_radius_on_simulator": float(np.max(np.abs(np.linalg.eigvals(closed_loop)))),
            "suboptimality_trace_pct": float(100 * (np.trace(P_K) / optimal_trace - 1)),
            "lqr_criterion_hard_ic": lqr_criterion(traj, design.Q, design.R),
        })
    rows.append({"model_discrepancy_frobenius": float(np.linalg.norm(A_z - design.A))})
    return rows


def r_sweep(params, design) -> list[dict]:
    """R es la perilla del LQR: más caro el esfuerzo -> ganancias menores, respuesta más lenta."""
    rows = []
    for factor in R_SWEEP_FACTORS:
        d = solve_dlqr(design.A, design.B, design.Q, factor * design.R)
        traj = run(LQRController(d.K, params), REFERENCE_HARD_INITIAL_STATE)
        rows.append({
            "R_factor": factor,
            "K": d.K.ravel().tolist(),
            "spectral_radius": d.spectral_radius,
            "dominant_time_constant_s": d.dominant_time_constant(params.tau),
            "hard_ic_stabilized": is_stabilized(traj, params),
            "hard_ic_peak_abs_force_N": float(np.max(np.abs(traj.forces))),
            "hard_ic_effort_abs_Ns": traj.control_effort_abs,
            "hard_ic_fraction_at_force_limit": traj.fraction_at_force_limit,
            "hard_ic_max_abs_x_m": float(np.max(np.abs(traj.states[:, 0]))),
        })
    return rows


def q_x_sweep(params, design) -> list[dict]:
    """Contraste con el barrido de R: el modo lento (carro) lo fija el compromiso x–θ de Q."""
    rows = []
    for factor in Q_X_SWEEP_FACTORS:
        Q = design.Q.copy()
        Q[0, 0] *= factor
        d = solve_dlqr(design.A, design.B, Q, design.R)
        rows.append({
            "q_x_factor": factor,
            "dominant_time_constant_s": d.dominant_time_constant(params.tau),
            "abs_eigenvalues": sorted(float(v) for v in np.abs(d.closed_loop_eigenvalues)),
        })
    return rows


def criteria_crossover(params, design) -> dict:
    """Cada método con el criterio del otro, sobre 32 CI nuevas (fuera de muestra para ambos)."""
    config = TuningConfig()
    box = np.asarray(config.initial_state_box)
    initial_states = np.random.default_rng(CROSSOVER_SEED).uniform(-box, box, size=(config.n_initial_states, 4))
    controllers = {
        "PID": CascadePID(load_tuned_gains(), params, theta_ref_limit=config.theta_ref_limit(params)),
        "LQR": LQRController(design.K, params),
    }
    result, per_ic = {}, {}
    for name, controller in controllers.items():
        trajectories = [run(controller, s0) for s0 in initial_states]
        itae = np.array([itae_cost(t, params, config) for t in trajectories])
        quad = np.array([lqr_criterion(t, design.Q, design.R) for t in trajectories])
        per_ic[name] = (itae, quad)
        result[name] = {
            "n_initial_states": len(trajectories),
            "all_survived": all(t.termination_reason == "time_limit" for t in trajectories),
            "itae_criterion_mean": float(itae.mean()), "itae_criterion_std": float(itae.std(ddof=1)),
            "lqr_criterion_mean": float(quad.mean()), "lqr_criterion_std": float(quad.std(ddof=1)),
        }
    # Comparación PAREADA (misma CI para ambos): la variabilidad entre CI es mucho mayor que la
    # diferencia entre métodos, así que las medias ± std por separado se solapan y engañan.
    (pid_itae, pid_quad), (lqr_itae, lqr_quad) = per_ic["PID"], per_ic["LQR"]
    result["paired"] = {
        "pid_better_on_itae_count": int(np.sum(pid_itae < lqr_itae)),
        "lqr_better_on_lqr_criterion_count": int(np.sum(lqr_quad < pid_quad)),
        "itae_relative_difference_lqr_vs_pid_mean_pct": float(100 * np.mean(lqr_itae / pid_itae - 1)),
        "lqr_criterion_relative_difference_pid_vs_lqr_mean_pct": float(100 * np.mean(pid_quad / lqr_quad - 1)),
    }
    return result


def plot_lqr_vs_pid(params, design) -> None:
    config = TuningConfig()
    trajectories = {
        "PID": run(CascadePID(load_tuned_gains(), params, theta_ref_limit=config.theta_ref_limit(params)),
                   REFERENCE_HARD_INITIAL_STATE),
        "LQR": run(LQRController(design.K, params), REFERENCE_HARD_INITIAL_STATE),
    }
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    panels = [(0, "x [m]", "Posición del carro", 1.0), (2, "θ [°]", "Ángulo del poste", np.degrees(1.0))]
    for ax, (index, ylabel, title, scale) in zip(axes[:2], panels):
        for name, traj in trajectories.items():
            ax.plot(traj.times, scale * traj.states[:, index], color=METHOD_COLORS[name], label=name)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].legend(loc="upper right")
    for name, traj in trajectories.items():
        axes[2].step(traj.times[:-1], traj.forces, where="post", color=METHOD_COLORS[name])
    for sign in (+1, -1):
        axes[2].axhline(sign * params.force_mag, color=MUTED, linewidth=1, linestyle=":")
    axes[2].set_ylabel("F [N]")
    axes[2].set_xlabel("t [s]")
    axes[2].set_title("Fuerza del actuador (saturación en ±10 N)", loc="left", fontsize=11)
    x0, v0, th0, w0 = REFERENCE_HARD_INITIAL_STATE
    fig.suptitle(
        "LQR (Bryson, modelo de Euler) frente a PID en cascada (ITAE)\n"
        f"Misma CI: x={x0} m, ẋ={v0} m/s, θ={th0} rad, θ̇={w0} rad/s",
        fontsize=12, color=TEXT_SECONDARY,
    )
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def main() -> int:
    params = load_params()
    design = design_default_lqr(params)
    A_c, B_c = continuous_linearization(params)
    record = {
        "controller": "LQRController (F = -K s, saturado a ±F_max)",
        "model": "euler_discretization(continuous_linearization(params), tau)",
        "A_c": A_c.tolist(), "B_c": B_c.tolist(), "A": design.A.tolist(), "B": design.B.tolist(),
        "Q": design.Q.tolist(), "R": design.R.tolist(), "K": design.K.tolist(), "P": design.P.tolist(),
        "closed_loop_abs_eigenvalues": sorted(float(v) for v in np.abs(design.closed_loop_eigenvalues)),
        "spectral_radius": design.spectral_radius,
        "dominant_time_constant_s": design.dominant_time_constant(params.tau),
        "discretization_comparison": discretization_comparison(params, design),
        "R_sweep": r_sweep(params, design),
        "q_x_sweep": q_x_sweep(params, design),
        "criteria_crossover_out_of_sample": criteria_crossover(params, design),
        "sanity_gymnasium_default_ic": sanity_statistics(LQRController(design.K, params), params),
        "hard_ic_stabilized": is_stabilized(run(LQRController(design.K, params), REFERENCE_HARD_INITIAL_STATE), params),
        "versions": {"gymnasium": gym.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "generated_by": "scripts/design_lqr.py",
    }
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_lqr_vs_pid(params, design)

    summary = {k: record[k] for k in ("K", "spectral_radius", "dominant_time_constant_s", "hard_ic_stabilized")}
    for key in ("discretization_comparison", "R_sweep", "q_x_sweep", "criteria_crossover_out_of_sample", "sanity_gymnasium_default_ic"):
        summary[key] = record[key]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nGuardado {RECORD_PATH}\nFigura {FIGURE_PATH}")

    ok = design.spectral_radius < 1.0 and record["sanity_gymnasium_default_ic"]["stabilized_rate"] == 1.0
    if not ok:
        print("ATENCIÓN: el LQR NO supera el sanity check; revisar antes de continuar.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
