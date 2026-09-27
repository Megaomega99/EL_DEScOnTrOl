"""Paso 5: MPC lineal. Elige horizonte y margen con datos, compara solvers y con PID/LQR,
y verifica en Gymnasium.

Uso:  python scripts/design_mpc.py      (~6 min: una optimización cada 20 ms simulados)

Salidas:
    results/tuning/mpc.json                 configuración, barridos, comparación de solvers,
                                            estados cerca de la frontera, sanity check, versiones
    results/figures/05_mpc_vs_lqr_pid.png   PID, LQR y MPC desde un estado donde el LQR falla
"""

from __future__ import annotations

import json
import sys

import cvxpy as cp
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cartpole_lab import CartPoleTask, load_params, run_episode
from cartpole_lab.controllers.lqr import LQRController, design_default_lqr, lqr_criterion
from cartpole_lab.controllers.mpc import LinearMPC, MPCConfig
from cartpole_lab.controllers.pid import CascadePID, load_tuned_gains
from cartpole_lab.paths import FIGURES_DIR, TUNING_DIR
from cartpole_lab.plotting import METHOD_COLORS, MUTED, TEXT_SECONDARY, apply_style
from cartpole_lab.sanity import NEAR_BOUNDARY_INITIAL_STATES, is_stabilized, sanity_statistics

RECORD_PATH = TUNING_DIR / "mpc.json"
FIGURE_PATH = FIGURES_DIR / "05_mpc_vs_lqr_pid.png"
HORIZONS = (3, 5, 10, 25, 50)
REFERENCE_HORIZON = 50
MARGIN_SWEEP_DEG = (12.0, 11.99, 10.0, 8.11)
FIGURE_INITIAL_STATE = (1.0, 1.0, 0.0, 0.0)  # carro desplazado y lanzado: el LQR falla aquí
# Instancias de QP difíciles (restricciones blandas muy activas) + dos fáciles.
SOLVER_BENCHMARK_STATES = ((0, 0, 0.19, 1.0), (0, 0, 0.205, 0.5), (0, 0, 0.16, 1.5), (1.9, 2.0, 0, 0),
                           (0.01, 0, 0.01, 0), (0.5, 0.2, 0.1, 0.2))
OSQP_ACCURATE = (("eps_abs", 1e-7), ("eps_rel", 1e-7), ("polish", True), ("max_iter", 20000), ("warm_start", True))


def run(controller, initial_state):
    task = CartPoleTask(action_mode="continuous")
    try:
        traj = run_episode(task, controller, initial_state=initial_state)
    finally:
        task.close()
    times = list(getattr(controller, "solve_times", []))
    return traj, times


def evaluate_on_boundary_states(controller, design) -> dict:
    outcomes, costs, times, max_theta = [], [], [], []
    for s0 in NEAR_BOUNDARY_INITIAL_STATES:
        traj, t = run(controller, s0)
        outcomes.append(is_stabilized(traj, load_params()))
        costs.append(lqr_criterion(traj, design.Q, design.R) if not traj.terminated else float("nan"))
        times += t
        max_theta.append(float(np.degrees(np.max(np.abs(traj.states[1:, 2])))))
    return {"stabilized": outcomes, "n_stabilized": int(sum(outcomes)), "costs": costs,
            "solve_times_s": times, "max_abs_theta_deg": max(max_theta)}


def timing_summary(times: list[float]) -> dict:
    if not times:
        return {}
    ms = np.asarray(times) * 1e3
    return {"mean_ms": float(ms.mean()), "p99_ms": float(np.percentile(ms, 99)), "max_ms": float(ms.max()),
            "n_solves": int(ms.size), "fraction_of_20ms_budget_worst_case": float(ms.max() / 20.0)}


def horizon_sweep(params, design) -> list[dict]:
    results = {N: evaluate_on_boundary_states(LinearMPC(params, MPCConfig(horizon=N)), design) for N in HORIZONS}
    reference = np.asarray(results[REFERENCE_HORIZON]["costs"])
    rows = []
    for N, r in results.items():
        relative = np.asarray(r["costs"]) / reference - 1
        rows.append({"horizon": N, "n_stabilized": r["n_stabilized"],
                     "relative_cost_vs_N50_mean_pct": float(100 * np.nanmean(relative)),
                     "relative_cost_vs_N50_worst_pct": float(100 * np.nanmax(relative)),
                     "timing": timing_summary(r["solve_times_s"])})
    return rows


def margin_sweep(params, design) -> list[dict]:
    rows = []
    for deg in MARGIN_SWEEP_DEG:
        limit = min(np.radians(deg), params.theta_threshold_radians)
        r = evaluate_on_boundary_states(LinearMPC(params, MPCConfig(theta_constraint_rad=limit)), design)
        rows.append({"theta_constraint_deg": deg, "n_stabilized": r["n_stabilized"],
                     "stabilized": r["stabilized"], "max_abs_theta_deg": r["max_abs_theta_deg"]})
    return rows


def solver_benchmark(params) -> dict:
    """Clarabel frente a OSQP (ajustes precisos) en QP difíciles: estado, precisión y peor tiempo."""
    configs = {"CLARABEL": MPCConfig(), "OSQP (eps 1e-7, polish)": MPCConfig(solver="OSQP", solver_options=OSQP_ACCURATE)}
    reference = [LinearMPC(params)(s) for s in SOLVER_BENCHMARK_STATES]
    out = {}
    for name, config in configs.items():
        controller = LinearMPC(params, config)
        statuses, errors = [], []
        for s, ref in zip(SOLVER_BENCHMARK_STATES, reference):
            try:
                errors.append(abs(controller(s) - ref))
                statuses.append("optimal")
            except RuntimeError as err:
                statuses.append(str(err).split("'")[-2])
        controller.solve_times.clear()
        for s in SOLVER_BENCHMARK_STATES * 5:
            try:
                controller(s)
            except RuntimeError:
                pass
        out[name] = {"statuses": statuses, "max_abs_error_vs_clarabel_N": float(max(errors)) if errors else None,
                     "timing": timing_summary(controller.solve_times)}
    return out


def method_comparison(params, design) -> dict:
    controllers = {
        "PID": CascadePID(load_tuned_gains(), params),
        "LQR": LQRController(design.K, params),
        "MPC": LinearMPC(params),
    }
    return {name: evaluate_on_boundary_states(c, design)["stabilized"] for name, c in controllers.items()}


def plot_comparison(params, design) -> None:
    controllers = {"PID": CascadePID(load_tuned_gains(), params), "LQR": LQRController(design.K, params),
                   "MPC": LinearMPC(params)}
    trajectories = {name: run(c, FIGURE_INITIAL_STATE)[0] for name, c in controllers.items()}
    theta_lim = np.degrees(MPCConfig().theta_constraint(params))
    apply_style()
    fig, (ax_x, ax_th, ax_f) = plt.subplots(3, 1, figsize=(9, 7.8), sharex=True)
    for name, traj in trajectories.items():
        label = name + (" (falla: θ > 12°)" if traj.terminated else "")
        ax_x.plot(traj.times, traj.states[:, 0], color=METHOD_COLORS[name], label=label)
        ax_th.plot(traj.times, np.degrees(traj.states[:, 2]), color=METHOD_COLORS[name])
        ax_f.step(traj.times[:-1], traj.forces, where="post", color=METHOD_COLORS[name])
        if traj.terminated:
            ax_th.plot(traj.times[-1], np.degrees(traj.states[-1, 2]), "x", color=METHOD_COLORS[name], markersize=9)
    for sign in (+1, -1):
        ax_th.axhline(sign * np.degrees(params.theta_threshold_radians), color=MUTED, linestyle=":", linewidth=1)
        ax_th.axhline(sign * theta_lim, color=METHOD_COLORS["MPC"], linestyle="--", linewidth=1, alpha=0.6)
        ax_f.axhline(sign * params.force_mag, color=MUTED, linestyle=":", linewidth=1)
    ax_x.set_ylabel("x [m]"); ax_x.set_title("Posición del carro", loc="left", fontsize=11); ax_x.legend(loc="upper right")
    ax_th.set_ylabel("θ [°]")
    ax_th.set_title(f"Ángulo del poste: fallo en ±12° (punteado) y restricción del MPC en ±{theta_lim:.1f}° (discontinua)",
                    loc="left", fontsize=11)
    ax_f.set_ylabel("F [N]"); ax_f.set_xlabel("t [s]")
    ax_f.set_title("Fuerza del actuador (±10 N)", loc="left", fontsize=11)
    ax_f.set_xlim(0, 6)
    x0, v0, th0, w0 = FIGURE_INITIAL_STATE
    fig.suptitle(f"MPC (N = {MPCConfig().horizon}) frente a LQR y PID\nCI: x={x0} m, ẋ={v0} m/s, θ={th0} rad, θ̇={w0} rad/s",
                 fontsize=12, color=TEXT_SECONDARY)
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def main() -> int:
    params = load_params()
    design = design_default_lqr(params)
    config = MPCConfig()
    record = {"controller": "LinearMPC (CVXPY + Clarabel)", "config": config.to_dict(),
              "theta_constraint_rad": config.theta_constraint(params), "x_constraint_m": config.x_constraint(params)}
    print("Barrido de horizonte...");        record["horizon_sweep"] = horizon_sweep(params, design)
    print("Barrido de margen...");           record["margin_sweep"] = margin_sweep(params, design)
    print("Comparación de solvers...");      record["solver_benchmark"] = solver_benchmark(params)
    print("Comparación con PID y LQR...");   record["near_boundary_comparison"] = method_comparison(params, design)
    print("Sanity check (N = 50)...");       record["sanity_gymnasium_default_ic"] = sanity_statistics(LinearMPC(params), params)
    record["near_boundary_initial_states"] = [list(s) for s in NEAR_BOUNDARY_INITIAL_STATES]
    record["versions"] = {"gymnasium": gym.__version__, "numpy": np.__version__, "cvxpy": cp.__version__}
    record["generated_by"] = "scripts/design_mpc.py"
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_comparison(params, design)

    summary = {k: record[k] for k in ("horizon_sweep", "margin_sweep", "solver_benchmark")}
    summary["near_boundary_counts"] = {k: sum(v) for k, v in record["near_boundary_comparison"].items()}
    summary["sanity"] = {k: record["sanity_gymnasium_default_ic"][k] for k in ("stabilized_rate", "termination_reasons")}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\nGuardado {RECORD_PATH}\nFigura {FIGURE_PATH}")
    ok = record["sanity_gymnasium_default_ic"]["stabilized_rate"] == 1.0
    if not ok:
        print("ATENCIÓN: el MPC NO supera el sanity check; revisar antes de continuar.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
