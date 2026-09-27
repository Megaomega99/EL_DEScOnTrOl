"""Paso 6: sintoniza los factores de escala del fuzzy (protocolo del PID), verifica y compara.

Uso:  python scripts/tune_fuzzy.py      (~3-4 min)

Salidas:
    results/tuning/fuzzy.json               parámetros, ejecuciones, estabilidad, PD equivalente,
                                            comparaciones y sanity check
    results/figures/06_fuzzy_reglas.png     membresías y superficie de control F(e, θ̇)
    results/figures/06_fuzzy_vs_pid.png     respuesta frente al PID desde el estado difícil
"""

from __future__ import annotations

import json
import sys
import time

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from cartpole_lab import CartPoleTask, load_params, run_episode
from cartpole_lab.controllers.fuzzy import (
    LABELS, RULE_TABLE, TUNED_FUZZY_PATH, CascadeFuzzy, FuzzyGains, fuzzy_force, memberships,
)
from cartpole_lab.controllers.fuzzy_tuning import (
    FUZZY_GAIN_BOUNDS, FuzzyTuningConfig, fuzzy_closed_loop_spectral_radius, tune_fuzzy,
)
from cartpole_lab.controllers.pid import CascadePID, load_tuned_gains, load_tuning_record
from cartpole_lab.controllers.pid_tuning import itae_cost
from cartpole_lab.paths import FIGURES_DIR, TUNING_DIR
from cartpole_lab.plotting import GRID, METHOD_COLORS, MUTED, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, apply_style
from cartpole_lab.sanity import NEAR_BOUNDARY_INITIAL_STATES, REFERENCE_HARD_INITIAL_STATE, is_stabilized, sanity_statistics

RULES_FIGURE = FIGURES_DIR / "06_fuzzy_reglas.png"
RESPONSE_FIGURE = FIGURES_DIR / "06_fuzzy_vs_pid.png"
CROSSOVER_SEED = 2027  # mismas CI fuera de muestra que la comparación PID-LQR del paso 4
SENSITIVITY_U_MAX = 40.0  # N: "empujar fuerte" más allá del actuador (solo como experimento)
# Divergente azul <-> gris <-> rojo (paleta de referencia): signo de la fuerza.
FORCE_CMAP = LinearSegmentedColormap.from_list("fuerza", ["#2a78d6", "#f0efec", "#e34948"])


def run(controller, initial_state):
    task = CartPoleTask(action_mode="continuous")
    try:
        return run_episode(task, controller, initial_state=initial_state)
    finally:
        task.close()


def run_tuning(config, params):
    start = time.perf_counter()
    result = tune_fuzzy(config, params)
    elapsed = time.perf_counter() - start
    print(f"Sintonización: {len(result.runs)} ejecuciones en {elapsed:.1f} s")
    for r in result.runs:
        print(f"  semilla {r.de_seed}: J={r.cost:.5f} nit={r.nit} convergió={r.converged} | "
              + ", ".join(f"{k}={v:.4g}" for k, v in r.gains.to_dict().items()))
    tuning = {"method": "differential_evolution (scipy), mismo protocolo que el PID", "config": config.to_dict(),
              "selected_de_seed": result.best.de_seed, "cost": result.best.cost, "wall_time_s": elapsed,
              "runs": [{"de_seed": r.de_seed, "cost": r.cost, "nit": r.nit, "nfev": r.nfev, "converged": r.converged,
                        "message": r.message, "gains": r.gains.to_dict()} for r in result.runs]}
    return result.best.gains, tuning


def output_scale_sensitivity(params) -> dict:
    """¿La desventaja frente al PID se debe a limitar U al actuador? Se re-sintoniza con U ≤ 40 N
    (una semilla) y se compara el PD equivalente con el PID."""
    bounds = list(FUZZY_GAIN_BOUNDS)
    bounds[2] = (0.5, SENSITIVITY_U_MAX)
    best = tune_fuzzy(FuzzyTuningConfig(bounds=tuple(bounds), de_seeds=(0,)), params).best
    return {"u_max_N": SENSITIVITY_U_MAX, "cost": best.cost, "gains": best.gains.to_dict(),
            "equivalent_pd": best.gains.equivalent_pd()}


def comparisons(gains, params, config) -> dict:
    fuzzy = CascadeFuzzy(gains, params)
    pid = CascadePID(load_tuned_gains(), params)
    near = [is_stabilized(run(fuzzy, s0), params) for s0 in NEAR_BOUNDARY_INITIAL_STATES]
    mpc_record = json.loads((TUNING_DIR / "mpc.json").read_text(encoding="utf-8"))
    box = np.asarray(config.initial_state_box)
    initial_states = np.random.default_rng(CROSSOVER_SEED).uniform(-box, box, size=(config.n_initial_states, 4))
    itae = {name: np.array([itae_cost(run(c, s0), params, config) for s0 in initial_states])
            for name, c in (("PID", pid), ("Fuzzy", fuzzy))}
    return {
        "near_boundary": {"Fuzzy": near, **mpc_record["near_boundary_comparison"]},
        "itae_out_of_sample": {
            name: {"mean": float(v.mean()), "std": float(v.std(ddof=1))} for name, v in itae.items()
        } | {"fuzzy_better_count": int(np.sum(itae["Fuzzy"] < itae["PID"])), "n": int(itae["PID"].size),
             "fuzzy_vs_pid_relative_mean_pct": float(100 * np.mean(itae["Fuzzy"] / itae["PID"] - 1))},
    }


def plot_rules(gains, params) -> None:
    apply_style()
    fig = plt.figure(figsize=(12, 5.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], hspace=0.55, wspace=0.25)
    color = METHOD_COLORS["Fuzzy"]
    for row, (scale, unit, title, factor) in enumerate(
        [(gains.theta_scale, "e [°]", "Error de ángulo e = θ − θ_ref", np.degrees(1.0)),
         (gains.theta_dot_scale, "θ̇ [rad/s]", "Velocidad angular θ̇", 1.0)]
    ):
        ax = fig.add_subplot(grid[row, 0])
        z = np.linspace(-1.4, 1.4, 600)
        mu = memberships(z)
        for k, label in enumerate(LABELS):
            ax.plot(z * scale * factor, mu[:, k], color=color)
            ax.text((k - 2) * 0.5 * scale * factor, 1.06, label, ha="center", fontsize=10, color=TEXT_PRIMARY)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("pertenencia")
        ax.set_xlabel(unit)
        ax.set_title(title, loc="left", fontsize=11)

    ax = fig.add_subplot(grid[:, 1])
    e = np.linspace(-1.3, 1.3, 261) * gains.theta_scale
    w = np.linspace(-1.3, 1.3, 261) * gains.theta_dot_scale
    E, W = np.meshgrid(e, w, indexing="ij")
    states = np.stack([np.zeros_like(E), np.zeros_like(E), E, W], axis=-1)  # x = 0 -> e = θ
    # Con x = ẋ = 0, θ_ref = 0 para cualquier límite: e = θ exactamente.
    force = fuzzy_force(states, gains, force_limit=params.force_mag, theta_ref_limit=0.0)
    mesh = ax.pcolormesh(W, np.degrees(E), force, cmap=FORCE_CMAP, vmin=-params.force_mag, vmax=params.force_mag,
                         shading="auto")
    centers = np.linspace(-1, 1, 5)
    for i, ce in enumerate(centers):
        for j, cw in enumerate(centers):
            label = LABELS[int(round(2 * RULE_TABLE[i, j])) + 2]
            ax.text(cw * gains.theta_dot_scale, np.degrees(ce * gains.theta_scale), label,
                    ha="center", va="center", fontsize=8.5, color=TEXT_PRIMARY,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.75})
    ax.add_patch(plt.Rectangle((-0.5 * gains.theta_dot_scale, -0.5 * np.degrees(gains.theta_scale)),
                               gains.theta_dot_scale, np.degrees(gains.theta_scale),
                               fill=False, edgecolor=TEXT_SECONDARY, linestyle="--", linewidth=1.2))
    ax.set_xlabel("θ̇ [rad/s]")
    ax.set_ylabel("e [°]")
    ax.set_title("Superficie de control F(e, θ̇): 25 reglas interpoladas\n"
                 "(recuadro: celda central, donde el fuzzy es exactamente un PD)", loc="left", fontsize=11)
    ax.grid(False)
    bar = fig.colorbar(mesh, ax=ax)
    bar.set_label("F [N]   (− izquierda | derecha +)")
    fig.suptitle("Controlador fuzzy Takagi-Sugeno (orden 0): membresías sintonizadas y tabla de reglas fija",
                 fontsize=12, color=TEXT_SECONDARY, y=1.03)
    RULES_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(RULES_FIGURE, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_response(gains, params) -> None:
    trajectories = {"PID": run(CascadePID(load_tuned_gains(), params), REFERENCE_HARD_INITIAL_STATE),
                    "Fuzzy": run(CascadeFuzzy(gains, params), REFERENCE_HARD_INITIAL_STATE)}
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for name, traj in trajectories.items():
        axes[0].plot(traj.times, traj.states[:, 0], color=METHOD_COLORS[name], label=name)
        axes[1].plot(traj.times, np.degrees(traj.states[:, 2]), color=METHOD_COLORS[name])
        axes[2].step(traj.times[:-1], traj.forces, where="post", color=METHOD_COLORS[name])
    for sign in (+1, -1):
        axes[2].axhline(sign * params.force_mag, color=MUTED, linestyle=":", linewidth=1)
    for ax, ylabel, title in zip(axes, ("x [m]", "θ [°]", "F [N]"),
                                 ("Posición del carro", "Ángulo del poste", "Fuerza del actuador (±10 N)")):
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].legend(loc="upper right")
    axes[2].set_xlabel("t [s]")
    x0, v0, th0, w0 = REFERENCE_HARD_INITIAL_STATE
    fig.suptitle(f"Fuzzy frente a PID (misma cascada y mismo criterio de sintonización)\n"
                 f"CI: x={x0} m, ẋ={v0} m/s, θ={th0} rad, θ̇={w0} rad/s", fontsize=12, color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(RESPONSE_FIGURE, dpi=150)
    plt.close(fig)


def main() -> int:
    params = load_params()
    config = FuzzyTuningConfig()
    gains, tuning = run_tuning(config, params)
    limit = config.theta_ref_limit(params)
    pid_record = load_tuning_record()
    record = {
        "controller": "CascadeFuzzy (Takagi-Sugeno orden 0, tabla MacVicar-Whelan 5x5)",
        "gains": gains.to_dict(), "theta_ref_limit_rad": limit, "tuning": tuning,
        "rule_table": RULE_TABLE.tolist(), "labels": list(LABELS),
        "equivalent_pd": gains.equivalent_pd(),
        "pid_equivalent_K_for_reference": pid_record["diagnostics"]["stability"]["equivalent_state_feedback_K"],
        "closed_loop_spectral_radius": fuzzy_closed_loop_spectral_radius(gains, params, limit),
        "comparisons": comparisons(gains, params, config),
        "sensitivity_output_scale_beyond_actuator": output_scale_sensitivity(params),
        "sanity_gymnasium_default_ic": sanity_statistics(CascadeFuzzy(gains, params), params),
        "hard_ic_stabilized": is_stabilized(run(CascadeFuzzy(gains, params), REFERENCE_HARD_INITIAL_STATE), params),
        "versions": {"gymnasium": gym.__version__, "numpy": np.__version__},
        "generated_by": "scripts/tune_fuzzy.py",
    }
    TUNED_FUZZY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_FUZZY_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_rules(gains, params)
    plot_response(gains, params)
    shown = {k: record[k] for k in ("gains", "equivalent_pd", "pid_equivalent_K_for_reference",
                                    "closed_loop_spectral_radius", "hard_ic_stabilized")}
    shown["near_boundary_counts"] = {k: sum(v) for k, v in record["comparisons"]["near_boundary"].items()}
    shown["itae_out_of_sample"] = record["comparisons"]["itae_out_of_sample"]
    shown["sensitivity"] = record["sensitivity_output_scale_beyond_actuator"]
    shown["sanity"] = {k: record["sanity_gymnasium_default_ic"][k]
                       for k in ("stabilized_rate", "control_effort_abs_Ns", "fraction_of_steps_at_force_limit")}
    print(json.dumps(shown, indent=1, ensure_ascii=False))
    ok = record["closed_loop_spectral_radius"] < 1.0 and record["sanity_gymnasium_default_ic"]["stabilized_rate"] == 1.0
    if not ok:
        print("ATENCIÓN: el fuzzy NO supera el sanity check; revisar antes de continuar.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
