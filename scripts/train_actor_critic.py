"""Paso 10: actor-crítico A2C + GAE con acción continua, N = 10 semillas, mismo protocolo que 8 y 9.

Uso:
    python scripts/train_actor_critic.py                          entrena, evalúa, exporta y dibuja
    python scripts/train_actor_critic.py --regenerate-artifacts   reentrena SOLO la semilla exportada,
                                                                  verifica que reproduce el registro y
                                                                  regenera pesos y figuras de política

Protocolo:
* Semillas 0–9. Validación: CI 2000–2009 cada 10 240 pasos (política DETERMINISTA, F = F_max·tanh(μ)).
  Test: CI 1000–1049 (protocolo común), de la red final y de la mejor vista en validación.
* "Aprender" (umbral oficial 475): pasos hasta la primera evaluación ≥ 475, y episodios hasta que la
  media móvil de 100 episodios de ENTRENAMIENTO (política estocástica) ≥ 475.
* Se exporta a JSON la mejor red de la semilla con mejor validación (igual que en los pasos 8 y 9).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cartpole_lab import CartPoleTask, run_episode
from cartpole_lab.controllers.lqr import LQRController, design_default_lqr
from cartpole_lab.controllers.pid import CascadePID, load_tuned_gains
from cartpole_lab.params import load_params
from cartpole_lab.paths import FIGURES_DIR, POLICIES_DIR, RESULTS_DIR, WEIGHTS_DIR
from cartpole_lab.plotting import FORCE_CMAP, METHOD_COLORS, MUTED, TEXT_SECONDARY, apply_style
from cartpole_lab.rl.actor_critic import ACConfig, ACPolicy, export_actor, train_a2c
from cartpole_lab.rl.dqn import input_scales, network_forward_numpy
from cartpole_lab.rl.evaluation import SOLVED_THRESHOLD, episodes_to_solve_moving_average, first_crossing, moving_average
from cartpole_lab.sanity import (
    NEAR_BOUNDARY_INITIAL_STATES,
    REFERENCE_HARD_INITIAL_STATE,
    UNRECOVERABLE_INITIAL_STATES,
    is_stabilized,
    sanity_statistics,
)

SEEDS = tuple(range(10))
RECORD_PATH = RESULTS_DIR / "rl" / "actor_critic.json"
WEIGHTS_PATH = WEIGHTS_DIR / "actor_critic.json"
FINAL_WEIGHTS_PATH = WEIGHTS_DIR / "actor_critic_final.json"
COLOR = METHOD_COLORS["Actor-crítico"]


def _config_from_dict(d: dict) -> ACConfig:
    return ACConfig(**{**d, "hidden": tuple(d["hidden"])})


def _train_job(job: tuple[int, dict]) -> dict:
    seed, config_dict = job
    run = train_a2c(_config_from_dict(config_dict), seed=seed)
    evals = run.eval_returns.mean(axis=1)
    solved = first_crossing(evals)
    last = max(1, len(evals) // 3)  # último tercio: ¿se MANTIENE? (al menos una evaluación)
    return {
        "seed": seed, "wall_time_s": run.wall_time_s,
        "eval_steps": run.eval_steps.tolist(), "eval_mean": evals.tolist(),
        "policy_std": run.policy_std.tolist(), "explained_variance": run.explained_variance.tolist(),
        "final_log_std": run.final_log_std, "best_log_std": run.best_log_std,
        "n_training_episodes": int(run.episode_returns.size),
        "training_moving_average": moving_average(run.episode_returns).tolist()[::10],
        "final_greedy": float(evals[-3:].mean()), "last_third_min": float(evals[-last:].min()),
        "best_eval_step": run.best_eval_step, "best_eval_mean": run.best_eval_mean,
        "steps_to_solve_greedy": int(run.eval_steps[solved]) if solved is not None else None,
        "episodes_to_solve_classic": episodes_to_solve_moving_average(run.episode_returns),
        "final_layers": run.final_layers, "best_layers": run.best_layers,
    }


def _test_job(job: tuple[int, str, list]) -> dict:
    seed, which, layers = job
    params = load_params()
    return {"seed": seed, "which": which,
            **sanity_statistics(ACPolicy(layers, input_scales(params), params.force_mag), params)}


def _mean_std(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0}


def _test_summary(tests: list[dict]) -> dict:
    keys = ("survival_rate", "stabilized_rate")
    summary = {k: _mean_std([t[k] for t in tests]) for k in keys}
    summary.update(
        effort_Ns=_mean_std([t["control_effort_abs_Ns"]["mean"] for t in tests]),
        fraction_at_force_limit=_mean_std([t["fraction_of_steps_at_force_limit"]["mean"] for t in tests]),
        last_second_max_abs_theta_deg=_mean_std([t["last_second_max_abs_theta_deg"]["mean"] for t in tests]),
        last_second_max_abs_x_m=_mean_std([t["last_second_max_abs_x_m"]["mean"] for t in tests]),
        termination_reasons_total={k: sum(t["termination_reasons"].get(k, 0) for t in tests)
                                   for k in sorted({r for t in tests for r in t["termination_reasons"]})},
    )
    return summary


def summarize(runs: list[dict], tests: list[dict]) -> dict:
    solved = [r["steps_to_solve_greedy"] for r in runs if r["steps_to_solve_greedy"] is not None]
    classic = [r["episodes_to_solve_classic"] for r in runs if r["episodes_to_solve_classic"] is not None]
    return {
        "n_seeds": len(runs),
        "final_greedy_validation": _mean_std([r["final_greedy"] for r in runs]),
        "seeds_perfect_in_last_third": int(sum(r["last_third_min"] >= 500 for r in runs)),
        "best_checkpoint_validation": _mean_std([r["best_eval_mean"] for r in runs]),
        "seeds_solved_greedy": len(solved), "steps_to_solve_greedy": solved,
        "seeds_solved_classic": len(classic), "episodes_to_solve_classic": classic,
        "final_policy_std": _mean_std([float(np.exp(r["final_log_std"])) for r in runs]),
        "test_final_network": _test_summary([t for t in tests if t["which"] == "final"]),
        "test_best_checkpoint": _test_summary([t for t in tests if t["which"] == "best"]),
        "wall_time_per_run_s": _mean_std([r["wall_time_s"] for r in runs]),
    }


def plot_learning_curves(runs: list[dict]) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    steps = np.array(runs[0]["eval_steps"])
    curves = np.array([r["eval_mean"] for r in runs])
    mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1)
    ax = axes[0]
    for curve in curves:
        ax.plot(steps, curve, color=COLOR, alpha=0.18, linewidth=1)
    ax.plot(steps, mean, color=COLOR, label=f"media de {len(runs)} semillas")
    ax.fill_between(steps, mean - std, mean + std, color=COLOR, alpha=0.15, linewidth=0, label="± 1 desv. típica")
    ax.axhline(SOLVED_THRESHOLD, color=MUTED, linestyle=":", linewidth=1)
    ax.set_ylim(0, 520)
    ax.set_title("Actor-crítico: política determinista en validación", loc="left", fontsize=11)
    ax.set_xlabel("pasos de entorno")
    ax.set_ylabel("pasos sin caer (máx. 500)")
    ax.legend(loc="lower right", fontsize=9)
    ax = axes[1]
    n_updates = len(runs[0]["policy_std"])
    update_steps = (np.arange(n_updates) + 1) * (steps[-1] / n_updates)
    for r in runs:
        ax.plot(update_steps, r["policy_std"], color=COLOR, alpha=0.4, linewidth=1)
    ax.set_title("σ de la política (unidades de u; F = 10·tanh(u))", loc="left", fontsize=11)
    ax.set_xlabel("pasos de entorno")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_curvas_actor_critico.png", dpi=150)
    plt.close(fig)


def plot_policy_and_response(layers: list[dict], seed: int, step: int, final_layers: list[dict]) -> None:
    """Superficie F(θ, θ̇) (carro centrado y quieto) y respuesta temporal frente a LQR y PID."""
    params = load_params()
    policy = ACPolicy(layers, input_scales(params), params.force_mag)
    theta = np.linspace(-params.theta_threshold_radians, params.theta_threshold_radians, 241)
    theta_dot = np.linspace(-2.0, 2.0, 241)
    T, W = np.meshgrid(theta, theta_dot, indexing="ij")
    states = np.stack([np.zeros_like(T), np.zeros_like(T), T, W], axis=-1).reshape(-1, 4)
    force = params.force_mag * np.tanh(network_forward_numpy(layers, states / np.asarray(input_scales(params)))[:, 0])
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(W, np.degrees(T), force.reshape(T.shape), cmap=FORCE_CMAP, vmin=-10, vmax=10, shading="auto")
    ax.set_xlabel("θ̇ [rad/s]")
    ax.set_ylabel("θ [°]")
    ax.set_title(f"Actor-crítico (semilla {seed}, paso {step}): fuerza con el carro centrado y quieto", loc="left", fontsize=11)
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label="fuerza [N]")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_politica_actor_critico.png", dpi=150)
    plt.close(fig)

    controllers = {"PID": CascadePID(load_tuned_gains(), params), "LQR": LQRController(design_default_lqr(params).K, params),
                   "Actor-crítico": policy,
                   "Actor-crítico (red final)": ACPolicy(final_layers, input_scales(params), params.force_mag)}
    task = CartPoleTask(action_mode="continuous")
    try:
        trajectories = {n: run_episode(task, c, initial_state=REFERENCE_HARD_INITIAL_STATE) for n, c in controllers.items()}
    finally:
        task.close()
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for name, traj in trajectories.items():
        color = METHOD_COLORS[name.split(" (")[0]]  # la red final es el mismo método: mismo color, otra línea
        style = "--" if "final" in name else "-"
        label = f"{name} (paso {step})" if name == "Actor-crítico" else name
        label += "  → sale del raíl" if traj.termination_reason == "cart_position_limit" else ""
        axes[0].plot(traj.times, traj.states[:, 0], color=color, linestyle=style, label=label)
        axes[1].plot(traj.times, np.degrees(traj.states[:, 2]), color=color, linestyle=style)
        axes[2].step(traj.times[:-1], traj.forces, where="post", color=color, linestyle=style, linewidth=1.4)
    for ax, label, title in zip(axes, ("x [m]", "θ [°]", "F [N]"),
                                ("Posición del carro", "Ángulo del poste", "Fuerza del actuador (±10 N)")):
        ax.set_ylabel(label)
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[2].set_xlabel("t [s]")
    x0, v0, th0, w0 = REFERENCE_HARD_INITIAL_STATE
    fig.suptitle(f"Actor-crítico aprendido frente a PID y LQR diseñados\nCI: x={x0} m, ẋ={v0} m/s, θ={th0} rad, θ̇={w0} rad/s",
                 fontsize=12, color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_actor_critico_vs_clasicos.png", dpi=150)
    plt.close(fig)


def export_best(runs: list[dict], config: ACConfig) -> dict:
    """Regla común (pasos 8-9): la mejor red en validación, la primera ante empates. Como aquí casi todas
    las evaluaciones valen 500, eso elige la red MÁS TEMPRANA; por eso se exporta también la red FINAL
    de la misma semilla (docs/10_actor_critico.md)."""
    best = max(runs, key=lambda r: r["best_eval_mean"])
    params = load_params()
    common = {"method": "actor_critic_a2c_gae", "seed": best["seed"], "config": config.to_dict(),
              "torch_version": torch.__version__}
    export_actor(best["best_layers"], WEIGHTS_PATH, input_scales=input_scales(params), force_limit=params.force_mag,
                 log_std=best["best_log_std"], metadata={**common, "checkpoint_step": best["best_eval_step"], "validation_mean_return": best["best_eval_mean"],
                           "selected_by": "mejor red vista en VALIDACIÓN (semillas 2000-2009), la primera ante empates"})
    export_actor(best["final_layers"], FINAL_WEIGHTS_PATH, input_scales=input_scales(params), force_limit=params.force_mag,
                 log_std=best["final_log_std"], metadata={**common, "checkpoint_step": config.total_steps, "validation_mean_return": best["final_greedy"],
                           "selected_by": "red FINAL de la misma semilla (alternativa a la regla común)"})
    return best


def boundary_checks(best: dict) -> dict:
    """Más allá del test (CI cerca del centro): estado difícil de referencia, 12 estados cerca de la frontera
    de la Fase 1 y estados irrecuperables. ¿Sobrevive? ¿Se estabiliza? ¿Falla de forma explícita?"""
    params = load_params()
    task = CartPoleTask(action_mode="continuous")
    out = {}
    try:
        for which in ("best_layers", "final_layers"):
            policy = ACPolicy(best[which], input_scales(params), params.force_mag)
            near = [run_episode(task, policy, initial_state=s) for s in NEAR_BOUNDARY_INITIAL_STATES]
            extreme = {n: run_episode(task, policy, initial_state=s).termination_reason
                       for n, (s, _) in UNRECOVERABLE_INITIAL_STATES.items()}
            hard = run_episode(task, policy, initial_state=REFERENCE_HARD_INITIAL_STATE)
            out[which.replace("_layers", "")] = {
                "reference_hard": {"reason": hard.termination_reason, "stabilized": is_stabilized(hard, params)},
                "near_boundary_survived": sum(t.termination_reason == "time_limit" for t in near),
                "near_boundary_stabilized": sum(is_stabilized(t, params) for t in near),
                "near_boundary_reasons": [t.termination_reason for t in near],
                "unrecoverable_reasons": extreme,
            }
    finally:
        task.close()
    return out


def save_seed_policies(runs: list[dict], config: ACConfig) -> None:
    """Actor de CADA semilla (mejor en validación y final), para la evaluación de la Fase 3."""
    params = load_params()
    for r in runs:
        for which in ("best", "final"):
            export_actor(r[f"{which}_layers"], POLICIES_DIR / "actor_critic" / f"seed_{r['seed']:02d}_{which}.json",
                         input_scales=input_scales(params), force_limit=params.force_mag, log_std=r[f"{which}_log_std"],
                         metadata={"method": "actor_critic", "seed": r["seed"], "which": which,
                                   "checkpoint_step": r["best_eval_step"] if which == "best" else config.total_steps})


def main(config: ACConfig = ACConfig(), seeds: tuple[int, ...] = SEEDS) -> int:
    with ProcessPoolExecutor(max_workers=min(12, len(seeds))) as pool:
        runs = list(pool.map(_train_job, [(s, config.to_dict()) for s in seeds]))
        tests = list(pool.map(_test_job, [(r["seed"], w, r[k]) for r in runs
                                          for w, k in (("final", "final_layers"), ("best", "best_layers"))]))
    record = {"protocol": __doc__, "config": config.to_dict(), "seeds": list(seeds), "summary": summarize(runs, tests),
              "runs": [{k: v for k, v in r.items() if k not in ("final_layers", "best_layers")} for r in runs],
              "versions": {"gymnasium": gym.__version__, "numpy": np.__version__, "torch": torch.__version__},
              "generated_by": "scripts/train_actor_critic.py"}
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    best = export_best(runs, config)
    save_seed_policies(runs, config)
    record["boundary_checks_exported_seed"] = boundary_checks(best)
    RECORD_PATH.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_learning_curves(runs)
    plot_policy_and_response(best["best_layers"], best["seed"], best["best_eval_step"], best["final_layers"])
    print(json.dumps(record["boundary_checks_exported_seed"], indent=1, ensure_ascii=False))
    print(json.dumps(record["summary"], indent=1, ensure_ascii=False))
    return 0


def regenerate_artifacts() -> int:
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    target = max(record["runs"], key=lambda r: r["best_eval_mean"])
    fresh = _train_job((target["seed"], record["config"]))
    for key in ("eval_mean", "best_eval_step", "best_eval_mean", "policy_std", "best_log_std", "final_log_std"):
        if not np.allclose(fresh[key], target[key], rtol=1e-9, atol=1e-9):
            raise RuntimeError(f"No reproducible: semilla {target['seed']} difiere en {key}")
    export_best([fresh], _config_from_dict(record["config"]))
    plot_policy_and_response(fresh["best_layers"], fresh["seed"], fresh["best_eval_step"], fresh["final_layers"])
    print(f"Semilla {target['seed']} reproducida; pesos y figuras regenerados")
    return 0


if __name__ == "__main__":
    sys.exit(regenerate_artifacts() if "--regenerate-artifacts" in sys.argv else main())
