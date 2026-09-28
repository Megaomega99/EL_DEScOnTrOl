"""Paso 9: DQN y Double DQN con N = 10 semillas cada uno, mismo protocolo que el paso 8.

Uso:
    python scripts/train_dqn.py                           entrena ambas variantes, evalúa, exporta y dibuja
    python scripts/train_dqn.py --regenerate-artifacts    reentrena SOLO las semillas exportadas, comprueba
                                                          que reproducen el registro EXACTAMENTE y regenera
                                                          pesos y figura de política

Protocolo (declarado antes de ver resultados de Double DQN):
* Dos variantes con hiperparámetros IDÉNTICOS: DQN (Mnih et al., 2015) y Double DQN (van Hasselt
  et al., 2016). Motivo: el DQN exploratorio estimó Q(s=0) ≈ 109, por encima de la cota física
  de 100 (sobreestimación), y Double DQN es el remedio estándar para ese diagnóstico.
* Semillas 0–9. Validación: CI 2000–2009, cada 5 000 pasos. Test: CI 1000–1049 (protocolo común).
* En test: red final y mejor red vista en validación (parada temprana).
* "Aprender" (umbral oficial 475): pasos y episodios hasta la primera evaluación greedy ≥ 475,
  y episodios hasta que la media móvil de 100 episodios de entrenamiento ≥ 475.
* Se exporta a JSON (para el navegador) la mejor red de la semilla con mejor validación, por variante.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cartpole_lab.params import load_params
from cartpole_lab.paths import FIGURES_DIR, POLICIES_DIR, RESULTS_DIR, WEIGHTS_DIR
from cartpole_lab.plotting import FORCE_CMAP, METHOD_COLORS, MUTED, TEXT_SECONDARY, apply_style
from cartpole_lab.rl.dqn import (
    DQNConfig, DQNPolicy, export_network, input_scales, network_forward_numpy, q_upper_bound, train_dqn,
)
from cartpole_lab.rl.evaluation import SOLVED_THRESHOLD, episodes_to_solve_moving_average, first_crossing
from cartpole_lab.sanity import sanity_statistics

SEEDS = tuple(range(10))
VARIANTS = {"dqn": "DQN", "double_dqn": "Double DQN"}
RECORD_PATH = RESULTS_DIR / "rl" / "dqn.json"
COLOR = METHOD_COLORS["DQN"]  # misma familia de método: mismo color; las variantes van en paneles distintos


def config_for(variant: str, base: DQNConfig = DQNConfig()) -> DQNConfig:
    return dataclasses.replace(base, double=(variant == "double_dqn"))


def _config_from_dict(d: dict) -> DQNConfig:
    return DQNConfig(**{**d, "hidden": tuple(d["hidden"]), "force_levels": tuple(d["force_levels"])})


def _train_job(job: tuple[str, int, dict]) -> dict:
    variant, seed, config_dict = job
    run = train_dqn(_config_from_dict(config_dict), seed=seed)
    evals = run.eval_returns.mean(axis=1)
    solved = first_crossing(evals)  # índice de la primera evaluación greedy ≥ 475
    classic_episode = episodes_to_solve_moving_average(run.episode_returns)
    return {
        "variant": variant, "seed": seed, "wall_time_s": run.wall_time_s,
        "eval_steps": run.eval_steps.tolist(), "eval_episodes": run.eval_episodes.tolist(), "eval_mean": evals.tolist(),
        "mean_losses": run.mean_losses.tolist(), "q_at_rest": run.q_at_rest.tolist(),
        "action_gap_median": run.action_gap_median.tolist(), "q_max_median": run.q_max_median.tolist(),
        "n_training_episodes": int(run.episode_returns.size), "final_greedy": float(evals[-3:].mean()),
        "best_eval_step": run.best_eval_step, "best_eval_mean": run.best_eval_mean,
        "steps_to_solve_greedy": int(run.eval_steps[solved]) if solved is not None else None,
        "episodes_to_solve_greedy": int(run.eval_episodes[solved]) if solved is not None else None,
        "episodes_to_solve_classic": classic_episode,
        "steps_to_solve_classic": int(run.episode_end_steps[classic_episode - 1]) if classic_episode else None,
        "final_layers": run.final_layers, "best_layers": run.best_layers,
    }


def _test_job(job: tuple[str, int, str, list]) -> dict:
    variant, seed, which, layers = job
    params = load_params()
    policy = DQNPolicy(layers, input_scales(params), tuple(DQNConfig().force_levels))
    return {"variant": variant, "seed": seed, "which": which, **sanity_statistics(policy, params)}


def _mean_std(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0}


def _test_summary(tests: list[dict]) -> dict:
    return {"survival_rate": _mean_std([x["survival_rate"] for x in tests]),
            "stabilized_rate": _mean_std([x["stabilized_rate"] for x in tests]),
            "effort_Ns": _mean_std([x["control_effort_abs_Ns"]["mean"] for x in tests]),
            "fraction_at_force_limit": _mean_std([x["fraction_of_steps_at_force_limit"]["mean"] for x in tests]),
            "termination_reasons_total": {k: sum(x["termination_reasons"].get(k, 0) for x in tests)
                                          for k in sorted({r for x in tests for r in x["termination_reasons"]})}}


def summarize(runs: list[dict], tests: list[dict]) -> dict:
    solved = [r for r in runs if r["steps_to_solve_greedy"] is not None]
    classic = [r for r in runs if r["episodes_to_solve_classic"] is not None]
    q_final = [r["q_at_rest"][-1] for r in runs]
    gap_ratio = [np.array(r["action_gap_median"]) / np.abs(np.array(r["q_max_median"])) for r in runs]
    return {
        "n_seeds": len(runs),
        "final_greedy_validation": _mean_std([r["final_greedy"] for r in runs]),
        "best_checkpoint_validation": _mean_std([r["best_eval_mean"] for r in runs]),
        "best_checkpoint_step": _mean_std([r["best_eval_step"] for r in runs]),
        "seeds_solved_greedy": len(solved),
        "steps_to_solve_greedy": [r["steps_to_solve_greedy"] for r in solved],
        "episodes_to_solve_greedy": [r["episodes_to_solve_greedy"] for r in solved],
        "seeds_solved_classic": len(classic),
        "episodes_to_solve_classic": [r["episodes_to_solve_classic"] for r in classic],
        "steps_to_solve_classic": [r["steps_to_solve_classic"] for r in classic],
        "q_at_rest_final": _mean_std(q_final),
        "seeds_with_q_above_physical_bound_at_end": int(sum(q > q_upper_bound(DQNConfig().gamma) for q in q_final)),
        "relative_action_gap_first_eval": _mean_std([g[0] for g in gap_ratio]),
        "relative_action_gap_last_eval": _mean_std([g[-1] for g in gap_ratio]),
        "test_final_network": _test_summary([t for t in tests if t["which"] == "final"]),
        "test_best_checkpoint": _test_summary([t for t in tests if t["which"] == "best"]),
        "wall_time_per_run_s": _mean_std([r["wall_time_s"] for r in runs]),
    }


def plot_value_diagnostics(runs: list[dict]) -> None:
    """¿Por qué colapsa? Q en reposo frente a la cota física, y la diferencia entre acciones."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for variant, style in (("dqn", "-"), ("double_dqn", "--")):
        vr = [r for r in runs if r["variant"] == variant]
        steps = vr[0]["eval_steps"]
        axes[0].plot(steps, np.mean([r["q_at_rest"] for r in vr], axis=0), color=COLOR, linestyle=style,
                     label=VARIANTS[variant])
        relative_gap = np.array([r["action_gap_median"] for r in vr]) / np.abs(np.array([r["q_max_median"] for r in vr]))
        axes[1].semilogy(steps, np.median(relative_gap, axis=0), color=COLOR, linestyle=style, label=VARIANTS[variant])
    axes[0].axhline(q_upper_bound(DQNConfig().gamma), color=TEXT_SECONDARY, linestyle=":", linewidth=1.2)
    axes[0].text(0, q_upper_bound(DQNConfig().gamma) + 2, "máximo físico 1/(1−γ) = 100", fontsize=9, color=TEXT_SECONDARY)
    axes[0].set_title("max Q(s = 0): sobreestimación (media de semillas)", loc="left", fontsize=11)
    axes[1].set_title("Diferencia entre las 2 mejores acciones / Q\n(mediana en 256 estados de prueba y semillas)",
                      loc="left", fontsize=11)
    for ax in axes:
        ax.set_xlabel("pasos de entorno")
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_diagnostico_dqn.png", dpi=150)
    plt.close(fig)


def plot_learning_curves(runs: list[dict]) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (variant, label) in zip(axes, VARIANTS.items()):
        vr = [r for r in runs if r["variant"] == variant]
        steps, curves = np.array(vr[0]["eval_steps"]), np.array([r["eval_mean"] for r in vr])
        mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1)
        for curve in curves:
            ax.plot(steps, curve, color=COLOR, alpha=0.18, linewidth=1)
        ax.plot(steps, mean, color=COLOR, label=f"media de {len(vr)} semillas")
        ax.fill_between(steps, mean - std, mean + std, color=COLOR, alpha=0.15, linewidth=0, label="± 1 desv. típica")
        ax.axhline(SOLVED_THRESHOLD, color=MUTED, linestyle=":", linewidth=1)
        ax.set_ylim(0, 520)
        ax.set_title(f"{label}: retorno greedy (validación)", loc="left", fontsize=11)
        ax.set_xlabel("pasos de entorno")
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("pasos sin caer (máx. 500)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_curvas_dqn.png", dpi=150)
    plt.close(fig)


def plot_policies(best_by_variant: dict[str, dict]) -> None:
    """Fuerza elegida en el plano (θ, θ̇) con el carro centrado y quieto (comparable con las figuras 06 y 08)."""
    params = load_params()
    theta = np.linspace(-params.theta_threshold_radians, params.theta_threshold_radians, 241)
    theta_dot = np.linspace(-2.0, 2.0, 241)
    T, W = np.meshgrid(theta, theta_dot, indexing="ij")
    states = np.stack([np.zeros_like(T), np.zeros_like(T), T, W], axis=-1).reshape(-1, 4)
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (variant, best) in zip(axes, best_by_variant.items()):
        q = network_forward_numpy(best["best_layers"], states / np.asarray(input_scales(params)))
        force = np.asarray(DQNConfig().force_levels)[q.argmax(axis=1)].reshape(T.shape)
        mesh = ax.pcolormesh(W, np.degrees(T), force, cmap=FORCE_CMAP, vmin=-10, vmax=10, shading="auto")
        ax.set_title(f"{VARIANTS[variant]}: semilla {best['seed']}, paso {best['best_eval_step']}\n"
                     "(mejor red vista en validación)", loc="left", fontsize=11)
        ax.set_xlabel("θ̇ [rad/s]")
        ax.grid(False)
    axes[0].set_ylabel("θ [°]")
    fig.colorbar(mesh, ax=axes, label="fuerza elegida [N]")
    fig.suptitle("Política aprendida (corte con el carro centrado y quieto)", fontsize=12, color=TEXT_SECONDARY, y=1.06)
    fig.savefig(FIGURES_DIR / "09_politica_dqn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_best(runs: list[dict], variant: str, config: DQNConfig) -> dict:
    best = max((r for r in runs if r["variant"] == variant), key=lambda r: r["best_eval_mean"])
    export_network(best["best_layers"], WEIGHTS_DIR / f"{variant}.json", input_scales=input_scales(load_params()),
                   force_levels=config.force_levels,
                   metadata={"method": variant, "seed": best["seed"], "checkpoint_step": best["best_eval_step"],
                             "validation_mean_return": best["best_eval_mean"], "config": config.to_dict(),
                             "selected_by": "mejor red vista en VALIDACIÓN (semillas 2000-2009), nunca en test",
                             "torch_version": torch.__version__})
    return best


def save_seed_policies(runs: list[dict], base: DQNConfig) -> None:
    """Red de CADA semilla y variante (mejor en validación y final), para la evaluación de la Fase 3."""
    scales = input_scales(load_params())
    for r in runs:
        folder = POLICIES_DIR / r["variant"]
        for which, key in (("best", "best_layers"), ("final", "final_layers")):
            export_network(r[key], folder / f"seed_{r['seed']:02d}_{which}.json", input_scales=scales,
                           force_levels=config_for(r["variant"], base).force_levels,
                           metadata={"method": r["variant"], "seed": r["seed"], "which": which,
                                     "checkpoint_step": r["best_eval_step"] if which == "best" else base.total_steps})


def main(base: DQNConfig = DQNConfig(), seeds: tuple[int, ...] = SEEDS) -> int:
    jobs = [(v, s, config_for(v, base).to_dict()) for v in VARIANTS for s in seeds]
    with ProcessPoolExecutor(max_workers=min(12, len(jobs))) as pool:
        runs = list(pool.map(_train_job, jobs))
        tests = list(pool.map(_test_job, [(r["variant"], r["seed"], w, r[k]) for r in runs
                                          for w, k in (("final", "final_layers"), ("best", "best_layers"))]))
    record = {
        "protocol": __doc__, "configs": {v: config_for(v, base).to_dict() for v in VARIANTS}, "seeds": list(seeds),
        "q_upper_bound": q_upper_bound(base.gamma),
        "summary": {v: summarize([r for r in runs if r["variant"] == v], [t for t in tests if t["variant"] == v])
                    for v in VARIANTS},
        "runs": [{k: val for k, val in r.items() if k not in ("final_layers", "best_layers")} for r in runs],
        "versions": {"gymnasium": gym.__version__, "numpy": np.__version__, "torch": torch.__version__},
        "generated_by": "scripts/train_dqn.py",
    }
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    best = {v: export_best(runs, v, config_for(v, base)) for v in VARIANTS}
    save_seed_policies(runs, base)
    plot_learning_curves(runs)
    plot_value_diagnostics(runs)
    plot_policies(best)
    print(json.dumps(record["summary"], indent=1, ensure_ascii=False))
    return 0


def regenerate_artifacts() -> int:
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    fresh_runs = {}
    for variant in VARIANTS:
        target = max((r for r in record["runs"] if r["variant"] == variant), key=lambda r: r["best_eval_mean"])
        fresh = _train_job((variant, target["seed"], record["configs"][variant]))
        for key in ("eval_mean", "best_eval_step", "best_eval_mean", "final_greedy", "q_at_rest"):
            # Tolerancia mínima (no igualdad de bits): PyTorch garantiza determinismo con los mismos
            # ajustes, no idénticos bits entre procesos distintos. En la práctica coinciden exactamente.
            if not np.allclose(fresh[key], target[key], rtol=1e-9, atol=1e-9):
                raise RuntimeError(f"No reproducible: {variant} semilla {target['seed']} difiere en {key}")
        fresh_runs[variant] = export_best([fresh], variant, _config_from_dict(record["configs"][variant]))
        print(f"{variant} semilla {target['seed']}: reproducida exactamente")
    plot_policies(fresh_runs)
    return 0


if __name__ == "__main__":
    sys.exit(regenerate_artifacts() if "--regenerate-artifacts" in sys.argv else main())
