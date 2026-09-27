"""Pasos 7-8: estudio de discretización + Q-learning y SARSA tabulares con N = 10 semillas.

Uso:  python scripts/train_tabular.py          (~10-15 min en paralelo, 12 procesos)

Protocolo (declarado ANTES de ver resultados):
* 3 discretizaciones × 2 métodos × 10 semillas, mismos hiperparámetros (rl/tabular.py).
* Curva de aprendizaje: cada `eval_every` episodios, retorno de la política GREEDY en 10 CI
  de VALIDACIÓN (semillas 2000–2009).
* Selección de la discretización: mayor retorno greedy final (media de las 3 últimas
  evaluaciones, promediada sobre semillas y métodos). Si otra queda a menos de una
  desviación típica (entre semillas) y tiene menos estados, gana la más pequeña.
* "Episodios para aprender" (umbral oficial de CartPole-v1: 475):
    - greedy : primer punto de evaluación con media greedy ≥ 475;
    - clásico: primer episodio con media móvil de 100 episodios de ENTRENAMIENTO ≥ 475.
  None si no se alcanza: se reporta, no se fuerza.
* Evaluación final en las CI de TEST (semillas 1000–1049, protocolo de sanity común) de DOS
  políticas por semilla: la tabla al terminar y la mejor vista en validación (parada temprana).
  Motivo: con cajas la política oscila y la última no es necesariamente la mejor.

Salidas: results/rl/tabular.json, results/weights/tabular_*.json y figuras 07_*, 08_*.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cartpole_lab.paths import FIGURES_DIR, RESULTS_DIR, WEIGHTS_DIR
from cartpole_lab.plotting import FORCE_CMAP, METHOD_COLORS, MUTED, TEXT_PRIMARY, TEXT_SECONDARY, apply_style
from cartpole_lab.rl.discretization import DISCRETIZATIONS
from cartpole_lab.rl.evaluation import (
    SOLVED_THRESHOLD,
    episodes_to_solve_greedy,
    episodes_to_solve_moving_average,
    moving_average,
    select_by_parsimony,
)
from cartpole_lab.rl.tabular import FORCE_LEVELS, GreedyTabularPolicy, TabularConfig, train_tabular
from cartpole_lab.sanity import sanity_statistics
from cartpole_lab.params import load_params

N_EPISODES = 10_000  # las curvas largas de exploración siguen oscilando tras ~5000: no hay meseta estable
EVAL_EVERY = 250
SEEDS = tuple(range(10))
METHOD_NAMES = {"q_learning": "Q-learning", "sarsa": "SARSA"}
RECORD_PATH = RESULTS_DIR / "rl" / "tabular.json"


def config_for(method: str, n_episodes: int = N_EPISODES, eval_every: int = EVAL_EVERY) -> TabularConfig:
    return TabularConfig(method=method, n_episodes=n_episodes, eval_every=eval_every)


def _train_job(job: tuple[str, str, int, int, int]) -> dict:
    # El presupuesto viaja en la tarea (no en variables globales): los procesos hijos
    # re-importan el módulo y no verían cambios hechos en el proceso principal.
    method, disc_name, seed, n_episodes, eval_every = job
    start = time.perf_counter()
    run = train_tabular(config_for(method, n_episodes, eval_every), DISCRETIZATIONS[disc_name], seed=seed)
    evals = run.eval_returns.mean(axis=1)
    return {
        "method": method, "discretization": disc_name, "seed": seed, "wall_time_s": time.perf_counter() - start,
        "eval_episodes": run.eval_episodes.tolist(), "eval_mean": evals.tolist(),
        "final_greedy": float(evals[-3:].mean()),
        "moving_average_max": float(moving_average(run.episode_returns).max()),
        "episodes_to_solve_greedy": episodes_to_solve_greedy(run.eval_episodes, evals),
        "episodes_to_solve_classic": episodes_to_solve_moving_average(run.episode_returns),
        "best_eval_episode": run.best_eval_episode, "best_eval_mean": run.best_eval_mean,
        "q_table": run.q_table.tolist(), "best_q_table": run.best_q_table.tolist(),
    }


def select_discretization(results: list[dict]) -> dict:
    summary = {}
    for name, disc in DISCRETIZATIONS.items():
        finals = np.array([r["final_greedy"] for r in results if r["discretization"] == name])
        summary[name] = {"n_states": disc.n_states, "final_greedy_mean": float(finals.mean()),
                         "final_greedy_std": float(finals.std(ddof=1)), "n_runs": int(finals.size)}
    decision = select_by_parsimony(
        {k: {"mean": v["final_greedy_mean"], "std": v["final_greedy_std"], "n_states": v["n_states"]} for k, v in summary.items()}
    )
    return {"per_discretization": summary, **decision}


def _test_job(job: tuple[str, str, int, str, list]) -> dict:
    method, disc_name, seed, variant, q_table = job
    policy = GreedyTabularPolicy(np.array(q_table), DISCRETIZATIONS[disc_name])
    return {"method": method, "seed": seed, "variant": variant, **sanity_statistics(policy, load_params())}


def _test_summary(tests: list[dict]) -> dict:
    def mean_std(values):
        v = np.asarray(values, dtype=np.float64)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1))}

    return {
        "survival_rate": mean_std([t["survival_rate"] for t in tests]),
        "stabilized_rate": mean_std([t["stabilized_rate"] for t in tests]),
        "effort_Ns": mean_std([t["control_effort_abs_Ns"]["mean"] for t in tests]),
        "fraction_at_force_limit": mean_std([t["fraction_of_steps_at_force_limit"]["mean"] for t in tests]),
        "termination_reasons_total": {k: sum(t["termination_reasons"].get(k, 0) for t in tests)
                                      for k in {r for t in tests for r in t["termination_reasons"]}},
    }


def summarize_method(results: list[dict], tests: list[dict], method: str) -> dict:
    runs = [r for r in results if r["method"] == method]
    solved = [r["episodes_to_solve_greedy"] for r in runs]
    classic = [r["episodes_to_solve_classic"] for r in runs]

    def mean_std(values):
        v = np.asarray(values, dtype=np.float64)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1))}

    return {
        "n_seeds": len(runs),
        "final_greedy_validation": mean_std([r["final_greedy"] for r in runs]),
        "moving_average_max": mean_std([r["moving_average_max"] for r in runs]),
        "seeds_solved_greedy": sum(s is not None for s in solved),
        "episodes_to_solve_greedy": [s for s in solved if s is not None],
        "seeds_solved_classic": sum(s is not None for s in classic),
        "episodes_to_solve_classic": [s for s in classic if s is not None],
        "best_checkpoint_episode": mean_std([r["best_eval_episode"] for r in runs]),
        "best_checkpoint_validation": mean_std([r["best_eval_mean"] for r in runs]),
        # TEST (semillas 1000-1049): política al terminar vs mejor política vista en validación.
        "test_final_policy": _test_summary([t for t in tests if t["method"] == method and t["variant"] == "final"]),
        "test_best_checkpoint": _test_summary([t for t in tests if t["method"] == method and t["variant"] == "best"]),
    }


def plot_learning_curves(results: list[dict], chosen: str) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (method, label) in zip(axes, METHOD_NAMES.items()):
        runs = [r for r in results if r["method"] == method and r["discretization"] == chosen]
        episodes = np.array(runs[0]["eval_episodes"])
        curves = np.array([r["eval_mean"] for r in runs])
        mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1)
        color = METHOD_COLORS[label]
        for curve in curves:
            ax.plot(episodes, curve, color=color, alpha=0.18, linewidth=1)
        ax.plot(episodes, mean, color=color, label=f"media de {len(runs)} semillas")
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.15, linewidth=0, label="± 1 desv. típica")
        ax.axhline(SOLVED_THRESHOLD, color=MUTED, linestyle=":", linewidth=1)
        ax.text(episodes[-1], SOLVED_THRESHOLD + 6, "umbral 'resuelto' (475)", ha="right", fontsize=9, color=TEXT_SECONDARY)
        ax.set_title(label, loc="left", fontsize=11)
        ax.set_xlabel("episodios de entrenamiento")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("retorno greedy (pasos sin caer, máx. 500)")
    fig.suptitle(f"Aprendizaje tabular con {DISCRETIZATIONS[chosen].n_states} cajas: política greedy en 10 CI de validación",
                 fontsize=12, color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_curvas_tabulares.png", dpi=150)
    plt.close(fig)


def plot_discretization_study(selection: dict, results: list[dict]) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = list(DISCRETIZATIONS)
    for offset, (method, label) in zip((-0.12, 0.12), METHOD_NAMES.items()):
        for i, name in enumerate(names):
            finals = np.array([r["final_greedy"] for r in results if r["method"] == method and r["discretization"] == name])
            ax.scatter(np.full(finals.size, i + offset), finals, color=METHOD_COLORS[label], alpha=0.45, s=22)
            ax.errorbar(i + offset, finals.mean(), yerr=finals.std(ddof=1), fmt="o", color=METHOD_COLORS[label],
                        markersize=8, capsize=4, label=label if i == 0 else None)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n({DISCRETIZATIONS[n].n_states} cajas)" for n in names])
    ax.axhline(SOLVED_THRESHOLD, color=MUTED, linestyle=":", linewidth=1)
    ax.set_ylabel("retorno greedy final (validación)")
    ax.set_title(f"¿Cuántas cajas? Elegida: {selection['chosen']} (criterio declarado de antemano)", loc="left", fontsize=11)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_discretizaciones.png", dpi=150)
    plt.close(fig)


def plot_policy_map(results: list[dict], chosen: str) -> None:
    """Acción greedy aprendida en el plano (θ, θ̇), con el carro centrado y quieto: ¿qué 'reglas' aprendió?"""
    disc = DISCRETIZATIONS[chosen]
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    th_edges, w_edges = np.degrees(disc.edges[2]), np.degrees(disc.edges[3])
    th_lim, w_lim = 12.0, max(100.0, 1.6 * w_edges[-1])
    th_b = np.concatenate([[-th_lim], th_edges, [th_lim]])
    w_b = np.concatenate([[-w_lim], w_edges, [w_lim]])
    for ax, (method, label) in zip(axes, METHOD_NAMES.items()):
        runs = [r for r in results if r["method"] == method and r["discretization"] == chosen]
        best = max(runs, key=lambda r: r["best_eval_mean"])
        policy = GreedyTabularPolicy(np.array(best["best_q_table"]), disc)
        grid = np.array([[policy([0.0, 0.0, np.radians(0.5 * (th_b[i] + th_b[i + 1])), np.radians(0.5 * (w_b[j] + w_b[j + 1]))])
                          for j in range(len(w_b) - 1)] for i in range(len(th_b) - 1)])
        mesh = ax.pcolormesh(w_b, th_b, grid, cmap=FORCE_CMAP, vmin=-10, vmax=10, shading="flat")
        for i in range(len(th_b) - 1):
            for j in range(len(w_b) - 1):
                ax.text(0.5 * (w_b[j] + w_b[j + 1]), 0.5 * (th_b[i] + th_b[i + 1]), f"{grid[i, j]:+.0f}",
                        ha="center", va="center", fontsize=9, color=TEXT_PRIMARY)
        ax.set_title(f"{label}: semilla {best['seed']}, episodio {best['best_eval_episode']}\n"
                     "(mejor política vista en validación)", loc="left", fontsize=11)
        ax.set_xlabel("θ̇ [°/s]  (las cajas extremas llegan a ±∞)")
        ax.grid(False)
    axes[0].set_ylabel("θ [°]")
    fig.colorbar(mesh, ax=axes, label="fuerza elegida [N]")
    fig.suptitle("Política aprendida (corte con el carro centrado y quieto): fuerza greedy en cada caja de (θ, θ̇)",
                 fontsize=12, color=TEXT_SECONDARY, y=1.08)
    fig.savefig(FIGURES_DIR / "08_politica_aprendida.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_weights(chosen_runs: list[dict], chosen: str) -> None:
    """Por método: la MEJOR política vista en validación de la semilla con mejor validación."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for method in METHOD_NAMES:
        best = max((r for r in chosen_runs if r["method"] == method), key=lambda r: r["best_eval_mean"])
        payload = {
            "method": method, "seed": best["seed"], "discretization": chosen,
            "edges": [list(e) for e in DISCRETIZATIONS[chosen].edges], "force_levels": list(FORCE_LEVELS),
            "q_table": best["best_q_table"], "checkpoint_episode": best["best_eval_episode"],
            "validation_mean_return": best["best_eval_mean"],
            "selected_by": "mejor política vista en VALIDACIÓN (semillas 2000-2009), nunca en test",
        }
        (WEIGHTS_DIR / f"tabular_{method}.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def regenerate_artifacts() -> int:
    """Reentrena SOLO las semillas seleccionadas (determinista) para regenerar pesos y mapa de política.
    Falla si los números no coinciden EXACTAMENTE con results/rl/tabular.json (prueba de reproducibilidad)."""
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    chosen = record["selection"]["chosen"]
    n_episodes, eval_every = record["n_episodes"], record["configs"]["q_learning"]["eval_every"]
    regenerated = []
    for method in METHOD_NAMES:
        runs = [r for r in record["runs"] if r["method"] == method and r["discretization"] == chosen]
        target = max(runs, key=lambda r: r["best_eval_mean"])
        fresh = _train_job((method, chosen, target["seed"], n_episodes, eval_every))
        for key in ("final_greedy", "best_eval_mean", "best_eval_episode", "eval_mean"):
            if fresh[key] != target[key]:
                raise RuntimeError(f"No reproducible: {method} semilla {target['seed']} difiere en {key}")
        regenerated.append(fresh)
        print(f"{method} semilla {target['seed']}: reproducido exactamente (best {fresh['best_eval_mean']} en ep {fresh['best_eval_episode']})")
    export_weights(regenerated, chosen)
    plot_policy_map(regenerated, chosen)
    return 0


def main(n_episodes: int = N_EPISODES, eval_every: int = EVAL_EVERY, seeds: tuple[int, ...] = SEEDS) -> int:
    jobs = [(m, d, s, n_episodes, eval_every) for d in DISCRETIZATIONS for m in METHOD_NAMES for s in seeds]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(_train_job, jobs))
    print(f"Entrenamiento: {len(jobs)} ejecuciones en {time.perf_counter() - start:.0f} s")

    selection = select_discretization(results)
    chosen = selection["chosen"]
    print(json.dumps(selection, indent=1))
    chosen_runs = [r for r in results if r["discretization"] == chosen]
    with ProcessPoolExecutor(max_workers=12) as pool:
        test_jobs = [(r["method"], chosen, r["seed"], variant, r[key]) for r in chosen_runs
                     for variant, key in (("final", "q_table"), ("best", "best_q_table"))]
        tests = list(pool.map(_test_job, test_jobs))

    record = {
        "protocol": __doc__, "n_episodes": n_episodes, "seeds": list(seeds),
        "configs": {m: config_for(m, n_episodes, eval_every).to_dict() for m in METHOD_NAMES},
        "force_levels": list(FORCE_LEVELS), "selection": selection,
        "methods": {m: summarize_method(chosen_runs, tests, m) for m in METHOD_NAMES},
        "runs": [{k: v for k, v in r.items() if k not in ("q_table", "best_q_table")} for r in results],
        "versions": {"gymnasium": gym.__version__, "numpy": np.__version__},
        "generated_by": "scripts/train_tabular.py",
    }
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    export_weights(chosen_runs, chosen)
    plot_learning_curves(results, chosen)
    plot_discretization_study(selection, results)
    plot_policy_map(results, chosen)
    print(json.dumps(record["methods"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(regenerate_artifacts() if "--regenerate-artifacts" in sys.argv else main())
