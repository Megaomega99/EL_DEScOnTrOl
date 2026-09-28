"""Paso 12: evaluación comparativa de TODOS los métodos, según docs/12_protocolo_evaluacion.md.

Uso:
    python scripts/evaluate_all.py                  evaluación completa (~10 min con 12 procesos)
    python scripts/evaluate_all.py --quick          prueba de humo (pocas CI; no escribe resultados)

Salida: results/metrics/comparison.json   (resumen por método + mallas + datos por episodio del nominal)
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import numpy as np

from cartpole_lab import CartPoleTask, ForcePulse, load_params, run_episode
from cartpole_lab.evaluation.methods import PolicySpec, all_specs, build_controller
from cartpole_lab.evaluation.metrics import across_seeds, aggregate_episodes, episode_metrics, grid_outcome
from cartpole_lab.evaluation.protocol import (
    GRID_N, IMPULSE_STEP, IMPULSES_NS, MAIN_METHODS, NOMINAL_SEEDS, SUPPLEMENTARY_METHODS,
    cart_grid_states, impulse_sign, pole_grid_states,
)
from cartpole_lab.paths import RESULTS_DIR

OUTPUT = RESULTS_DIR / "metrics" / "comparison.json"
N_WORKERS = 12
GRID_CHUNK = 49
GRIDS = {"pole_grid": pole_grid_states, "cart_grid": cart_grid_states}
SUMMARY_KEYS = ("survival_rate", "settled_rate", "pole_settled_rate", "settling_time_s", "pole_settling_time_s",
                "effort_abs_Ns", "fraction_at_force_limit", "cost")
# Un hilo de BLAS por proceso: 12 procesos × N hilos sobresuscribirían la CPU. Los procesos hijos (spawn en
# Windows) heredan el entorno al crearse e importan numpy de nuevo, así que basta con fijarlo antes del pool.
SINGLE_THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


class Job(NamedTuple):
    method: str
    seed: int | None
    scenario: str  # "nominal" | "impulse" | nombre de malla
    arg: float | int | None  # impulso J [N·s] o índice de inicio del trozo de malla
    seeds: tuple[int, ...] | None  # semillas de CI (nominal e impulso)


def build_jobs(specs: list[PolicySpec], seeds: list[int], grid_limit: int | None) -> list[Job]:
    jobs = []
    for spec in specs:
        jobs.append(Job(spec.method, spec.seed, "nominal", None, tuple(seeds)))
        jobs += [Job(spec.method, spec.seed, "impulse", J, tuple(seeds)) for J in IMPULSES_NS]
        for grid, states_fn in GRIDS.items():
            n = len(states_fn()) if grid_limit is None else grid_limit
            jobs += [Job(spec.method, spec.seed, grid, start, None) for start in range(0, n, GRID_CHUNK)]
    return sorted(jobs, key=lambda j: j.method != "MPC")  # el MPC (lento) primero: mejor reparto de carga


def _job(job: Job) -> dict:
    params = load_params()
    controller = build_controller(PolicySpec(job.method, job.seed), params)
    out = {"method": job.method, "seed": job.seed, "scenario": job.scenario, "arg": job.arg}
    if job.scenario == "nominal":
        task = CartPoleTask(action_mode="continuous")
        out["episodes"] = [episode_metrics(run_episode(task, controller, seed=s)) for s in job.seeds]
    elif job.scenario == "impulse":
        tasks = {sign: CartPoleTask(action_mode="continuous",
                                    disturbance=ForcePulse(IMPULSE_STEP, 1, sign * job.arg / params.tau))
                 for sign in (1.0, -1.0)}
        episodes = [episode_metrics(run_episode(tasks[impulse_sign(i)], controller, seed=s), impulse_step=IMPULSE_STEP)
                    for i, s in enumerate(job.seeds)]
        out["summary"] = aggregate_episodes(episodes)
    else:
        task = CartPoleTask(action_mode="continuous")
        states = GRIDS[job.scenario]()[job.arg:job.arg + GRID_CHUNK]
        out["outcomes"] = [grid_outcome(run_episode(task, controller, initial_state=s)) for s in states]
    return out


def _j50(survival_by_j: dict[float, float]) -> float:
    """Mayor impulso del barrido con supervivencia ≥ 50 % (0 si ninguno)."""
    ok = [J for J, rate in survival_by_j.items() if rate >= 0.5]
    return max(ok) if ok else 0.0


def summarize_method(method: str, results: list[dict]) -> dict:
    mine = [r for r in results if r["method"] == method]
    seeds = sorted({r["seed"] for r in mine}, key=lambda s: -1 if s is None else s)
    per_seed = {}
    for seed in seeds:
        rs = [r for r in mine if r["seed"] == seed]
        nominal = aggregate_episodes(next(r for r in rs if r["scenario"] == "nominal")["episodes"])
        impulse = {r["arg"]: r["summary"] for r in rs if r["scenario"] == "impulse"}
        grids = {g: [o for r in sorted((r for r in rs if r["scenario"] == g), key=lambda r: r["arg"]) for o in r["outcomes"]]
                 for g in GRIDS}
        per_seed[seed] = {"nominal": nominal, "impulse": impulse, "grids": grids,
                          "j50": _j50({J: s["survival_rate"] for J, s in impulse.items()})}
    return _combine(method, per_seed)


def _grid_stats(outcomes_by_seed: list[list[int]]) -> dict:
    arr = np.array(outcomes_by_seed)  # (semillas, celdas)
    return {"stabilized_fraction_per_cell": (arr == 2).mean(axis=0).reshape(GRID_N, -1).tolist() if arr.shape[1] == GRID_N ** 2 else None,
            "survived_fraction_per_cell": (arr >= 1).mean(axis=0).reshape(GRID_N, -1).tolist() if arr.shape[1] == GRID_N ** 2 else None,
            "stabilized_area": _ms((arr == 2).mean(axis=1)), "survived_area": _ms((arr >= 1).mean(axis=1))}


def _ms(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}


def _combine(method: str, per_seed: dict) -> dict:
    seeds = list(per_seed)
    is_rl = seeds != [None]
    if is_rl:
        nominal = across_seeds([per_seed[s]["nominal"] for s in seeds], SUMMARY_KEYS)
    else:
        nominal = per_seed[None]["nominal"]
    impulses = sorted(per_seed[seeds[0]]["impulse"])
    impulse = {str(J): (across_seeds([per_seed[s]["impulse"][J] for s in seeds],
                                     ("survival_rate", "recovery_time_s", "peak_abs_theta_after_deg"))
                        if is_rl else per_seed[None]["impulse"][J]) for J in impulses}
    return {
        "method": method, "is_rl": is_rl, "n_seeds": len(seeds) if is_rl else None,
        "statistic": "media ± desv. típica ENTRE SEMILLAS (N = 10)" if is_rl else "entre condiciones iniciales (1 controlador)",
        "nominal": nominal, "impulse": impulse, "j50": _ms([per_seed[s]["j50"] for s in seeds]),
        "grids": {g: _grid_stats([per_seed[s]["grids"][g] for s in seeds]) for g in GRIDS},
        "per_seed_nominal": {str(s): per_seed[s]["nominal"] for s in seeds} if is_rl else None,
    }


def learning_costs() -> dict:
    """Coste de aprendizaje del RL, tomado de los registros de la Fase 2 (no se recalcula)."""
    rl = RESULTS_DIR / "rl"
    tab = json.loads((rl / "tabular.json").read_text(encoding="utf-8"))
    dqn = json.loads((rl / "dqn.json").read_text(encoding="utf-8"))
    ac = json.loads((rl / "actor_critic.json").read_text(encoding="utf-8"))
    chosen = tab["selection"]["chosen"]
    out = {}
    for method, name in (("q_learning", "Q-learning"), ("sarsa", "SARSA")):
        runs = [r for r in tab["runs"] if r["method"] == method and r["discretization"] == chosen]
        out[name] = {"episodes_to_first_greedy_475": [r["episodes_to_solve_greedy"] for r in runs],
                     "episodes_to_moving_average_475": [r["episodes_to_solve_classic"] for r in runs],
                     "training_budget": f"{tab['n_episodes']} episodios"}
    for variant, name in (("dqn", "DQN"), ("double_dqn", "Double DQN")):
        runs = [r for r in dqn["runs"] if r["variant"] == variant]
        out[name] = {"episodes_to_first_greedy_475": [r["episodes_to_solve_greedy"] for r in runs],
                     "steps_to_first_greedy_475": [r["steps_to_solve_greedy"] for r in runs],
                     "episodes_to_moving_average_475": [r["episodes_to_solve_classic"] for r in runs],
                     "training_budget": f"{dqn['configs'][variant]['total_steps']} pasos"}
    out["Actor-crítico"] = {"steps_to_first_greedy_475": [r["steps_to_solve_greedy"] for r in ac["runs"]],
                            "episodes_to_moving_average_475": [r["episodes_to_solve_classic"] for r in ac["runs"]],
                            "training_budget": f"{ac['config']['total_steps']} pasos"}
    return out


def main(quick: bool = False) -> int:
    specs = all_specs()
    seeds = list(NOMINAL_SEEDS)[:4] if quick else list(NOMINAL_SEEDS)
    if quick:
        specs = [s for s in specs if s.seed in (None, 0)]
    jobs = build_jobs(specs, seeds, grid_limit=GRID_CHUNK if quick else None)
    os.environ.update(SINGLE_THREAD_ENV)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(_job, jobs, chunksize=1))
    elapsed = time.perf_counter() - start
    methods = [m for m in MAIN_METHODS + SUPPLEMENTARY_METHODS if any(r["method"] == m for r in results)]
    summary = {m: summarize_method(m, results) for m in methods}
    print(f"{len(jobs)} tareas en {elapsed:.0f} s")
    for m, s in summary.items():
        n = s["nominal"]
        surv = n["survival_rate"]["mean"] if s["is_rl"] else n["survival_rate"]
        print(f"  {m:28s} supervivencia nominal {surv:5.2f} | J50 {s['j50']['mean']:.2f} N·s | "
              f"área estabilizada (poste) {s['grids']['pole_grid']['stabilized_area']['mean']:.2f}")
    if quick:
        print("(--quick: no se escriben resultados)")
        return 0
    record = {"protocol": "docs/12_protocolo_evaluacion.md", "n_jobs": len(jobs), "wall_time_s": elapsed,
              "nominal_seeds": [seeds[0], seeds[-1]], "impulses_Ns": list(IMPULSES_NS), "grid_n": GRID_N,
              "methods": summary, "learning_costs": learning_costs(),
              "nominal_episodes": {f"{r['method']}|{r['seed']}": r["episodes"] for r in results if r["scenario"] == "nominal"}}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Guardado {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(quick="--quick" in sys.argv))
