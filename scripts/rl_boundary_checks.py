"""Comprobación común de las políticas de RL EXPORTADAS fuera de la distribución de test.

Uso:  python scripts/rl_boundary_checks.py      (segundos)

El test común (semillas 1000–1049) empieza siempre cerca del equilibrio (±0.05, la distribución
oficial de CartPole-v1). Aquí se aplican a TODAS las políticas de RL exportadas las mismas pruebas
que a los controladores de la Fase 1:
* el estado difícil de referencia (x = 0.5 m, ẋ = 0.2 m/s, θ = 0.1 rad, θ̇ = 0.2 rad/s);
* los 12 estados recuperables cerca de la frontera (NEAR_BOUNDARY_INITIAL_STATES);
* los 2 estados irrecuperables: deben fallar de forma EXPLÍCITA (motivo registrado).

Salida: results/rl/boundary_checks.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

from cartpole_lab import CartPoleTask, load_params, run_episode
from cartpole_lab.paths import RESULTS_DIR, WEIGHTS_DIR
from cartpole_lab.rl.actor_critic import ACPolicy
from cartpole_lab.rl.discretization import BoxDiscretizer
from cartpole_lab.rl.dqn import DQNPolicy, load_network
from cartpole_lab.rl.tabular import GreedyTabularPolicy
from cartpole_lab.sanity import (
    NEAR_BOUNDARY_INITIAL_STATES,
    REFERENCE_HARD_INITIAL_STATE,
    UNRECOVERABLE_INITIAL_STATES,
    is_stabilized,
)

OUTPUT = RESULTS_DIR / "rl" / "boundary_checks.json"


def load_policies() -> dict:
    policies = {}
    for method in ("q_learning", "sarsa"):
        w = json.loads((WEIGHTS_DIR / f"tabular_{method}.json").read_text(encoding="utf-8"))
        disc = BoxDiscretizer(edges=tuple(tuple(e) for e in w["edges"]))
        policies[f"tabular_{method}"] = GreedyTabularPolicy(np.array(w["q_table"]), disc, tuple(w["force_levels"]))
    for method in ("dqn", "double_dqn"):
        net = load_network(WEIGHTS_DIR / f"{method}.json")
        policies[method] = DQNPolicy(net["layers"], tuple(net["input_scales"]), tuple(net["force_levels"]))
    for name in ("actor_critic", "actor_critic_final"):
        net = load_network(WEIGHTS_DIR / f"{name}.json")
        policies[name] = ACPolicy(net["layers"], tuple(net["input_scales"]), net["force_limit"])
    return policies


def check(policy, params, task: CartPoleTask) -> dict:
    hard = run_episode(task, policy, initial_state=REFERENCE_HARD_INITIAL_STATE)
    near = [run_episode(task, policy, initial_state=s) for s in NEAR_BOUNDARY_INITIAL_STATES]
    extreme = {n: run_episode(task, policy, initial_state=s) for n, (s, _) in UNRECOVERABLE_INITIAL_STATES.items()}
    return {
        "reference_hard": {"reason": hard.termination_reason, "stabilized": is_stabilized(hard, params)},
        "near_boundary_survived": int(sum(t.termination_reason == "time_limit" for t in near)),
        "near_boundary_stabilized": int(sum(is_stabilized(t, params) for t in near)),
        "near_boundary_reasons": [t.termination_reason for t in near],
        "unrecoverable_fail_explicitly": all(
            t.terminated and t.termination_reason in UNRECOVERABLE_INITIAL_STATES[n][1] and np.all(np.isfinite(t.states))
            for n, t in extreme.items()),
    }


def main() -> int:
    params = load_params()
    task = CartPoleTask(action_mode="continuous")
    try:
        results = {name: check(policy, params, task) for name, policy in load_policies().items()}
    finally:
        task.close()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"protocol": __doc__, "results": results}, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    for name, r in results.items():
        print(f"{name:22s} referencia: {r['reference_hard']['reason']:<20s} frontera: sobrevive {r['near_boundary_survived']:2d}/12,"
              f" estabiliza {r['near_boundary_stabilized']:2d}/12 | irrecuperables fallan explícitamente: {r['unrecoverable_fail_explicitly']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
