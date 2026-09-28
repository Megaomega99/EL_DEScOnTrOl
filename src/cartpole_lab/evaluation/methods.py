"""Qué política usa cada método en la evaluación (docs/12_protocolo_evaluacion.md §1) y cómo construirla.

Los controladores se construyen DENTRO de cada proceso de trabajo (el MPC compila su problema
de optimización al crearse), así que aquí solo se describen con `PolicySpec`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cartpole_lab.controllers.fuzzy import CascadeFuzzy, load_tuned_fuzzy_gains
from cartpole_lab.controllers.lqr import LQRController, design_default_lqr
from cartpole_lab.controllers.mpc import LinearMPC
from cartpole_lab.controllers.pid import CascadePID, load_tuned_gains
from cartpole_lab.evaluation.protocol import MAIN_METHODS, RL_SEEDS, SUPPLEMENTARY_METHODS
from cartpole_lab.params import CartPoleParams
from cartpole_lab.paths import POLICIES_DIR
from cartpole_lab.rl.actor_critic import ACPolicy
from cartpole_lab.rl.discretization import BoxDiscretizer
from cartpole_lab.rl.dqn import DQNPolicy, load_network
from cartpole_lab.rl.tabular import GreedyTabularPolicy
from cartpole_lab.rollout import Controller

# método -> (carpeta en results/policies, qué punto de control)
_RL_SOURCES = {
    "Q-learning": ("tabular_q_learning", "best"),
    "SARSA": ("tabular_sarsa", "best"),
    "DQN": ("dqn", "best"),
    "Actor-crítico": ("actor_critic", "best"),
    "Double DQN": ("double_dqn", "best"),
    "Actor-crítico (red final)": ("actor_critic", "final"),
}


@dataclass(frozen=True)
class PolicySpec:
    method: str
    seed: int | None  # None en los clásicos (un solo controlador)

    @property
    def is_rl(self) -> bool:
        return self.method in _RL_SOURCES

    @property
    def label(self) -> str:
        return self.method if self.seed is None else f"{self.method} · semilla {self.seed}"


def all_specs(include_supplementary: bool = True) -> list[PolicySpec]:
    methods = MAIN_METHODS + (SUPPLEMENTARY_METHODS if include_supplementary else ())
    return [PolicySpec(m, s) for m in methods for s in (RL_SEEDS if m in _RL_SOURCES else (None,))]


def rl_policy_path(spec: PolicySpec) -> Path:
    folder, which = _RL_SOURCES[spec.method]
    return POLICIES_DIR / folder / f"seed_{spec.seed:02d}_{which}.json"


def build_controller(spec: PolicySpec, params: CartPoleParams) -> Controller:
    if spec.method == "PID":
        return CascadePID(load_tuned_gains(), params)
    if spec.method == "LQR":
        return LQRController(design_default_lqr(params).K, params)
    if spec.method == "MPC":
        return LinearMPC(params)
    if spec.method == "Fuzzy":
        return CascadeFuzzy(load_tuned_fuzzy_gains(), params)
    if not spec.is_rl:
        raise ValueError(f"Método desconocido: {spec.method}")
    path = rl_policy_path(spec)
    if not path.exists():
        raise FileNotFoundError(f"Falta la política {path}: ejecutar los scripts de entrenamiento de la Fase 2")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "q_table" in data:
        discretizer = BoxDiscretizer(edges=tuple(tuple(e) for e in data["edges"]))
        return GreedyTabularPolicy(np.array(data["q_table"]), discretizer, tuple(data["force_levels"]))
    net = load_network(path)
    if "force_levels" in net:
        return DQNPolicy(net["layers"], tuple(net["input_scales"]), tuple(net["force_levels"]))
    return ACPolicy(net["layers"], tuple(net["input_scales"]), net["force_limit"])
