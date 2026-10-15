"""Utilidades de los notebooks docentes (Fase 4): lo que se repite en la versión completa y en la corta.

Por qué un módulo y no código en las celdas: las dos versiones del notebook dibujan las mismas
trayectorias y los mismos mapas de política, y cargan los mismos registros. Con una sola
implementación (probada en tests/test_teaching.py) las dos versiones no pueden divergir, y las
celdas quedan cortas y legibles durante la charla.

Nada de este módulo entrena ni sintoniza: solo simula controladores ya construidos, dibuja y lee
los resultados guardados en results/.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from cartpole_lab.env import CartPoleTask
from cartpole_lab.evaluation.methods import PolicySpec, build_controller
from cartpole_lab.evaluation.protocol import MAIN_METHODS
from cartpole_lab.params import CartPoleParams
from cartpole_lab.paths import FIGURES_DIR, RESULTS_DIR, WEIGHTS_DIR
from cartpole_lab.plotting import FORCE_CMAP, METHOD_COLORS, MUTED, TEXT_SECONDARY, apply_style
from cartpole_lab.rollout import Controller, Trajectory, run_episode

# Archivo de pesos exportado (docs/11_exportacion.md) de cada método de RL: de ahí sale la semilla
# que se enseña en la charla, en lugar de fijarla a mano aquí.
EXPORTED_WEIGHTS = {
    "Q-learning": "tabular_q_learning.json",
    "SARSA": "tabular_sarsa.json",
    "DQN": "dqn.json",
    "Actor-crítico": "actor_critic.json",
}


def load_result(relative_path: str) -> dict[str, Any]:
    """Lee un registro JSON de results/ (p. ej. "tuning/lqr.json"). Error explícito si falta."""
    path = RESULTS_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}: ejecutar antes el script que lo genera (ver README)")
    return json.loads(path.read_text(encoding="utf-8"))


def figure_path(name: str) -> Path:
    """Ruta de una figura ya generada en results/figures (p. ej. "13_tabla_resumen.png")."""
    path = FIGURES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"No existe la figura {path}")
    return path


def exported_seed(method: str) -> int:
    """Semilla de la política de RL exportada para el navegador (la que se enseña en la charla)."""
    data = json.loads((WEIGHTS_DIR / EXPORTED_WEIGHTS[method]).read_text(encoding="utf-8"))
    return int(data["metadata"]["seed"] if "metadata" in data else data["seed"])


def method_controllers(params: CartPoleParams, methods: tuple[str, ...] = MAIN_METHODS) -> dict[str, Controller]:
    """Un controlador por método, construido igual que en la evaluación de la Fase 3.

    Clásicos: los parámetros guardados en results/tuning. RL: la mejor política en validación de la
    semilla exportada (results/policies). No entrena ni sintoniza nada.
    """
    return {m: build_controller(PolicySpec(m, exported_seed(m) if m in EXPORTED_WEIGHTS else None), params)
            for m in methods}


def simulate(controllers: Mapping[str, Controller], *, initial_state=None, seed: int | None = None,
             task: CartPoleTask | None = None) -> dict[str, Trajectory]:
    """Un episodio por controlador, todos desde la misma condición inicial y en la misma tarea."""
    own_task = task is None
    task = CartPoleTask(action_mode="continuous") if own_task else task
    try:
        return {name: run_episode(task, c, seed=seed, initial_state=initial_state) for name, c in controllers.items()}
    finally:
        if own_task:
            task.close()


def method_color(name: str) -> str:
    """Color del método (las variantes, p. ej. "Actor-crítico (red final)", heredan el de su familia)."""
    return METHOD_COLORS.get(name, METHOD_COLORS.get(name.split(" (")[0], TEXT_SECONDARY))


def plot_trajectories(trajectories: Mapping[str, Trajectory], params: CartPoleParams, *, title: str = "",
                      impulse_time: float | None = None) -> Figure:
    """x(t), θ(t) y F(t) de varios métodos en tres paneles, con los límites de fallo y del actuador."""
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for name, traj in trajectories.items():
        color = method_color(name)
        style = "--" if "(" in name else "-"
        label = name if traj.termination_reason == "time_limit" else f"{name}  → falla ({traj.termination_reason})"
        axes[0].plot(traj.times, traj.states[:, 0], color=color, linestyle=style, label=label)
        axes[1].plot(traj.times, np.degrees(traj.states[:, 2]), color=color, linestyle=style)
        axes[2].step(traj.times[:-1], traj.forces, where="post", color=color, linestyle=style, linewidth=1.3)
    limits = (params.x_threshold, np.degrees(params.theta_threshold_radians), params.force_mag)
    labels = (("x [m]", "Posición del carro (falla en ±2.4 m)"), ("θ [°]", "Ángulo del poste (falla en ±12°)"),
              ("F [N]", "Fuerza del actuador (límite ±10 N)"))
    for ax, limit, (ylabel, panel_title) in zip(axes, limits, labels):
        for sign in (+1, -1):
            ax.axhline(sign * limit, color=MUTED, linewidth=1, linestyle=":")
        if impulse_time is not None:
            ax.axvline(impulse_time, color=MUTED, linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title, loc="left", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9, ncol=2)
    axes[2].set_xlabel("t [s]")
    if title:
        fig.suptitle(title, fontsize=12, color=TEXT_SECONDARY)
    fig.tight_layout()
    return fig


def force_map(policy: Callable[[np.ndarray], float], params: CartPoleParams, *, n: int = 81,
              theta_dot_max: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuerza que pide `policy` en el plano (θ, θ̇) con el carro centrado y quieto (x = ẋ = 0).

    Es el mismo corte que las figuras 06, 08, 09 y 10: permite comparar "qué reglas" usa cada método.
    Si la política tiene estado interno (la integral del PID), se reinicia antes de cada estado.
    Devuelve (θ en grados, θ̇ en rad/s, F) con F de forma (n, n), filas = θ.
    """
    theta = np.linspace(-params.theta_threshold_radians, params.theta_threshold_radians, n)
    theta_dot = np.linspace(-theta_dot_max, theta_dot_max, n)
    reset = getattr(policy, "reset", None)
    force = np.empty((n, n))
    for i, th in enumerate(theta):
        for j, w in enumerate(theta_dot):
            if reset is not None:
                reset()
            force[i, j] = float(policy(np.array([0.0, 0.0, th, w])))
    return np.degrees(theta), theta_dot, force


def plot_force_maps(policies: Mapping[str, Callable[[np.ndarray], float]], params: CartPoleParams, *,
                    n: int = 81, title: str = "") -> Figure:
    """Un mapa F(θ, θ̇) por política, con la misma escala de color (azul = izquierda, rojo = derecha)."""
    apply_style()
    fig, axes = plt.subplots(1, len(policies), figsize=(4.2 * len(policies), 4.0), sharey=True, squeeze=False)
    mesh = None
    for ax, (name, policy) in zip(axes[0], policies.items()):
        theta_deg, theta_dot, force = force_map(policy, params, n=n)
        mesh = ax.pcolormesh(theta_dot, theta_deg, force, cmap=FORCE_CMAP, vmin=-params.force_mag,
                             vmax=params.force_mag, shading="auto")
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_xlabel("θ̇ [rad/s]")
        ax.grid(False)
    axes[0, 0].set_ylabel("θ [°]")
    fig.colorbar(mesh, ax=axes[0].tolist(), label="F [N]")
    if title:
        fig.suptitle(title, fontsize=12, color=TEXT_SECONDARY, y=1.04)
    return fig


def compare_with_record(live, recorded, *, rtol: float = 1e-9, atol: float = 1e-9) -> str:
    """Frase que dice si un resultado recalculado coincide con el registro, y si no, cuánto difiere.

    No se exige igualdad: el registro se generó en otra máquina, y en otro sistema operativo o con
    otra biblioteca matemática el redondeo puede cambiar un entrenamiento caótico. Se informa.
    """
    live_arr, rec_arr = np.asarray(live, dtype=np.float64), np.asarray(recorded, dtype=np.float64)
    if live_arr.shape != rec_arr.shape:
        return f"no comparable: forma {live_arr.shape} frente a {rec_arr.shape} en el registro"
    if np.allclose(live_arr, rec_arr, rtol=rtol, atol=atol):
        return "coincide con el registro"
    differing = np.flatnonzero(~np.isclose(live_arr, rec_arr, rtol=rtol, atol=atol).reshape(-1))
    return (f"difiere del registro en {differing.size} de {live_arr.size} valores "
            f"(máxima diferencia relativa {np.max(np.abs(live_arr - rec_arr) / np.maximum(np.abs(rec_arr), 1e-12)):.2g})")


def plot_live_vs_recorded(x_recorded, recorded_curves, x_live, live_curve, *, color: str, title: str,
                          xlabel: str, ylabel: str = "retorno greedy en validación (máx. 500)",
                          threshold: float | None = 475.0, ax=None):
    """Curvas de aprendizaje: cada semilla registrada (fina), su media y la ejecución en vivo (gruesa)."""
    apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    curves = np.asarray(recorded_curves, dtype=np.float64)
    for curve in curves:
        ax.plot(x_recorded, curve, color=color, alpha=0.15, linewidth=1)
    ax.plot(x_recorded, curves.mean(axis=0), color=color, alpha=0.6, linestyle="--",
            label=f"media de las {len(curves)} semillas del registro")
    ax.plot(x_live, live_curve, color=color, linewidth=2.5, label="ejecución en vivo")
    if threshold is not None:
        ax.axhline(threshold, color=MUTED, linestyle=":", linewidth=1)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right", fontsize=9)
    return ax


def summary_table_markdown() -> str:
    """La tabla resumen de la comparación (results/metrics/summary_table.md, generada por make_figures.py)."""
    path = RESULTS_DIR / "metrics" / "summary_table.md"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}: ejecutar scripts/make_figures.py")
    return path.read_text(encoding="utf-8")
