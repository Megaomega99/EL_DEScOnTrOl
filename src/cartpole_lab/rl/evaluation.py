"""Métricas de aprendizaje comunes a los métodos de RL (se reutilizarán con DQN y actor-crítico).

"Episodios para aprender": CartPole-v1 declara en su registro un `reward_threshold` de 475.
Se dan dos definiciones, porque miden cosas distintas:
* greedy : primer punto de evaluación en que la política GREEDY (sin exploración) promedia
           ≥ 475 en las CI de validación. Mide cuándo la política aprendida es buena.
* clásico: primer episodio en que la media móvil de 100 episodios de ENTRENAMIENTO (con
           exploración) llega a 475. Es la definición del antiguo leaderboard de Gym.
None si no se alcanza: se reporta así, nunca se fuerza.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

SOLVED_THRESHOLD = 475.0  # reward_threshold oficial de CartPole-v1 (gymnasium/envs/__init__.py)
MOVING_WINDOW = 100


def first_crossing(values: Sequence[float], threshold: float = SOLVED_THRESHOLD) -> int | None:
    """Índice de la primera evaluación con valor ≥ umbral (None si nunca)."""
    solved = np.flatnonzero(np.asarray(values) >= threshold)
    return int(solved[0]) if solved.size else None


def episodes_to_solve_greedy(
    eval_episodes: Sequence[int], eval_means: Sequence[float], threshold: float = SOLVED_THRESHOLD
) -> int | None:
    index = first_crossing(eval_means, threshold)
    return int(np.asarray(eval_episodes)[index]) if index is not None else None


def moving_average(values: Sequence[float], window: int = MOVING_WINDOW) -> np.ndarray:
    """moving[i] = media de los episodios i .. i+window−1 (0-indexados)."""
    return np.convolve(np.asarray(values, dtype=np.float64), np.ones(window) / window, mode="valid")


def episodes_to_solve_moving_average(
    episode_returns: Sequence[float], window: int = MOVING_WINDOW, threshold: float = SOLVED_THRESHOLD
) -> int | None:
    """Número de episodios (1-indexado) tras el que la media móvil alcanza el umbral por primera vez."""
    solved = np.flatnonzero(moving_average(episode_returns, window) >= threshold)
    return int(solved[0] + window) if solved.size else None


def select_by_parsimony(candidates: Mapping[str, Mapping[str, float]]) -> dict:
    """Regla declarada de antemano: gana la mejor media; si otro candidato queda a menos de UNA
    desviación típica de ella y tiene menos estados, gana el más pequeño.

    Se usa la desviación del MEJOR (entre semillas) como tolerancia: pregunta "¿es la ventaja del
    mejor mayor que su propia variabilidad?". Es una elección; usar una desviación agrupada sería
    otra igual de defendible.
    `candidates[nombre]` debe tener las claves 'mean', 'std' y 'n_states'.
    """
    best = max(candidates, key=lambda k: candidates[k]["mean"])
    tolerance = candidates[best]["std"]
    within = [k for k in candidates if candidates[best]["mean"] - candidates[k]["mean"] <= tolerance]
    chosen = min(within, key=lambda k: candidates[k]["n_states"])
    return {"best_by_mean": best, "within_one_std": within, "chosen": chosen}
