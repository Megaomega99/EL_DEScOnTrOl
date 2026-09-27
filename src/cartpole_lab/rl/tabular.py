r"""Q-learning y SARSA tabulares (paso 8) sobre las cajas de discretization.py.

ALGORITMOS
----------
Ambos aprenden Q(s, a), el retorno esperado al tomar la acción a en la caja s, con la
misma actualización por diferencia temporal (TD):

    Q(s, a) ← Q(s, a) + α · [ objetivo − Q(s, a) ]

y solo difieren en el objetivo:

    Q-learning (off-policy):  r + γ · max_a' Q(s', a')      ¿qué pasaría si a partir de ahora
                                                             actuase de forma óptima?
    SARSA      (on-policy) :  r + γ · Q(s', a'), con a' la acción que REALMENTE tomará
                             (incluida la exploración): aprende el valor de la política que sigue.

Fin de episodio (detalle que se equivoca a menudo; Pardo et al., 2018):
* Terminación (el poste cae o el carro sale): no hay futuro -> objetivo = r.
* Truncamiento por tiempo (500 pasos): el sistema NO ha fallado; cortar el episodio es
  un artificio del experimento -> se sigue usando γ·Q(s', ·) (bootstrap).

ACCIONES: los 5 niveles de fuerza acordados {−10, −5, 0, 5, 10} N sobre el actuador
continuo común. Número impar para incluir 0 N ("no hacer nada").

RECOMPENSA: la de CartPole-v1 sin tocar (+1 por paso). No se añade penalización del
esfuerzo; ver docs/01_entorno_cartpole.md §7.1.

HIPERPARÁMETROS Y JUSTIFICACIÓN
-------------------------------
* γ = 0.99. Horizonte efectivo 1/(1−γ) = 100 pasos = 2 s: 8 veces la constante de tiempo
  con la que cae el poste sin control (0.25 s), y del orden del modo lento del carro
  (~1 s en el LQR). El agente "ve" las consecuencias de sus actos lo bastante lejos como
  para evitar caídas que se gestan en uno o dos segundos. Además es el valor estándar.
* α = 0.1, constante. SUPUESTO heurístico. La teoría de convergencia (Robbins–Monro)
  exige un α decreciente, y aquí NO se cumple. Motivo: al agrupar estados en cajas,
  una misma caja contiene estados distintos y las transiciones parecen aleatorias; un
  α constante y moderado promedia ese ruido y sigue a una política que va cambiando.
* ε-greedy: ε baja linealmente de 1.0 a 0.01 durante la primera mitad del
  entrenamiento y luego se mantiene. Al principio explorar es obligatorio (Q no sabe
  nada); al final, casi siempre explotar. ε = 0.01 > 0 sigue visitando cajas raras.
  El calendario es una elección, no un estándar.
* Q inicial = 0. Las recompensas son +1, así que el valor real está entre 1 y 100 y 0 es
  "pesimista"; la exploración la aporta ε, no el optimismo inicial.
* Empates en argmax: al azar durante el entrenamiento (con Q = 0 al inicio, argmax
  elegiría siempre −10 N). En la política final se deshacen hacia la fuerza más suave:
  "ante la duda, no empujes".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from cartpole_lab.env import CartPoleTask
from cartpole_lab.rl.discretization import BoxDiscretizer
from cartpole_lab.rollout import run_episode

FORCE_LEVELS = (-10.0, -5.0, 0.0, 5.0, 10.0)
METHODS = ("q_learning", "sarsa")


@dataclass(frozen=True)
class TabularConfig:
    method: str
    n_episodes: int = 3000
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_fraction: float = 0.5  # ε llega a su mínimo a mitad del entrenamiento
    q_init: float = 0.0
    eval_every: int = 100  # evaluación greedy periódica (curva de aprendizaje sin ruido de ε)
    n_eval_episodes: int = 10
    eval_seed_start: int = 2000  # CI de VALIDACIÓN, disjuntas del test (1000–1049)
    force_levels: tuple[float, ...] = FORCE_LEVELS

    def __post_init__(self) -> None:
        checks = {
            "method": self.method in METHODS,
            "n_episodes": self.n_episodes >= 1,
            "alpha": 0.0 < self.alpha <= 1.0,
            "gamma": 0.0 <= self.gamma < 1.0,
            "epsilon": 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0,
            "epsilon_decay_fraction": 0.0 < self.epsilon_decay_fraction <= 1.0,
            # eval_every <= n_episodes: si no, no habría ninguna evaluación y la "mejor política"
            # sería en silencio la tabla sin entrenar.
            "eval": 1 <= self.eval_every <= self.n_episodes and self.n_eval_episodes >= 1,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"Configuración tabular inválida en: {failed}")

    def to_dict(self) -> dict:
        return asdict(self)


def epsilon_at(episode: int, config: TabularConfig) -> float:
    decay_episodes = max(1, int(round(config.epsilon_decay_fraction * config.n_episodes)))
    progress = min(1.0, episode / decay_episodes)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def td_target(method: str, reward: float, q_next: np.ndarray, next_action: int, gamma: float, terminated: bool) -> float:
    """Objetivo de la actualización TD. Es la ÚNICA diferencia entre Q-learning y SARSA."""
    if terminated:
        return reward
    bootstrap = np.max(q_next) if method == "q_learning" else q_next[next_action]
    return reward + gamma * float(bootstrap)


def _epsilon_greedy(q_row: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_row.size))
    return int(rng.choice(np.flatnonzero(q_row == q_row.max())))  # empates al azar


class GreedyTabularPolicy:
    """Política final: la acción de mayor Q en la caja actual (controlador para run_episode)."""

    action_mode = "continuous"

    def __init__(
        self, q_table: np.ndarray, discretizer: BoxDiscretizer, force_levels: tuple[float, ...] = FORCE_LEVELS
    ) -> None:
        if q_table.shape != (discretizer.n_states, len(force_levels)):
            raise ValueError(f"Tabla Q de forma {q_table.shape}, se esperaba {(discretizer.n_states, len(force_levels))}")
        # Copia: la política no debe cambiar si alguien sigue modificando la tabla original
        # (durante el entrenamiento, evaluate_greedy recibe la tabla que se está aprendiendo).
        self._q = np.array(q_table, dtype=np.float64, copy=True)
        self._discretizer = discretizer
        self._forces = np.asarray(force_levels, dtype=np.float64)
        # Orden de preferencia en empates: primero las fuerzas más suaves, luego izquierda antes que derecha.
        self._tie_order = np.lexsort((self._forces, np.abs(self._forces)))

    def reset(self) -> None:
        pass

    def __call__(self, state: np.ndarray) -> float:
        row = self._q[self._discretizer.index(state)]
        best = self._tie_order[np.argmax(row[self._tie_order])]
        return float(self._forces[best])


@dataclass(frozen=True)
class TabularRun:
    config: TabularConfig
    seed: int
    q_table: np.ndarray  # (n_estados, n_acciones), de solo lectura
    episode_returns: np.ndarray  # (n_episodes,) retorno de entrenamiento (con exploración)
    episode_lengths: np.ndarray
    epsilons: np.ndarray
    eval_episodes: np.ndarray  # episodios tras los que se evaluó
    eval_returns: np.ndarray  # (n_evaluaciones, n_eval_episodes) retorno greedy
    # Mejor política vista en VALIDACIÓN (parada temprana). Con cajas, la política "tirita":
    # la última tabla no es necesariamente la mejor (ver docs/08_tabular.md).
    best_q_table: np.ndarray
    best_eval_episode: int
    best_eval_mean: float


def evaluate_greedy(q_table: np.ndarray, discretizer: BoxDiscretizer, config: TabularConfig) -> np.ndarray:
    policy = GreedyTabularPolicy(q_table, discretizer, config.force_levels)
    task = CartPoleTask(action_mode="continuous")
    try:
        seeds = range(config.eval_seed_start, config.eval_seed_start + config.n_eval_episodes)
        return np.array([run_episode(task, policy, seed=s).total_reward for s in seeds])
    finally:
        task.close()


def _run_episode_learning(
    task: CartPoleTask,
    q: np.ndarray,
    discretizer: BoxDiscretizer,
    config: TabularConfig,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[float, int]:
    """Un episodio de aprendizaje. Modifica `q` IN PLACE: excepción deliberada a la regla de
    inmutabilidad del proyecto (copiar la tabla en cada actualización sería prohibitivo)."""
    state, _ = task.reset()
    box = discretizer.index(state)
    action = _epsilon_greedy(q[box], epsilon, rng)
    total, steps, done = 0.0, 0, False
    while not done:
        state, reward, terminated, truncated, _ = task.step(config.force_levels[action])
        next_box = discretizer.index(state)
        next_action = _epsilon_greedy(q[next_box], epsilon, rng)
        target = td_target(config.method, reward, q[next_box], next_action, config.gamma, terminated)
        q[box, action] += config.alpha * (target - q[box, action])
        box, action = next_box, next_action
        total, steps, done = total + reward, steps + 1, terminated or truncated
    return total, steps


def train_tabular(config: TabularConfig, discretizer: BoxDiscretizer, seed: int) -> TabularRun:
    """Entrena desde cero. Reproducible: `seed` fija el RNG del agente y el del entorno."""
    rng = np.random.default_rng(seed)
    q = np.full((discretizer.n_states, len(config.force_levels)), config.q_init, dtype=np.float64)
    task = CartPoleTask(action_mode="continuous")
    returns, lengths, epsilons, eval_eps, eval_returns = [], [], [], [], []
    best_q, best_episode, best_mean = q.copy(), 0, -np.inf
    try:
        task.reset(seed=seed)  # siembra el RNG del entorno; los resets siguientes continúan su secuencia
        for episode in range(config.n_episodes):
            epsilon = epsilon_at(episode, config)
            total, steps = _run_episode_learning(task, q, discretizer, config, epsilon, rng)
            returns.append(total), lengths.append(steps), epsilons.append(epsilon)
            if (episode + 1) % config.eval_every == 0:
                eval_eps.append(episode + 1)
                eval_returns.append(evaluate_greedy(q, discretizer, config))
                if eval_returns[-1].mean() > best_mean:  # estricto: ante empate se queda la primera
                    best_q, best_episode, best_mean = q.copy(), episode + 1, float(eval_returns[-1].mean())
    finally:
        task.close()
    arrays = {
        "q_table": q, "best_q_table": best_q, "episode_returns": np.array(returns),
        "episode_lengths": np.array(lengths), "epsilons": np.array(epsilons),
        "eval_episodes": np.array(eval_eps), "eval_returns": np.array(eval_returns),
    }
    for array in arrays.values():
        array.setflags(write=False)  # el resultado es inmutable en TODOS sus arrays
    return TabularRun(config=config, seed=seed, best_eval_episode=best_episode, best_eval_mean=best_mean, **arrays)
