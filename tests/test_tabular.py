"""Q-learning y SARSA tabulares (paso 8): reglas de actualización, manejo de fin de episodio,
reproducibilidad y política greedy resultante."""

import numpy as np
import pytest

from cartpole_lab.rl.discretization import BOXES_1983
from cartpole_lab.rl.tabular import (
    FORCE_LEVELS,
    GreedyTabularPolicy,
    TabularConfig,
    epsilon_at,
    td_target,
    train_tabular,
)


# --- Configuración y calendario de exploración ---------------------------------------------


def test_force_levels_are_the_agreed_five_including_zero():
    assert FORCE_LEVELS == (-10.0, -5.0, 0.0, 5.0, 10.0)


@pytest.mark.parametrize(
    "kwargs",
    [dict(method="dqn"), dict(alpha=0.0), dict(alpha=1.5), dict(gamma=1.0), dict(epsilon_end=0.5, epsilon_start=0.1),
     dict(n_episodes=0), dict(epsilon_decay_fraction=0.0)],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        TabularConfig(**({"method": "q_learning", "n_episodes": 3000} | kwargs))


def test_epsilon_decays_linearly_then_stays_at_its_floor():
    config = TabularConfig(method="q_learning", n_episodes=100, epsilon_start=1.0, epsilon_end=0.1,
                           epsilon_decay_fraction=0.5)
    assert epsilon_at(0, config) == 1.0
    assert epsilon_at(25, config) == pytest.approx(0.55)
    assert epsilon_at(50, config) == pytest.approx(0.1)
    assert epsilon_at(99, config) == pytest.approx(0.1)


# --- Objetivo TD: la diferencia entre Q-learning y SARSA, y el fin de episodio ---------------


def test_q_learning_bootstraps_on_the_best_next_action():
    q_next = np.array([1.0, 5.0, 2.0, 0.0, 3.0])
    assert td_target("q_learning", 1.0, q_next, next_action=0, gamma=0.9, terminated=False) == pytest.approx(1 + 0.9 * 5)


def test_sarsa_bootstraps_on_the_action_actually_taken_next():
    q_next = np.array([1.0, 5.0, 2.0, 0.0, 3.0])
    assert td_target("sarsa", 1.0, q_next, next_action=0, gamma=0.9, terminated=False) == pytest.approx(1 + 0.9 * 1)


@pytest.mark.parametrize("method", ["q_learning", "sarsa"])
def test_failure_does_not_bootstrap(method):
    """Tras caer el poste no hay futuro: el objetivo es solo la recompensa."""
    assert td_target(method, 1.0, np.full(5, 99.0), next_action=2, gamma=0.99, terminated=True) == 1.0


# --- Entrenamiento --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def short_runs():
    config = TabularConfig(method="q_learning", n_episodes=60, eval_every=20, n_eval_episodes=2)
    return config, train_tabular(config, BOXES_1983, seed=0), train_tabular(config, BOXES_1983, seed=0)


def test_training_is_reproducible_given_the_seed(short_runs):
    _, a, b = short_runs
    np.testing.assert_array_equal(a.q_table, b.q_table)
    np.testing.assert_array_equal(a.episode_returns, b.episode_returns)


def test_different_seeds_give_different_runs():
    config = TabularConfig(method="sarsa", n_episodes=30, eval_every=30, n_eval_episodes=1)
    a = train_tabular(config, BOXES_1983, seed=0)
    b = train_tabular(config, BOXES_1983, seed=1)
    assert not np.array_equal(a.episode_returns, b.episode_returns)


def test_training_log_shapes_and_contents(short_runs):
    config, run, _ = short_runs
    assert run.q_table.shape == (BOXES_1983.n_states, len(FORCE_LEVELS))
    assert run.episode_returns.shape == run.episode_lengths.shape == run.epsilons.shape == (config.n_episodes,)
    np.testing.assert_array_equal(run.episode_returns, run.episode_lengths)  # +1 por paso (CartPole-v1)
    assert run.eval_episodes.tolist() == [20, 40, 60]
    assert run.eval_returns.shape == (3, config.n_eval_episodes)
    assert np.all(run.q_table >= 0)  # recompensas positivas y Q inicial 0


@pytest.mark.parametrize(
    "field", ["q_table", "best_q_table", "episode_returns", "episode_lengths", "epsilons", "eval_episodes", "eval_returns"]
)
def test_every_result_array_is_immutable(short_runs, field):
    _, run, _ = short_runs
    array = getattr(run, field)
    with pytest.raises(ValueError):
        array.flat[0] = 1.0


def test_eval_every_larger_than_the_training_budget_is_rejected():
    with pytest.raises(ValueError):
        TabularConfig(method="q_learning", n_episodes=50, eval_every=100)


def test_policy_does_not_change_when_the_source_table_changes():
    q = np.zeros((BOXES_1983.n_states, 5))
    policy = GreedyTabularPolicy(q, BOXES_1983)
    q[BOXES_1983.index(np.zeros(4))] = [9, 0, 0, 0, 0]  # se modifica la tabla original
    assert policy(np.zeros(4)) == 0.0


class _RandomForce:
    action_mode = "continuous"

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def reset(self):
        pass

    def __call__(self, state):
        return float(self.rng.choice(FORCE_LEVELS))


@pytest.mark.parametrize("method", ["q_learning", "sarsa"])
def test_learning_actually_happens_on_a_modest_budget(method):
    """Sanity check del aprendizaje (NO de convergencia): con 1500 episodios, la mejor política
    greedy vista supera al menos 3 veces a la política aleatoria en las mismas CI de validación.
    La convergencia se estudia con curvas y 10 semillas en scripts/train_tabular.py."""
    from cartpole_lab.env import CartPoleTask
    from cartpole_lab.rollout import run_episode

    config = TabularConfig(method=method, n_episodes=1500, eval_every=100, n_eval_episodes=10)
    run = train_tabular(config, BOXES_1983, seed=3)
    task = CartPoleTask(action_mode="continuous")
    try:
        seeds = range(config.eval_seed_start, config.eval_seed_start + config.n_eval_episodes)
        random_mean = np.mean([run_episode(task, _RandomForce(), seed=s).total_reward for s in seeds])
    finally:
        task.close()
    assert run.best_eval_mean > 3 * random_mean


def test_best_snapshot_is_the_best_validation_checkpoint(short_runs):
    _, run, _ = short_runs
    means = run.eval_returns.mean(axis=1)
    assert run.best_eval_mean == means.max()
    assert run.best_eval_episode == run.eval_episodes[np.argmax(means)]  # el primero ante empates
    with pytest.raises(ValueError):
        run.best_q_table[0, 0] = 1.0


def test_learned_policy_pushes_towards_the_side_the_pole_leans():
    """Comprobación de sentido físico de lo APRENDIDO (no programado), en las cajas que la política
    visita a menudo: carro centrado, poste inclinado 1°–6° con velocidad moderada -> empuja hacia
    ese lado. (En cajas raras, p. ej. caída muy rápida, Q queda casi sin aprender porque una buena
    política las evita: limitación documentada en docs/08_tabular.md.)"""
    config = TabularConfig(method="q_learning", n_episodes=1500, eval_every=100, n_eval_episodes=10)
    run = train_tabular(config, BOXES_1983, seed=3)
    policy = GreedyTabularPolicy(run.best_q_table, BOXES_1983)
    leaning_right = np.array([0.0, 0.0, np.radians(3.0), np.radians(10.0)])  # θ ∈ (1°, 6°), |θ̇| < 50°/s
    assert policy(leaning_right) > 0 and policy(-leaning_right) < 0


# --- Política greedy ------------------------------------------------------------------------


def test_greedy_policy_picks_the_best_force_and_breaks_ties_towards_gentle_forces():
    q = np.zeros((BOXES_1983.n_states, 5))
    s = np.zeros(4)
    idx = BOXES_1983.index(s)
    policy = GreedyTabularPolicy(q, BOXES_1983)
    assert policy(s) == 0.0  # empate total -> "ante la duda, no empujes"
    q[idx] = [0, 3, 3, 1, 0]
    assert GreedyTabularPolicy(q, BOXES_1983)(s) == 0.0  # empate −5 / 0: gana la más suave
    q[idx] = [0, 3, 1, 3, 0]
    assert GreedyTabularPolicy(q, BOXES_1983)(s) == -5.0  # empate ±5: orden fijo (izquierda primero)
    q[idx] = [9, 3, 3, 1, 0]
    assert GreedyTabularPolicy(q, BOXES_1983)(s) == -10.0


def test_greedy_policy_rejects_mismatched_table():
    with pytest.raises(ValueError):
        GreedyTabularPolicy(np.zeros((10, 5)), BOXES_1983)
