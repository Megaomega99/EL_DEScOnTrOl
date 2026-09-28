"""DQN (paso 9): memoria de repetición, objetivo TD, red, exportación a JSON y reproducibilidad."""

import json

import numpy as np
import pytest
import torch

from cartpole_lab.rl.dqn import (
    DQNConfig,
    DQNPolicy,
    QNetwork,
    dqn_targets,
    export_network,
    input_scales,
    load_network,
    network_forward_numpy,
    train_dqn,
)
from cartpole_lab.rl.replay import ReplayBuffer
from cartpole_lab.rl.tabular import FORCE_LEVELS

# --- Memoria de repetición ---------------------------------------------------------------------


def test_replay_buffer_stores_and_samples_transitions():
    buffer = ReplayBuffer(capacity=10)
    for k in range(4):
        buffer.add(np.full(4, k), k % 5, 1.0, np.full(4, k + 1), terminated=(k == 3))
    assert len(buffer) == 4
    batch = buffer.sample(8, np.random.default_rng(0))
    assert batch.states.shape == (8, 4) and batch.actions.shape == (8,) and batch.terminated.dtype == bool
    np.testing.assert_array_equal(batch.next_states[:, 0], batch.states[:, 0] + 1)  # cada muestra es coherente


def test_replay_buffer_overwrites_the_oldest_when_full():
    buffer = ReplayBuffer(capacity=3)
    for k in range(5):
        buffer.add(np.full(4, k), 0, 1.0, np.zeros(4), terminated=False)
    assert len(buffer) == 3
    stored = set(buffer.sample(100, np.random.default_rng(1)).states[:, 0])
    assert stored == {2.0, 3.0, 4.0}  # se olvidan las transiciones 0 y 1


def test_replay_sampling_is_reproducible_and_never_reads_empty_slots():
    buffer = ReplayBuffer(capacity=100)
    for k in range(5):
        buffer.add(np.full(4, k + 1), 0, 1.0, np.zeros(4), terminated=False)
    a = buffer.sample(50, np.random.default_rng(7)).states
    b = buffer.sample(50, np.random.default_rng(7)).states
    np.testing.assert_array_equal(a, b)
    assert np.all(a[:, 0] >= 1)  # ninguna fila vacía (cero) muestreada


@pytest.mark.parametrize("capacity", [0, -5])
def test_replay_buffer_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=capacity)


def test_sampling_an_empty_buffer_raises():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=5).sample(1, np.random.default_rng(0))


# --- Objetivo TD y red -------------------------------------------------------------------------


def test_dqn_target_bootstraps_with_the_target_network_except_on_failure():
    rewards = torch.tensor([1.0, 1.0])
    next_q = torch.tensor([[0.0, 3.0, 1.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0, 0.0]])
    terminated = torch.tensor([False, True])
    np.testing.assert_allclose(dqn_targets(rewards, next_q, terminated, gamma=0.9).numpy(), [1 + 0.9 * 3, 1.0])


def test_input_scales_come_from_the_failure_limits(params):
    scales = input_scales(params)
    assert scales[0] == params.x_threshold and scales[2] == params.theta_threshold_radians
    assert scales[1] == 1.0 and scales[3] == 1.5


def test_network_maps_4_inputs_to_one_value_per_force_level():
    net = QNetwork(hidden=(64, 64), n_actions=len(FORCE_LEVELS))
    assert net(torch.zeros(3, 4)).shape == (3, 5)
    assert sum(p.numel() for p in net.parameters()) == 4 * 64 + 64 + 64 * 64 + 64 + 64 * 5 + 5


@pytest.mark.parametrize(
    "kwargs",
    [dict(gamma=1.0), dict(learning_rate=0.0), dict(batch_size=0), dict(buffer_capacity=10, batch_size=64),
     dict(learning_starts=200, total_steps=100), dict(epsilon_end=0.9, epsilon_start=0.5), dict(hidden=()),
     dict(grad_clip_norm=0.0)],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        DQNConfig(**kwargs)


# --- Exportación: lo que se evalúa es EXACTAMENTE lo que se exporta ------------------------------


def test_exported_json_reproduces_the_torch_network(tmp_path, params, rng):
    torch.manual_seed(0)
    net = QNetwork(hidden=(64, 64), n_actions=5)
    path = tmp_path / "red.json"
    export_network(net, path, input_scales=input_scales(params), force_levels=FORCE_LEVELS, metadata={"prueba": True})
    loaded = load_network(path)
    states = rng.uniform(-1, 1, size=(20, 4))
    normalized = states / np.asarray(loaded["input_scales"])
    with torch.no_grad():
        expected = net(torch.as_tensor(normalized, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(network_forward_numpy(loaded["layers"], normalized), expected, rtol=1e-5, atol=1e-5)
    assert json.loads(path.read_text(encoding="utf-8"))["metadata"] == {"prueba": True}


def test_policy_picks_the_force_with_the_highest_q(params):
    layers = [{"W": np.zeros((5, 4)).tolist(), "b": [0.0, 0.0, 0.0, 3.0, 0.0], "activation": "linear"}]
    policy = DQNPolicy(layers, input_scales(params), FORCE_LEVELS)
    assert policy(np.zeros(4)) == 5.0


def test_policy_rejects_state_with_wrong_shape(params):
    layers = [{"W": np.zeros((5, 4)).tolist(), "b": [0.0] * 5, "activation": "linear"}]
    with pytest.raises(ValueError):
        DQNPolicy(layers, input_scales(params), FORCE_LEVELS)(np.zeros(3))


# --- Entrenamiento --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_config():
    return DQNConfig(total_steps=1500, learning_starts=200, eval_every_steps=500, n_eval_episodes=2,
                     buffer_capacity=1000, target_update_every=100)


def test_training_is_reproducible_given_the_seed(tiny_config):
    a, b = train_dqn(tiny_config, seed=0), train_dqn(tiny_config, seed=0)
    np.testing.assert_array_equal(a.episode_returns, b.episode_returns)
    for la, lb in zip(a.final_layers, b.final_layers):
        np.testing.assert_array_equal(la["W"], lb["W"])


def test_training_log_contents(tiny_config):
    run = train_dqn(tiny_config, seed=1)
    assert run.eval_steps.tolist() == [500, 1000, 1500]
    assert run.eval_returns.shape == (3, tiny_config.n_eval_episodes)
    assert run.best_eval_mean == run.eval_returns.mean(axis=1).max()
    assert run.episode_returns.sum() <= tiny_config.total_steps  # +1 por paso; el último episodio puede quedar a medias
    assert len(run.final_layers) == 3 and run.final_layers[-1]["activation"] == "linear"


def test_double_dqn_target_evaluates_the_online_choice_with_the_target_network():
    """Double DQN: la red ONLINE elige a' y la red OBJETIVO la evalúa (quita el sesgo del max)."""
    rewards = torch.tensor([1.0])
    next_q_target = torch.tensor([[0.0, 3.0, 1.0, 0.0, 9.0]])  # el max ingenuo tomaría 9
    next_q_online = torch.tensor([[0.0, 5.0, 1.0, 0.0, 2.0]])  # la online prefiere la acción 1
    target = dqn_targets(rewards, next_q_target, torch.tensor([False]), gamma=0.9, next_q_online=next_q_online)
    np.testing.assert_allclose(target.numpy(), [1 + 0.9 * 3.0])


def test_double_and_vanilla_differ_only_in_the_target(tiny_config):
    import dataclasses

    double_run = train_dqn(dataclasses.replace(tiny_config, double=True), seed=0)
    assert double_run.config.double is True
    assert not np.array_equal(double_run.episode_returns, train_dqn(tiny_config, seed=0).episode_returns)


def test_value_diagnostics_are_recorded_at_each_evaluation(tiny_config):
    run = train_dqn(tiny_config, seed=2)
    for series in (run.q_at_rest, run.action_gap_median, run.q_max_median):
        assert series.shape == run.eval_steps.shape and np.all(np.isfinite(series))
    assert np.all(run.action_gap_median >= 0)


def test_action_gap_is_the_difference_between_the_two_best_actions():
    from cartpole_lab.rl.dqn import PROBE_STATES, value_diagnostics

    net = QNetwork(hidden=(8,), n_actions=5)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.layers[-1].bias.copy_(torch.tensor([1.0, 4.0, 2.0, 3.5, 0.0]))  # Q constante
    diag = value_diagnostics(net, torch.ones(4))
    assert diag["gap_median"] == pytest.approx(0.5) and diag["q_max_median"] == pytest.approx(4.0)
    assert diag["q_at_rest"] == pytest.approx(4.0) and PROBE_STATES.shape == (256, 4)


def test_theoretical_upper_bound_of_q():
    from cartpole_lab.rl.dqn import q_upper_bound

    assert q_upper_bound(0.99) == pytest.approx(100.0)  # con bootstrap al truncar: horizonte infinito
