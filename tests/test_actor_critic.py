"""Actor-crítico A2C con ventaja GAE y política gaussiana comprimida con tanh (paso 10)."""

import numpy as np
import pytest
import torch

from cartpole_lab.rl.actor_critic import (
    ACConfig,
    ACPolicy,
    GaussianActor,
    ValueCritic,
    export_actor,
    gae,
    next_state_values,
    squashed_log_prob,
    train_a2c,
)
from cartpole_lab.rl.dqn import input_scales, load_network

# --- Ventaja GAE ---------------------------------------------------------------------------


def test_gae_matches_a_hand_computation():
    """Un episodio de 3 pasos que termina (fallo) en el último."""
    rewards = np.array([[1.0], [1.0], [1.0]])
    values = np.array([[0.5], [0.4], [0.3]])
    next_values = np.array([[0.4], [0.3], [0.0]])  # el último: fallo -> 0
    continues = np.array([[1.0], [1.0], [0.0]])
    gamma, lam = 0.9, 0.8
    d = rewards[:, 0] + gamma * next_values[:, 0] - values[:, 0]
    a2 = d[2]
    a1 = d[1] + gamma * lam * a2
    a0 = d[0] + gamma * lam * a1
    advantages, returns = gae(rewards, values, next_values, continues, gamma, lam)
    np.testing.assert_allclose(advantages[:, 0], [a0, a1, a2])
    np.testing.assert_allclose(returns, advantages + values)


def test_gae_does_not_leak_across_episode_boundaries():
    """Lo que ocurre después del fin de un episodio no puede cambiar la ventaja de ese episodio."""
    rewards = np.ones((4, 1))
    values = np.zeros((4, 1))
    continues = np.array([[1.0], [0.0], [1.0], [1.0]])  # el episodio acaba en t = 1
    next_values = np.zeros((4, 1))
    a, _ = gae(rewards, values, next_values, continues, 0.9, 0.9)
    b, _ = gae(rewards * np.array([[1], [1], [99], [99]]), values, next_values, continues, 0.9, 0.9)
    np.testing.assert_allclose(a[:2], b[:2])


def test_gae_with_lambda_one_is_the_discounted_return_minus_value():
    rewards = np.array([[1.0], [2.0], [3.0]])
    values = np.array([[0.7], [0.1], [0.2]])
    next_values = np.array([[0.1], [0.2], [0.0]])
    continues = np.array([[1.0], [1.0], [0.0]])
    a, _ = gae(rewards, values, next_values, continues, gamma=0.5, lam=1.0)
    discounted = [1 + 0.5 * 2 + 0.25 * 3, 2 + 0.5 * 3, 3]
    np.testing.assert_allclose(a[:, 0], np.array(discounted) - values[:, 0])


def test_next_state_values_distinguish_failure_from_time_limit():
    """Fallo -> 0 (no hay futuro). Truncamiento -> V(último estado real), NO el del reset."""
    v_next_obs = np.array([5.0, 5.0, 5.0])
    v_final_obs = np.array([7.0, 7.0, 7.0])
    terminated = np.array([True, False, False])
    truncated = np.array([False, True, False])
    np.testing.assert_array_equal(next_state_values(v_next_obs, v_final_obs, terminated, truncated), [0.0, 7.0, 5.0])


# --- Política gaussiana comprimida ---------------------------------------------------------


def test_squashed_log_prob_is_the_change_of_variables_density(rng):
    """log p(F) = log N(u; μ, σ) − log(F_max·(1 − tanh² u)), con F = F_max·tanh(u)."""
    u = torch.as_tensor(rng.normal(size=50))
    mu, log_std, f_max = torch.tensor(0.3), torch.tensor(-0.2), 10.0
    sigma = log_std.exp()
    gaussian = -0.5 * ((u - mu) / sigma) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
    expected = gaussian - torch.log(f_max * (1 - torch.tanh(u) ** 2))
    np.testing.assert_allclose(squashed_log_prob(u, mu, log_std, f_max).numpy(), expected.numpy(), rtol=1e-6)


def test_squashed_log_prob_is_finite_for_large_pre_squash_values():
    """Forma numéricamente estable: para |u| grande, 1 − tanh² u se redondea a 0 en float32."""
    u = torch.tensor([-30.0, 30.0])
    assert torch.all(torch.isfinite(squashed_log_prob(u, torch.tensor(0.0), torch.tensor(0.0), 10.0)))


def test_actor_and_critic_shapes():
    actor, critic = GaussianActor((64, 64)), ValueCritic((64, 64))
    mu = actor(torch.zeros(7, 4))
    assert mu.shape == (7,) and actor.log_std.shape == ()
    assert critic(torch.zeros(7, 4)).shape == (7,)


def test_deterministic_policy_force_is_strictly_inside_the_actuator_limit(params):
    """tanh garantiza |F| < F_max: el actuador nunca recorta, y no hay sesgo por recorte."""
    layers = [{"W": np.zeros((1, 4)).tolist(), "b": [2.0], "activation": "linear"}]
    force = ACPolicy(layers, input_scales(params), params.force_mag)(np.zeros(4))
    assert 0 < force < params.force_mag and force == pytest.approx(params.force_mag * np.tanh(2.0))


def test_exported_actor_reproduces_the_torch_mean_action(tmp_path, params, rng):
    torch.manual_seed(0)
    actor = GaussianActor((64, 64))
    path = tmp_path / "actor.json"
    export_actor(actor, path, input_scales=input_scales(params), force_limit=params.force_mag, metadata={})
    loaded = load_network(path)
    policy = ACPolicy(loaded["layers"], tuple(loaded["input_scales"]), loaded["force_limit"])
    for state in rng.uniform(-1, 1, size=(10, 4)):
        with torch.no_grad():
            mu = actor(torch.as_tensor(state / np.asarray(input_scales(params)), dtype=torch.float32))
        assert policy(state) == pytest.approx(float(params.force_mag * torch.tanh(mu)), rel=1e-5, abs=1e-5)


# --- Configuración y entrenamiento -----------------------------------------------------------


def test_export_from_layers_requires_and_stores_log_std(tmp_path, params):
    """Camino que usa el script (capas ya extraídas, no el objeto de PyTorch)."""
    layers = [{"W": np.zeros((1, 4)).tolist(), "b": [0.0], "activation": "linear"}]
    with pytest.raises(ValueError, match="log_std"):
        export_actor(layers, tmp_path / "a.json", input_scales=input_scales(params), force_limit=10.0, metadata={})
    export_actor(layers, tmp_path / "b.json", input_scales=input_scales(params), force_limit=10.0, metadata={}, log_std=-0.07)
    assert load_network(tmp_path / "b.json")["log_std"] == -0.07


@pytest.mark.parametrize(
    "kwargs",
    [dict(gamma=1.0), dict(gae_lambda=1.5), dict(n_envs=0), dict(n_envs=101), dict(n_steps=0), dict(actor_lr=0.0),
     dict(eval_every_steps=10**9), dict(hidden=())],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        ACConfig(**kwargs)


@pytest.fixture(scope="module")
def tiny():
    return ACConfig(total_steps=2048, n_envs=4, n_steps=32, eval_every_steps=1024, n_eval_episodes=2)


def test_training_is_reproducible_given_the_seed(tiny):
    a, b = train_a2c(tiny, seed=0), train_a2c(tiny, seed=0)
    np.testing.assert_array_equal(a.eval_returns, b.eval_returns)
    np.testing.assert_array_equal(a.policy_std, b.policy_std)
    for la, lb in zip(a.final_layers, b.final_layers):
        np.testing.assert_array_equal(la["W"], lb["W"])


def test_training_log_contents(tiny):
    run = train_a2c(tiny, seed=1)
    assert run.eval_steps.tolist() == [1024, 2048]
    assert run.eval_returns.shape == (2, tiny.n_eval_episodes)
    n_updates = tiny.total_steps // (tiny.n_envs * tiny.n_steps)
    assert run.policy_std.shape == run.explained_variance.shape == (n_updates,)
    assert np.all(run.policy_std > 0)
    assert run.best_eval_mean == run.eval_returns.mean(axis=1).max()
    assert np.isfinite(run.best_log_std) and np.isfinite(run.final_log_std)
