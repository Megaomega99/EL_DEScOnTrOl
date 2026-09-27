"""Tests del wrapper común `CartPoleTask`.

Propiedad central: el wrapper añade fuerza continua, perturbaciones y métricas,
pero NO cambia la física, la recompensa ni la terminación de CartPole-v1.
"""

import gymnasium as gym
import numpy as np
import pytest

from cartpole_lab.cost import QuadraticCost
from cartpole_lab.disturbances import ForcePulse
from cartpole_lab.dynamics import euler_step
from cartpole_lab.env import CartPoleTask
from cartpole_lab.params import ENV_ID
from tests.conftest import random_interior_states, set_raw_state


@pytest.fixture
def task():
    t = CartPoleTask(action_mode="continuous")
    yield t
    t.close()


@pytest.fixture
def discrete_task():
    t = CartPoleTask(action_mode="discrete")
    yield t
    t.close()


# --- Equivalencia física con Gymnasium -------------------------------------------------


def test_continuous_full_force_is_bitwise_identical_to_discrete_gymnasium(task, raw_env):
    task.reset(seed=7)
    raw_env.reset(seed=7)
    for action in [1, 1, 0, 1, 0, 0, 1]:
        task.step(+10.0 if action == 1 else -10.0)
        raw_env.step(action)
        np.testing.assert_array_equal(task.unwrapped.state, raw_env.unwrapped.state)


def test_arbitrary_continuous_force_follows_the_dynamics_model(task, params, rng):
    for state in random_interior_states(rng, params, n=100):
        force = rng.uniform(-params.force_mag, params.force_mag)
        task.reset(seed=0, options={"initial_state": state})
        task.step(force)
        np.testing.assert_array_equal(task.unwrapped.state, euler_step(state, force, params))


def test_force_mag_is_restored_after_each_step(task, params):
    task.reset(seed=0)
    task.step(3.3)
    assert task.unwrapped.force_mag == params.force_mag


def test_discrete_mode_reproduces_gymnasium_observations_and_rewards(discrete_task, raw_env):
    obs_w, _ = discrete_task.reset(seed=11)
    obs_g, _ = raw_env.reset(seed=11)
    np.testing.assert_array_equal(obs_w, obs_g.astype(np.float64))
    for action in [0, 1, 1, 0, 1]:
        obs_w, r_w, term_w, trunc_w, _ = discrete_task.step(action)
        obs_g, r_g, term_g, trunc_g, _ = raw_env.step(action)
        np.testing.assert_array_equal(obs_w, obs_g.astype(np.float64))
        assert (r_w, term_w, trunc_w) == (r_g, term_g, trunc_g)


def test_observation_is_float64_and_inside_observation_space(task):
    obs, _ = task.reset(seed=0)
    assert obs.dtype == np.float64 and obs.shape == (4,)
    assert task.observation_space.contains(obs)


# --- Validación de acciones (fallar de forma ruidosa, nunca silenciosa) -----------------


def test_saturation_clips_and_reports(task, params):
    task.reset(seed=0, options={"initial_state": np.zeros(4)})
    _, _, _, _, info = task.step(25.0)
    assert info["force"] == params.force_mag
    assert info["force_requested"] == 25.0
    assert info["saturated"] is True
    np.testing.assert_array_equal(
        task.unwrapped.state, euler_step(np.zeros(4), params.force_mag, params)
    )


def test_unsaturated_force_is_reported_as_such(task):
    task.reset(seed=0)
    _, _, _, _, info = task.step(np.array([2.5]))  # también acepta array de tamaño 1
    assert info["force"] == 2.5 and info["saturated"] is False


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, "10", [1.0, 2.0], True, np.array([True])])
def test_continuous_mode_rejects_invalid_actions(task, bad):
    task.reset(seed=0)
    with pytest.raises((TypeError, ValueError)):
        task.step(bad)


@pytest.mark.parametrize("bad", [2, -1, 0.5, 1.0, True])
def test_discrete_mode_rejects_invalid_actions(discrete_task, bad):
    discrete_task.reset(seed=0)
    with pytest.raises((TypeError, ValueError)):
        discrete_task.step(bad)


@pytest.mark.parametrize("action", [np.int64(1), np.array(1)])
def test_discrete_mode_accepts_numpy_integers(discrete_task, action):
    """Las redes de la Fase 2 devuelven enteros de NumPy (p. ej. argmax), no int de Python."""
    discrete_task.reset(seed=0)
    _, _, _, _, info = discrete_task.step(action)
    assert info["force"] == 10.0


def test_unknown_action_mode_raises():
    with pytest.raises(ValueError):
        CartPoleTask(action_mode="torque")


# --- Condición inicial ---------------------------------------------------------------


def test_initial_state_is_set_exactly(task):
    s0 = np.array([0.1, -0.2, 0.05, 0.3])
    obs, info = task.reset(seed=0, options={"initial_state": s0})
    np.testing.assert_array_equal(task.unwrapped.state, s0)
    np.testing.assert_allclose(obs, s0, rtol=1e-6)  # obs pasa por float32 (API oficial)
    assert info["initial_state_source"] == "user"


def test_reset_without_initial_state_uses_gymnasium_distribution(task, raw_env):
    obs_w, info = task.reset(seed=3)
    obs_g, _ = raw_env.reset(seed=3)
    np.testing.assert_array_equal(obs_w, obs_g.astype(np.float64))
    assert info["initial_state_source"] == "gymnasium_uniform"


@pytest.mark.parametrize(
    "bad_state",
    [
        [0.0, 0.0, 0.25, 0.0],  # |θ| > 12°: el episodio nacería terminado
        [3.0, 0.0, 0.0, 0.0],  # |x| > 2.4 m
        [0.0, np.nan, 0.0, 0.0],
        [0.0, 0.0, 0.0],  # forma incorrecta
    ],
)
def test_invalid_initial_state_raises(task, bad_state):
    with pytest.raises(ValueError):
        task.reset(seed=0, options={"initial_state": bad_state})


def test_initial_state_combined_with_reset_bounds_is_ambiguous_and_raises(task):
    with pytest.raises(ValueError):
        task.reset(seed=0, options={"initial_state": np.zeros(4), "low": -0.1})


def test_gymnasium_reset_bounds_still_pass_through(task):
    obs, _ = task.reset(seed=0, options={"low": -0.001, "high": 0.001})
    assert np.all(np.abs(obs) <= 0.001)


# --- Terminación y truncamiento: mismos criterios que CartPole-v1 ----------------------


def test_pole_fall_is_reported(task):
    task.reset(seed=0, options={"initial_state": [0.0, 0.0, 0.1, 0.0]})
    terminated = False
    while not terminated:
        _, _, terminated, truncated, info = task.step(0.0)
        assert not truncated
    assert info["termination_reason"] == "pole_angle_limit"


def test_cart_out_of_bounds_is_reported(task):
    # θ = θ̇ = 0 y F = 0: el carro se desliza a velocidad constante con el poste vertical.
    task.reset(seed=0, options={"initial_state": [2.3, 2.0, 0.0, 0.0]})
    terminated = False
    while not terminated:
        _, _, terminated, _, info = task.step(0.0)
    assert info["termination_reason"] == "cart_position_limit"


def test_time_limit_truncation_is_reported():
    task = CartPoleTask(action_mode="continuous", max_episode_steps=20)
    try:
        task.reset(seed=0, options={"initial_state": np.zeros(4)})
        for k in range(20):
            _, reward, terminated, truncated, info = task.step(0.0)
            assert reward == 1.0 and not terminated
        assert truncated and info["termination_reason"] == "time_limit"
        assert info["step"] == 20 and info["time"] == pytest.approx(0.4)
    finally:
        task.close()


def test_running_steps_report_no_termination(task):
    task.reset(seed=0)
    _, _, _, _, info = task.step(0.0)
    assert info["termination_reason"] is None


def test_step_after_episode_end_raises(task):
    task.reset(seed=0, options={"initial_state": [0.0, 0.0, 0.2, 2.0]})
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = task.step(0.0)
    with pytest.raises(RuntimeError):
        task.step(0.0)


def test_step_before_reset_raises(task):
    with pytest.raises(RuntimeError):
        task.step(0.0)


# --- Perturbaciones externas ---------------------------------------------------------


def test_disturbance_is_added_after_actuator_saturation(params):
    pulse = ForcePulse(start_step=0, duration_steps=1, force=30.0)
    task = CartPoleTask(action_mode="continuous", disturbance=pulse)
    try:
        task.reset(seed=0, options={"initial_state": np.zeros(4)})
        _, _, _, _, info = task.step(99.0)  # se satura a +10 N; la perturbación NO
        assert info["force"] == params.force_mag
        assert info["disturbance_force"] == 30.0
        assert info["total_force"] == params.force_mag + 30.0
        np.testing.assert_array_equal(
            task.unwrapped.state, euler_step(np.zeros(4), 40.0, params)
        )
        _, _, _, _, info = task.step(0.0)
        assert info["disturbance_force"] == 0.0
    finally:
        task.close()


def test_disturbance_also_works_in_discrete_mode(params):
    pulse = ForcePulse(start_step=0, duration_steps=1, force=-4.0)
    task = CartPoleTask(action_mode="discrete", disturbance=pulse)
    try:
        task.reset(seed=0, options={"initial_state": np.zeros(4)})
        _, _, _, _, info = task.step(1)
        assert info["total_force"] == 6.0
        np.testing.assert_array_equal(task.unwrapped.state, euler_step(np.zeros(4), 6.0, params))
    finally:
        task.close()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_non_finite_disturbance_raises_before_touching_the_simulator(bad_value):
    """Sin esta comprobación, un NaN corrompe la trayectoria y el episodio nunca termina."""
    task = CartPoleTask(action_mode="continuous", disturbance=lambda step: bad_value)
    try:
        task.reset(seed=0, options={"initial_state": np.zeros(4)})
        with pytest.raises(ValueError):
            task.step(0.0)
        np.testing.assert_array_equal(task.unwrapped.state, np.zeros(4))  # estado intacto
    finally:
        task.close()


def test_error_inside_gymnasium_step_invalidates_the_episode(task, monkeypatch):
    task.reset(seed=0)

    def broken_step(action):
        raise RuntimeError("fallo simulado dentro de Gymnasium")

    monkeypatch.setattr(task.env, "step", broken_step)
    with pytest.raises(RuntimeError, match="simulado"):
        task.step(0.0)
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="reset"):  # no se puede seguir sin reset()
        task.step(0.0)


def test_force_pulse_from_time(params):
    pulse = ForcePulse.at_time(t_start=5.0, duration=0.04, force=20.0, dt=params.tau)
    assert (pulse.start_step, pulse.duration_steps) == (250, 2)
    assert [pulse(k) for k in (249, 250, 251, 252)] == [0.0, 20.0, 20.0, 0.0]
    assert pulse.impulse(params.tau) == pytest.approx(0.8)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(start_step=-1, duration_steps=1, force=1.0),
        dict(start_step=0, duration_steps=0, force=1.0),
        dict(start_step=0, duration_steps=1, force=np.nan),
    ],
)
def test_force_pulse_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        ForcePulse(**kwargs)


def test_force_pulse_rejects_times_off_the_simulation_grid(params):
    with pytest.raises(ValueError):
        ForcePulse.at_time(t_start=5.01, duration=0.02, force=1.0, dt=params.tau)


# --- Recompensa y coste comunes ------------------------------------------------------


def test_reward_is_the_unmodified_gymnasium_reward(task):
    task.reset(seed=0, options={"initial_state": [0.0, 0.0, 0.2, 2.0]})
    rewards, terminated = [], False
    while not terminated:
        _, r, terminated, _, _ = task.step(0.0)
        rewards.append(r)
    assert rewards == [1.0] * len(rewards)  # +1 por paso, incluido el terminal (v1)


def test_stage_cost_is_evaluated_at_pre_step_state_and_commanded_force(task, params):
    s0 = np.array([0.5, 1.0, -0.1, 0.2])
    task.reset(seed=0, options={"initial_state": s0})
    _, _, _, _, info = task.step(4.0)
    assert info["cost"] == pytest.approx(task.cost(s0, 4.0))


def test_normalized_cost_is_one_per_term_at_the_failure_boundary(params):
    cost = QuadraticCost.normalized(params, effort_weight=0.1)
    edge = np.array([params.x_threshold, 123.0, params.theta_threshold_radians, -45.0])
    assert cost(edge, params.force_mag) == pytest.approx(1.0 + 1.0 + 0.1)
    assert cost(np.zeros(4), 0.0) == 0.0


@pytest.mark.parametrize(
    "weights, r", [((1, 1, 1), 0.1), ((1, -1, 1, 1), 0.1), ((1, 1, 1, 1), -0.1)]
)
def test_quadratic_cost_validates_weights(weights, r):
    with pytest.raises(ValueError):
        QuadraticCost(state_weights=weights, force_weight=r)


def test_dt_and_params_are_exposed(task, params):
    assert task.dt == params.tau
    assert task.params == params
    assert task.action_space.low[0] == -params.force_mag
    assert task.action_space.high[0] == params.force_mag


def test_rgb_render_passes_through():
    task = CartPoleTask(action_mode="continuous", render_mode="rgb_array")
    try:
        task.reset(seed=0)
        frame = task.render()
        assert frame.shape == (400, 600, 3)
    finally:
        task.close()


def test_wrapped_env_is_the_official_cartpole_v1(task):
    assert task.unwrapped.spec.id == ENV_ID
    assert isinstance(task.unwrapped, gym.envs.classic_control.CartPoleEnv)
