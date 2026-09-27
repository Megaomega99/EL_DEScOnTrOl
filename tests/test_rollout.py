"""Tests del bucle de simulación común `run_episode` y de la trayectoria registrada."""

import numpy as np
import pytest

from cartpole_lab.env import CartPoleTask
from cartpole_lab.rollout import run_episode


class ConstantForce:
    """Controlador trivial de prueba: aplica siempre la misma fuerza."""

    action_mode = "continuous"

    def __init__(self, force):
        self.force = force
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, state):
        return self.force


class RandomDiscrete:
    """Política aleatoria con su propio generador sembrado (reproducible)."""

    action_mode = "discrete"

    def __init__(self, seed):
        self.seed = seed

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, state):
        return int(self.rng.integers(2))


@pytest.fixture
def task():
    t = CartPoleTask(action_mode="continuous")
    yield t
    t.close()


def test_exact_equilibrium_without_force_survives_the_full_episode(task):
    traj = run_episode(task, ConstantForce(0.0), initial_state=np.zeros(4))
    assert traj.n_steps == 500 and traj.truncated and not traj.terminated
    assert traj.termination_reason == "time_limit"
    np.testing.assert_array_equal(traj.states, 0.0)


def test_tiny_perturbation_without_control_fails_loudly(task):
    """Sanity check negativo: sin control, el equilibrio inestable se pierde."""
    traj = run_episode(task, ConstantForce(0.0), initial_state=[0.0, 0.0, 0.01, 0.0])
    assert traj.terminated and traj.termination_reason == "pole_angle_limit"
    assert traj.n_steps < 500


def test_trajectory_shapes_and_derived_metrics(task):
    ctrl = ConstantForce(2.0)
    traj = run_episode(task, ctrl, initial_state=np.zeros(4))
    T = traj.n_steps
    assert traj.states.shape == (T + 1, 4) and traj.times.shape == (T + 1,)
    for arr in (traj.forces, traj.disturbance_forces, traj.rewards, traj.costs, traj.saturated):
        assert arr.shape == (T,)
    np.testing.assert_allclose(traj.times, np.arange(T + 1) * task.dt)
    assert traj.duration == pytest.approx(T * task.dt)
    assert traj.total_reward == float(T)
    assert traj.control_effort_abs == pytest.approx(2.0 * T * task.dt)
    assert traj.control_effort_sq == pytest.approx(4.0 * T * task.dt)
    assert traj.total_cost == pytest.approx(traj.costs.sum())
    assert ctrl.resets == 1


def test_first_recorded_state_is_the_initial_state(task):
    s0 = np.array([0.1, 0.0, -0.02, 0.0])
    traj = run_episode(task, ConstantForce(0.0), initial_state=s0)
    np.testing.assert_allclose(traj.states[0], s0, rtol=1e-6)


def test_action_mode_mismatch_raises(task):
    with pytest.raises(ValueError):
        run_episode(task, RandomDiscrete(seed=0))


def test_same_seeds_give_identical_trajectories():
    task = CartPoleTask(action_mode="discrete")
    try:
        a = run_episode(task, RandomDiscrete(seed=5), seed=42)
        b = run_episode(task, RandomDiscrete(seed=5), seed=42)
        c = run_episode(task, RandomDiscrete(seed=5), seed=43)
    finally:
        task.close()
    np.testing.assert_array_equal(a.states, b.states)
    assert a.states.shape != c.states.shape or not np.array_equal(a.states, c.states)


def test_fraction_at_force_limit_counts_controllers_that_clip_internally(task):
    """Un controlador que ya entrega ±F_max nunca activa `saturated` (no se le recorta nada),
    pero SÍ está en el límite del actuador: la métrica física debe contarlo."""
    traj = run_episode(task, ConstantForce(10.0), initial_state=np.zeros(4))
    assert not traj.saturated.any()
    assert traj.fraction_at_force_limit == 1.0


def test_fraction_at_force_limit_is_zero_for_gentle_control(task):
    traj = run_episode(task, ConstantForce(0.0), initial_state=np.zeros(4))
    assert traj.fraction_at_force_limit == 0.0 and traj.force_limit == task.params.force_mag
