"""Tests del MPC lineal (paso 5): equivalencia con el LQR, restricciones, penalización exacta
y sanity checks. Los episodios con MPC son lentos (una optimización cada 20 ms), por eso aquí
se usan pocas semillas; el sanity check completo (N = 50) lo hace scripts/design_mpc.py."""

import json

import cvxpy as cp
import numpy as np
import pytest

from cartpole_lab.controllers.lqr import LQRController, design_default_lqr
from cartpole_lab.controllers.mpc import LinearMPC, MPCConfig
from cartpole_lab.env import CartPoleTask
from cartpole_lab.paths import TUNING_DIR
from cartpole_lab.rollout import run_episode
from cartpole_lab.sanity import REFERENCE_HARD_INITIAL_STATE, UNRECOVERABLE_INITIAL_STATES, is_stabilized


@pytest.fixture(scope="module")
def lqr(params):
    return design_default_lqr(params)


@pytest.fixture(scope="module")
def mpc(params):
    return LinearMPC(params)


@pytest.fixture
def task():
    t = CartPoleTask(action_mode="continuous")
    yield t
    t.close()


# --- Configuración ---------------------------------------------------------------------------


def test_default_config_uses_the_lqr_model_validity_angle_as_constraint(params):
    config = MPCConfig()
    assert config.theta_constraint(params) == pytest.approx(np.arccos(0.99))
    assert config.x_constraint(params) == params.x_threshold
    assert config.horizon == 25 and config.solver == "CLARABEL"


@pytest.mark.parametrize(
    "kwargs",
    [dict(horizon=0), dict(slack_weight=0.0), dict(theta_constraint_rad=0.3), dict(theta_constraint_rad=-0.1)],
)
def test_invalid_config_raises(params, kwargs):
    with pytest.raises(ValueError):
        LinearMPC(params, MPCConfig(**kwargs))


def test_rejects_state_with_wrong_shape(mpc):
    with pytest.raises(ValueError):
        mpc(np.zeros(3))


# --- Relación con el LQR ---------------------------------------------------------------------


@pytest.mark.parametrize("state", [[0.01, 0.0, 0.01, 0.0], [-0.05, 0.02, 0.005, -0.03]])
def test_equals_lqr_when_no_constraint_is_active(mpc, lqr, state):
    """Con coste terminal P (coste óptimo del LQR) y restricciones inactivas, la solución del
    horizonte finito es EXACTAMENTE la del LQR (principio de optimalidad de Bellman)."""
    expected = (-lqr.K @ np.asarray(state)).item()
    assert mpc(state) == pytest.approx(expected, rel=1e-5)


def test_same_weights_as_lqr(mpc, lqr):
    """El MPC hereda Q, R y P del LQR: la única diferencia son horizonte y restricciones."""
    np.testing.assert_array_equal(mpc.design.Q, lqr.Q)
    np.testing.assert_array_equal(mpc.design.R, lqr.R)
    np.testing.assert_array_equal(mpc.design.P, lqr.P)


# --- Restricciones ---------------------------------------------------------------------------


def test_force_constraint_is_respected_exactly(mpc, params):
    force = mpc([0.0, 0.0, 0.2, 3.0])
    assert abs(force) <= params.force_mag
    assert abs(force) == pytest.approx(params.force_mag)


def test_soft_constraints_match_hard_constraints_when_feasible(params):
    """Penalización EXACTA: si el problema con restricciones duras es factible, la versión
    blanda da la misma solución, con holguras nulas."""
    state = np.array([1.0, 1.0, 0.0, 0.0])  # activa la restricción de θ en el plan
    soft = LinearMPC(params)
    hard = LinearMPC(params, MPCConfig(slack_weight=None))
    assert soft(state) == pytest.approx(hard(state), rel=1e-5, abs=1e-6)
    assert soft.last_plan["max_slack"] < 1e-7


def test_slack_weight_exceeds_the_constraint_multipliers(params):
    """Condición suficiente para la exactitud de la penalización L1: el peso supera a los
    multiplicadores de Lagrange de las restricciones de estado del problema duro."""
    hard = LinearMPC(params, MPCConfig(slack_weight=None))
    hard([1.0, 1.0, 0.0, 0.0])
    assert hard.last_plan["max_state_constraint_dual"] < MPCConfig().slack_weight / 10


def test_hard_constraints_can_be_infeasible_but_soft_ones_always_solve(params):
    beyond = [0.0, 0.0, 0.205, 0.0]  # θ0 = 11.7° ya viola la restricción de 8.1° en el paso 1
    hard = LinearMPC(params, MPCConfig(slack_weight=None))
    with pytest.raises(RuntimeError, match="infeasible"):
        hard(beyond)
    soft = LinearMPC(params)
    soft(beyond)
    assert soft.last_plan["max_slack"] > 0


def test_solver_failure_raises_instead_of_returning_garbage(params, monkeypatch):
    controller = LinearMPC(params)

    def failing_solve(*args, **kwargs):
        controller._problem._status = cp.SOLVER_ERROR
        return None

    monkeypatch.setattr(controller._problem, "solve", failing_solve)
    with pytest.raises(RuntimeError):
        controller([0.0, 0.0, 0.01, 0.0])


def test_plan_and_timing_are_recorded_and_reset(params):
    controller = LinearMPC(params)
    controller([0.0, 0.0, 0.01, 0.0])
    assert controller.last_plan["states"].shape == (MPCConfig().horizon + 1, 4)
    assert len(controller.solve_times) == 1
    controller.reset()
    assert controller.solve_times == [] and controller.last_plan is None


# --- Sanity checks en el CartPole-v1 real ----------------------------------------------------


def test_stabilizes_from_near_equilibrium(task, params, mpc):
    for seed in range(1000, 1005):
        assert is_stabilized(run_episode(task, mpc, seed=seed), params), f"semilla {seed}"


def test_recovers_the_reference_hard_initial_state(task, params, mpc):
    assert is_stabilized(run_episode(task, mpc, initial_state=REFERENCE_HARD_INITIAL_STATE), params)


def test_recovers_a_state_where_lqr_fails(task, params, mpc, lqr):
    """El resultado central del paso: el LQR, para frenar el carro, inclina el poste más allá
    de 12°; el MPC, con la restricción θ ≤ 8.1°, planifica sin salir de la zona segura."""
    state = [1.0, 1.0, 0.0, 0.0]
    lqr_traj = run_episode(task, LQRController(lqr.K, params), initial_state=state)
    assert lqr_traj.termination_reason == "pole_angle_limit"
    mpc_traj = run_episode(task, mpc, initial_state=state)
    assert is_stabilized(mpc_traj, params)


def test_planning_exactly_on_the_failure_limit_fails(task, params):
    """Sin margen (restricción EXACTAMENTE en 12°), el plan roza el límite y el pequeño error
    del modelo lineal (~0.005°) lo cruza."""
    no_margin = LinearMPC(params, MPCConfig(theta_constraint_rad=params.theta_threshold_radians))
    traj = run_episode(task, no_margin, initial_state=[1.0, 1.0, 0.0, 0.0])
    assert traj.termination_reason == "pole_angle_limit"


def test_in_this_ideal_simulator_a_tiny_margin_already_suffices(task, params):
    """Documenta la sensibilidad: 0.01° de margen basta aquí. Los 8.1° elegidos son
    conservadores a propósito (robustez fuera de esta simulación perfecta)."""
    tiny_margin = LinearMPC(params, MPCConfig(theta_constraint_rad=params.theta_threshold_radians - np.radians(0.01)))
    assert is_stabilized(run_episode(task, tiny_margin, initial_state=[1.0, 1.0, 0.0, 0.0]), params)


@pytest.mark.parametrize("name", sorted(UNRECOVERABLE_INITIAL_STATES))
def test_extreme_initial_states_fail_loudly(task, params, mpc, name):
    initial_state, reasons = UNRECOVERABLE_INITIAL_STATES[name]
    traj = run_episode(task, mpc, initial_state=initial_state)
    assert traj.terminated and traj.termination_reason in reasons
    assert traj.n_steps > 1 and np.all(np.isfinite(traj.states))


def test_saved_record_matches_the_current_config():
    record = json.loads((TUNING_DIR / "mpc.json").read_text(encoding="utf-8"))
    assert record["config"] == MPCConfig().to_dict()
