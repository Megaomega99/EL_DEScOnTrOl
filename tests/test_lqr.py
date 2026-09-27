"""Tests del LQR (paso 4): diseño, optimalidad, comparación de modelos y sanity checks."""

import dataclasses
import json

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from cartpole_lab.controllers.lqr import (
    LQRController,
    bryson_weights,
    cost_matrix_of_gain,
    design_default_lqr,
    lqr_criterion,
    solve_clqr,
    solve_dlqr,
)
from cartpole_lab.env import CartPoleTask
from cartpole_lab.linearization import continuous_linearization, euler_discretization, zoh_discretization
from cartpole_lab.paths import TUNING_DIR
from cartpole_lab.rollout import run_episode
from cartpole_lab.sanity import (
    REFERENCE_HARD_INITIAL_STATE,
    UNRECOVERABLE_INITIAL_STATES,
    is_stabilized,
)


@pytest.fixture(scope="module")
def plant(params):
    return euler_discretization(*continuous_linearization(params), params.tau)


@pytest.fixture(scope="module")
def design(params):
    return design_default_lqr(params)


# --- Pesos Q, R (regla de Bryson) ---------------------------------------------------------------


def test_bryson_weights_use_design_tolerances_not_failure_limits(params):
    Q, R = bryson_weights(params)
    max_x, max_theta = 0.5 * params.x_threshold, np.arccos(0.99)
    np.testing.assert_allclose(np.diag(Q), [1 / max_x**2, 0.0, 1 / max_theta**2, 0.0])
    np.testing.assert_allclose(R, [[1 / params.force_mag**2]])
    assert np.count_nonzero(Q - np.diag(np.diag(Q))) == 0
    # Tolerancias de diseño MÁS ESTRICTAS que los límites de fallo del coste de evaluación.
    assert max_x < params.x_threshold and max_theta < params.theta_threshold_radians


@pytest.mark.parametrize(
    "Q, R",
    [
        (np.diag([1.0, 0.0, -1.0, 0.0]), np.eye(1)),  # Q no semidefinida
        (np.eye(4) + np.triu(np.ones((4, 4)), 1), np.eye(1)),  # Q no simétrica
        (np.eye(4), np.zeros((1, 1))),  # R no definida positiva
        (np.eye(3), np.eye(1)),  # forma incorrecta
    ],
)
def test_solver_rejects_invalid_weights(plant, Q, R):
    with pytest.raises(ValueError):
        solve_dlqr(*plant, Q, R)


# --- Solución de Riccati y propiedades del lazo cerrado ----------------------------------------


def test_riccati_solution_is_consistent_and_positive_definite(plant, design):
    A, B = plant
    B = B[:, None]
    P, Q, R, K = design.P, design.Q, design.R, design.K
    residual = A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) + Q - P
    np.testing.assert_allclose(residual, 0.0, atol=1e-8 * np.abs(P).max())
    np.testing.assert_allclose(P, P.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(P) > 0)
    np.testing.assert_allclose(K, np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A))
    np.testing.assert_allclose(P, solve_discrete_are(A, B, Q, R), rtol=1e-10)


def test_closed_loop_is_stable(design):
    assert design.spectral_radius < 1.0


def test_feedback_signs_match_the_physical_intuition(design):
    """F = −K s con −K > 0 en todo: poste a la derecha -> empujar a la derecha, y carro a la
    derecha -> también empujar a la derecha primero (fase no mínima), como en el PID."""
    assert np.all(-design.K > 0)


def test_lqr_gain_is_optimal_against_any_other_stabilizing_gain(params, plant, design):
    """Propiedad que define al LQR: P_K − P ⪰ 0 para cualquier K estabilizante, en la planta real."""
    A_c, B_c = continuous_linearization(params)
    others = {
        "zoh": solve_dlqr(*zoh_discretization(A_c, B_c, params.tau), design.Q, design.R).K,
        "continuo": solve_clqr(A_c, B_c, design.Q, design.R),
        "perturbado": design.K * 1.2,
    }
    for name, K in others.items():
        P_K = cost_matrix_of_gain(*plant, design.Q, design.R, K)
        assert np.linalg.eigvalsh(P_K - design.P).min() > -1e-8 * np.abs(design.P).max(), name


def test_cost_of_an_unstable_gain_is_rejected(plant, design):
    with pytest.raises(ValueError):
        cost_matrix_of_gain(*plant, design.Q, design.R, np.zeros((1, 4)))


def test_discrete_design_converges_to_continuous_lqr_as_tau_shrinks(params, design):
    A_c, B_c = continuous_linearization(params)
    K_c = solve_clqr(A_c, B_c, design.Q, design.R)

    def gap(tau):
        return np.linalg.norm(solve_dlqr(*euler_discretization(A_c, B_c, tau), design.Q, design.R).K - K_c)

    assert gap(params.tau / 4) < gap(params.tau)


# --- Controlador ------------------------------------------------------------------------------


def test_controller_applies_negative_feedback_and_saturates(params, design):
    ctrl = LQRController(design.K, params)
    state = np.array([0.01, 0.0, 0.01, 0.0])
    assert ctrl(state) == pytest.approx((-design.K @ state).item())
    assert ctrl([0.0, 0.0, 0.2, 5.0]) == params.force_mag


def test_controller_gain_is_immutable(params, design):
    ctrl = LQRController(design.K, params)
    with pytest.raises(ValueError):
        ctrl.K[0, 0] = 0.0


@pytest.mark.parametrize("bad_K", [np.ones(3), np.ones((2, 4)), np.array([[np.nan, 1, 1, 1]])])
def test_controller_rejects_invalid_gain(params, bad_K):
    with pytest.raises(ValueError):
        LQRController(bad_K, params)


# --- Sanity checks en el CartPole-v1 real ------------------------------------------------------


@pytest.fixture
def task():
    t = CartPoleTask(action_mode="continuous")
    yield t
    t.close()


def test_lqr_stabilizes_from_near_equilibrium(task, params, design):
    ctrl = LQRController(design.K, params)
    for seed in range(1000, 1020):
        traj = run_episode(task, ctrl, seed=seed)
        assert is_stabilized(traj, params), f"semilla {seed}"


def test_lqr_recovers_the_reference_hard_initial_state(task, params, design):
    traj = run_episode(task, LQRController(design.K, params), initial_state=REFERENCE_HARD_INITIAL_STATE)
    assert is_stabilized(traj, params)


@pytest.mark.parametrize("name", sorted(UNRECOVERABLE_INITIAL_STATES))
def test_extreme_initial_states_fail_loudly(task, params, design, name):
    initial_state, reasons = UNRECOVERABLE_INITIAL_STATES[name]
    traj = run_episode(task, LQRController(design.K, params), initial_state=initial_state)
    assert traj.terminated and traj.termination_reason in reasons
    assert traj.n_steps > 1 and np.all(np.isfinite(traj.states))


def test_saved_design_record_matches_the_current_design(design):
    """Si cambian Q, R o el modelo sin regenerar results/tuning/lqr.json, el registro es obsoleto."""
    record = json.loads((TUNING_DIR / "lqr.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(record["K"], design.K.tolist(), rtol=1e-12)
    np.testing.assert_allclose(record["Q"], design.Q.tolist(), rtol=1e-12)
    np.testing.assert_allclose(record["R"], design.R.tolist(), rtol=1e-12)


def test_design_is_invariant_to_scaling_Q_and_R_together(plant, design):
    """Solo importa el cociente Q/R: por eso la regla de Bryson fija escalas relativas."""
    scaled = solve_dlqr(*plant, 7.0 * design.Q, 7.0 * design.R)
    np.testing.assert_allclose(scaled.K, design.K, rtol=1e-9)


def test_increasing_R_reduces_the_gain(plant, design):
    cheap = solve_dlqr(*plant, design.Q, 0.1 * design.R)
    expensive = solve_dlqr(*plant, design.Q, 10.0 * design.R)
    assert np.linalg.norm(expensive.K) < np.linalg.norm(design.K) < np.linalg.norm(cheap.K)


def test_design_dataclass_is_frozen(design):
    with pytest.raises(dataclasses.FrozenInstanceError):
        design.K = None


def test_lqr_criterion_on_a_trajectory_matches_the_riccati_prediction_near_equilibrium(task, params, design):
    """Muy cerca del equilibrio (lineal, sin saturar) el coste simulado debe ser s0ᵀ P s0."""
    s0 = np.array([0.001, 0.0, 0.001, 0.0])
    traj = run_episode(task, LQRController(design.K, params), initial_state=s0)
    predicted = float(s0 @ design.P @ s0)
    assert lqr_criterion(traj, design.Q, design.R) == pytest.approx(predicted, rel=1e-3)


def test_lqr_criterion_rejects_failed_episodes(task, params, design):
    initial_state, _ = UNRECOVERABLE_INITIAL_STATES["poste_cayendo_rapido"]
    traj = run_episode(task, LQRController(design.K, params), initial_state=initial_state)
    with pytest.raises(ValueError):
        lqr_criterion(traj, design.Q, design.R)


def test_controller_rejects_state_with_wrong_shape(params, design):
    with pytest.raises(ValueError):
        LQRController(design.K, params)(np.zeros(3))


def test_weight_and_cost_matrices_are_immutable(params, plant, design):
    Q, R = bryson_weights(params)
    P_K = cost_matrix_of_gain(*plant, design.Q, design.R, design.K)
    for array in (Q, R, P_K):
        with pytest.raises(ValueError):
            array[0, 0] = 1.0
