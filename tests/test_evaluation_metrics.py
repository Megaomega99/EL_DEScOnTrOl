"""Definiciones de las métricas de la Fase 3 (docs/12_protocolo_evaluacion.md §3), con trayectorias sintéticas."""

import numpy as np
import pytest

from cartpole_lab.evaluation.methods import PolicySpec, all_specs
from cartpole_lab.evaluation.metrics import (
    FAILED,
    STABILIZED,
    SURVIVED,
    aggregate_episodes,
    across_seeds,
    episode_metrics,
    grid_outcome,
    settling_time,
)
from cartpole_lab.evaluation.protocol import (
    IMPULSES_NS,
    MAIN_METHODS,
    cart_grid_states,
    impulse_sign,
    pole_grid_states,
)
from cartpole_lab.rollout import Trajectory

DT = 0.02
BAND = np.radians(0.5)


def synthetic(theta_deg, x=None, reason="time_limit"):
    theta = np.radians(np.asarray(theta_deg, dtype=float))
    n = theta.size
    states = np.zeros((n, 4))
    states[:, 2] = theta
    if x is not None:
        states[:, 0] = x
    forces = np.ones(n - 1)
    return Trajectory(
        times=np.arange(n) * DT, states=states, forces=forces, disturbance_forces=np.zeros(n - 1),
        rewards=np.ones(n - 1), costs=np.ones(n - 1), saturated=np.zeros(n - 1, bool),
        terminated=reason != "time_limit", truncated=reason == "time_limit", termination_reason=reason,
        dt=DT, force_limit=10.0,
    )


def test_settling_time_is_the_start_of_the_final_run_inside_the_band():
    # fuera, fuera, dentro, FUERA, dentro, dentro -> se asienta en el índice 4 (t = 0.08 s)
    traj = synthetic([3, 2, 0.1, 1.0, 0.2, 0.1])
    assert settling_time(traj, x_band=None) == pytest.approx(4 * DT)


def test_already_inside_from_the_start_settles_at_zero():
    assert settling_time(synthetic([0.1, 0.2, 0.0]), x_band=None) == 0.0


def test_ending_outside_the_band_is_not_settled():
    assert settling_time(synthetic([3, 0.1, 0.1, 2.0]), x_band=None) is None


def test_failed_episode_is_not_settled_even_if_inside():
    assert settling_time(synthetic([0.1, 0.1], reason="pole_angle_limit"), x_band=None) is None


def test_full_settling_requires_the_cart_to_be_centred_too():
    traj = synthetic([0.1, 0.1, 0.1], x=[0.0, 0.3, 0.3])  # poste perfecto, carro a 30 cm
    assert settling_time(traj, x_band=None) == 0.0
    assert settling_time(traj) is None


def test_recovery_time_is_measured_from_the_impulse():
    traj = synthetic([0.1, 0.1, 5.0, 2.0, 0.2, 0.1])  # impulso en el paso 2
    assert settling_time(traj, x_band=None, from_step=2) == pytest.approx(2 * DT)


def test_episode_metrics_with_impulse():
    # El pulso actúa en la transición 2 -> 3: el estado 2 es PREVIO al impulso y no cuenta para el pico
    # (aquí se le da un valor grande a propósito), pero la recuperación se cuenta desde t_2.
    traj = synthetic([0.1, 0.1, 6.0, 5.0, 2.0, 0.2, 0.1], x=[0, 0, 0.3, 0.1, 0.2, 0.01, 0.0])
    m = episode_metrics(traj, impulse_step=2)
    assert m["survived"] and m["recovery_time_s"] == pytest.approx(3 * DT)
    assert m["peak_abs_theta_after_deg"] == pytest.approx(5.0)
    assert m["peak_abs_x_after_m"] == pytest.approx(0.2)
    assert m["failed_before_impulse"] is False


def test_grid_outcome_codes():
    assert grid_outcome(synthetic([0.1, 0.1])) == STABILIZED
    assert grid_outcome(synthetic([0.1, 0.1], x=[0.5, 0.5])) == SURVIVED
    assert grid_outcome(synthetic([0.1, 20.0], reason="pole_angle_limit")) == FAILED


def test_aggregation_and_across_seed_statistics():
    episodes = [episode_metrics(synthetic([3, 0.1, 0.1])), episode_metrics(synthetic([3, 2, 0.1])),
                episode_metrics(synthetic([3, 3], reason="pole_angle_limit"))]
    summary = aggregate_episodes(episodes)
    assert summary["n_episodes"] == 3 and summary["survival_rate"] == pytest.approx(2 / 3)
    assert summary["pole_settling_time_s"]["n"] == 2  # el fallido no cuenta
    assert summary["pole_settling_time_s"]["mean"] == pytest.approx(1.5 * DT)
    seeds = across_seeds([summary, {**summary, "survival_rate": 1.0}], ["survival_rate", "effort_abs_Ns"])
    assert seeds["n_seeds"] == 2 and seeds["survival_rate"]["mean"] == pytest.approx((2 / 3 + 1) / 2)


def test_protocol_constants_match_the_declared_document():
    assert IMPULSES_NS == (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)  # tras la enmienda de §8
    assert [impulse_sign(i) for i in range(4)] == [1.0, -1.0, 1.0, -1.0]
    assert len(pole_grid_states()) == len(cart_grid_states()) == 21 * 21
    assert max(abs(s[2]) for s in pole_grid_states()) <= np.radians(12)  # CI válidas
    assert max(abs(s[0]) for s in cart_grid_states()) < 2.4


def test_policy_specs_cover_every_method_with_ten_seeds_for_rl():
    specs = all_specs()
    assert {s.method for s in specs} >= set(MAIN_METHODS)
    for method in ("Q-learning", "SARSA", "DQN", "Actor-crítico", "Double DQN"):
        assert sorted(s.seed for s in specs if s.method == method) == list(range(10))
    assert [s.seed for s in specs if s.method == "LQR"] == [None]
    assert PolicySpec("DQN", 3).is_rl and not PolicySpec("MPC", None).is_rl
