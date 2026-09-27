"""El criterio común de "estabilizado" y el protocolo de sanity check, iguales para todos los métodos."""

import numpy as np
import pytest

from cartpole_lab.controllers.pid_tuning import TuningConfig
from cartpole_lab.dynamics import continuous_dynamics
from cartpole_lab.env import CartPoleTask
from cartpole_lab.rollout import run_episode
from cartpole_lab.sanity import (
    REFERENCE_HARD_INITIAL_STATE,
    UNRECOVERABLE_INITIAL_STATES,
    best_rescue_violation,
    is_stabilized,
    momentum_bound_on_cart_excursion,
    sanity_statistics,
)


class ZeroForce:
    action_mode = "continuous"

    def reset(self):
        pass

    def __call__(self, state):
        return 0.0


def _episode(initial_state):
    task = CartPoleTask(action_mode="continuous")
    try:
        return run_episode(task, ZeroForce(), initial_state=initial_state)
    finally:
        task.close()


def test_exact_equilibrium_counts_as_stabilized(params):
    assert is_stabilized(_episode(np.zeros(4)), params)


def test_failed_episode_is_not_stabilized(params):
    assert not is_stabilized(_episode([0.0, 0.0, 0.05, 0.0]), params)


def test_surviving_but_displaced_episode_is_not_stabilized(params):
    # Sobrevive (poste vertical exacto) pero el carro queda lejos del centro.
    assert not is_stabilized(_episode([1.0, 0.0, 0.0, 0.0]), params)


def test_sanity_statistics_report_failures_honestly(params):
    stats = sanity_statistics(ZeroForce(), params, seeds=range(3))
    assert stats["n_episodes"] == 3
    assert stats["survival_rate"] == 0.0 and stats["stabilized_rate"] == 0.0
    assert set(stats["termination_reasons"]) == {"pole_angle_limit"}


class ConstantForce:
    action_mode = "continuous"

    def __init__(self, force):
        self.force = force

    def reset(self):
        pass

    def __call__(self, state):
        return self.force


@pytest.mark.parametrize(
    "name, rescue_force",
    [("poste_cayendo_rapido", +10.0), ("carro_lanzado_al_borde", -10.0)],
)
def test_unrecoverable_states_fail_even_with_full_force_in_the_rescue_direction(name, rescue_force):
    """Justifica el sanity check negativo: si ni la fuerza máxima en la dirección correcta
    lo evita, que un controlador falle ahí no es un defecto del controlador."""
    initial_state, reasons = UNRECOVERABLE_INITIAL_STATES[name]
    task = CartPoleTask(action_mode="continuous")
    try:
        traj = run_episode(task, ConstantForce(rescue_force), initial_state=initial_state)
    finally:
        task.close()
    assert traj.terminated and traj.termination_reason in reasons


def test_reference_hard_state_is_the_corner_of_the_pid_tuning_box():
    assert REFERENCE_HARD_INITIAL_STATE == TuningConfig().initial_state_box


def test_sanity_statistics_measure_time_at_the_actuator_limit(params):
    stats = sanity_statistics(ConstantForce(10.0), params, seeds=range(2))
    assert stats["fraction_of_steps_at_force_limit"]["mean"] == 1.0


def test_center_of_mass_obeys_newton_with_only_the_actuator_force(params, rng):
    """Base de la cota de momento: M·ẍ_c = F con x_c = x + (m l / M) sin θ, para cualquier estado."""
    ratio = params.polemass_length / params.total_mass
    for _ in range(50):
        state = rng.uniform([-2, -3, -0.3, -3], [2, 3, 0.3, 3])
        force = rng.uniform(-params.force_mag, params.force_mag)
        _, x_acc, _, theta_acc = continuous_dynamics(state, force, params)
        theta, theta_dot = state[2], state[3]
        com_acc = x_acc + ratio * (theta_acc * np.cos(theta) - theta_dot**2 * np.sin(theta))
        assert params.total_mass * com_acc == pytest.approx(force, abs=1e-12)


def test_momentum_bound_proves_the_cart_state_unrecoverable_with_wide_margin(params):
    initial_state, _ = UNRECOVERABLE_INITIAL_STATES["carro_lanzado_al_borde"]
    assert momentum_bound_on_cart_excursion(initial_state, params) > params.x_threshold + 0.2


def test_momentum_bound_is_not_triggered_by_a_recoverable_state(params):
    assert momentum_bound_on_cart_excursion(REFERENCE_HARD_INITIAL_STATE, params) < params.x_threshold


@pytest.mark.parametrize("name", sorted(UNRECOVERABLE_INITIAL_STATES))
def test_global_search_finds_no_force_sequence_that_avoids_failure(params, name):
    """Evidencia numérica: la mejor secuencia hallada viola un límite con margen amplio (> 20 %)."""
    initial_state, _ = UNRECOVERABLE_INITIAL_STATES[name]
    assert best_rescue_violation(initial_state, params) > 1.2


def test_global_search_does_find_a_rescue_when_one_exists(params):
    """Control positivo de la herramienta: sin él, un '> 1' podría ser un fallo del optimizador."""
    assert best_rescue_violation(REFERENCE_HARD_INITIAL_STATE, params) < 1.0
