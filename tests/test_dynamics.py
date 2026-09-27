"""El modelo dinámico explícito (para LQR/MPC) debe reproducir el simulador."""

import dataclasses

import numpy as np
import pytest

from cartpole_lab.dynamics import continuous_dynamics, euler_step
from cartpole_lab.params import CartPoleParams
from tests.conftest import random_interior_states, set_raw_state


def test_params_are_read_from_the_gymnasium_instance(raw_env, params):
    base = raw_env.unwrapped
    assert CartPoleParams.from_env(raw_env) == params
    # Las magnitudes derivadas deben coincidir BIT A BIT con las que usa step().
    assert params.total_mass == base.total_mass
    assert params.polemass_length == base.polemass_length


def test_from_env_rejects_non_cartpole_env():
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    try:
        with pytest.raises(TypeError):
            CartPoleParams.from_env(env)
    finally:
        env.close()


@pytest.mark.parametrize("action, force_sign", [(0, -1.0), (1, +1.0)])
def test_euler_step_reproduces_gymnasium_step(raw_env, params, rng, action, force_sign):
    raw_env.reset(seed=0)
    for state in random_interior_states(rng, params, n=200):
        set_raw_state(raw_env, state)
        raw_env.step(action)
        expected = raw_env.unwrapped.state
        predicted = euler_step(state, force_sign * params.force_mag, params)
        # Mismas operaciones en el mismo orden -> igualdad exacta en float64.
        np.testing.assert_array_equal(predicted, expected)


def test_euler_step_reproduces_gymnasium_for_intermediate_forces(raw_env, params, rng):
    """Verdad de referencia SIN pasar por CartPoleTask: fuerza arbitraria aplicada a mano
    sobre el CartPole crudo (mismo mecanismo que usa el wrapper, pero sin el wrapper)."""
    raw_env.reset(seed=0)
    base = raw_env.unwrapped
    for state in random_interior_states(rng, params, n=100):
        force = rng.uniform(-params.force_mag, params.force_mag)
        set_raw_state(raw_env, state)
        base.force_mag = abs(force)
        raw_env.step(1 if force >= 0 else 0)
        np.testing.assert_array_equal(euler_step(state, force, params), base.state)


def test_upright_rest_is_an_equilibrium(params):
    np.testing.assert_array_equal(continuous_dynamics(np.zeros(4), 0.0, params), np.zeros(4))


def test_dynamics_is_odd_symmetric(params, rng):
    """Espejo izquierda-derecha: f(-s, -F) = -f(s, F). Chequeo físico de signos."""
    for state in random_interior_states(rng, params, n=50):
        force = rng.uniform(-params.force_mag, params.force_mag)
        np.testing.assert_allclose(
            continuous_dynamics(-state, -force, params),
            -continuous_dynamics(state, force, params),
            rtol=0,
            atol=1e-12,
        )


def test_batched_evaluation_matches_single(params, rng):
    states = random_interior_states(rng, params, n=10)
    forces = rng.uniform(-params.force_mag, params.force_mag, size=10)
    batched = euler_step(states, forces, params)
    single = np.stack([euler_step(s, f, params) for s, f in zip(states, forces)])
    np.testing.assert_array_equal(batched, single)


def test_single_state_broadcasts_against_a_batch_of_forces(params, rng):
    """Caso de broadcasting de rango mixto: estado (4,) y fuerzas (N,) -> resultado (N, 4)."""
    state = random_interior_states(rng, params, n=1)[0]
    forces = rng.uniform(-params.force_mag, params.force_mag, size=7)
    batched = euler_step(state, forces, params)
    assert batched.shape == (7, 4)
    single = np.stack([euler_step(state, f, params) for f in forces])
    np.testing.assert_array_equal(batched, single)


def test_rejects_state_with_wrong_shape(params):
    with pytest.raises(ValueError):
        continuous_dynamics(np.zeros(3), 0.0, params)


def test_euler_step_refuses_other_integrators(params):
    semi_implicit = dataclasses.replace(params, kinematics_integrator="semi-implicit euler")
    with pytest.raises(ValueError):
        euler_step(np.zeros(4), 0.0, semi_implicit)
