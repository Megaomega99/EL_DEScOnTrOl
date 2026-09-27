"""Fixtures compartidas por todos los tests.

Los tests comparan SIEMPRE contra el CartPole-v1 original de Gymnasium
(`raw_env`), no contra valores copiados a mano: el objetivo es demostrar que
nuestra capa (wrapper + modelo dinámico) no altera la física del simulador.
"""

import gymnasium as gym
import numpy as np
import pytest

from cartpole_lab.params import ENV_ID, load_params


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture
def raw_env():
    """CartPole-v1 tal cual lo entrega `gym.make`, sin nuestro wrapper."""
    env = gym.make(ENV_ID)
    yield env
    env.close()


@pytest.fixture
def rng():
    # Semilla fija: los tests con estados aleatorios deben ser reproducibles.
    return np.random.default_rng(12345)


def set_raw_state(env, state):
    """Sobrescribe el estado interno de un CartPole crudo (solo para tests)."""
    env.unwrapped.state = np.array(state, dtype=np.float64)
    env.unwrapped.steps_beyond_terminated = None


def random_interior_states(rng, params, n):
    """Estados aleatorios dentro de la región no terminal, con velocidades moderadas."""
    high = np.array([params.x_threshold, 2.0, params.theta_threshold_radians, 2.0])
    return rng.uniform(-0.9 * high, 0.9 * high, size=(n, 4))
