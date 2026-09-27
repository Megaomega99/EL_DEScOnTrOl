"""Verificación de los SUPUESTOS sobre Gymnasium, antes de construir nada encima.

Si algún test de este archivo falla tras actualizar Gymnasium, NO se debe
"arreglar" el test: significa que el simulador cambió y hay que revisar el
modelo (dynamics.py), el wrapper y los controladores basados en modelo.
"""

import math
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from cartpole_lab.params import ENV_ID
from tests.conftest import set_raw_state

REPO_ROOT = Path(__file__).resolve().parents[1]
POLE_RGB = (202, 152, 101)  # color del poste en CartPoleEnv.render()
CART_RGB = (0, 0, 0)  # color del carro (y de la línea del raíl)


def test_installed_gymnasium_matches_pinned_version():
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    pinned = re.search(r"^gymnasium==(\S+)", text, flags=re.MULTILINE)
    assert pinned is not None, "requirements.txt debe fijar gymnasium con '=='"
    assert gym.__version__ == pinned.group(1)


def test_internal_constants_match_documented_values(params):
    # Valores documentados en docs/01_entorno_cartpole.md (leídos del código fuente
    # de gymnasium/envs/classic_control/cartpole.py, v1.3.0).
    assert params.gravity == 9.8
    assert params.masscart == 1.0
    assert params.masspole == 0.1
    assert params.length == 0.5  # MITAD de la longitud del poste
    assert params.force_mag == 10.0
    assert params.tau == 0.02
    assert params.kinematics_integrator == "euler"
    assert params.x_threshold == 2.4
    assert params.theta_threshold_radians == 12 * 2 * math.pi / 360


def test_cartpole_v1_time_limit_is_500_steps():
    assert gym.spec(ENV_ID).max_episode_steps == 500


def test_positive_force_accelerates_cart_right_and_rotates_pole_negative(raw_env):
    raw_env.reset(seed=0)
    set_raw_state(raw_env, [0.0, 0.0, 0.0, 0.0])
    raw_env.step(1)  # acción 1 = empujar a la derecha (+force_mag)
    _, x_dot, _, theta_dot = raw_env.unwrapped.state
    assert x_dot > 0.0, "F > 0 debe acelerar el carro hacia +x"
    assert theta_dot < 0.0, "F > 0 debe hacer girar el poste hacia θ < 0"


def test_gravity_amplifies_positive_angle(raw_env):
    """Sin fuerza, un θ > 0 crece: el equilibrio vertical es inestable."""
    raw_env.reset(seed=0)
    set_raw_state(raw_env, [0.0, 0.0, 0.05, 0.0])
    # Fuerza nula: única forma con la API discreta es anular force_mag (solo en este test).
    raw_env.unwrapped.force_mag = 0.0
    raw_env.step(1)
    assert raw_env.unwrapped.state[3] > 0.0


def _pixel_centroid_x(frame, rgb):
    mask = np.all(frame == rgb, axis=-1)
    # La línea del raíl es negra y ocupa todo el ancho: se descartan esas filas.
    mask[mask.sum(axis=1) == frame.shape[1]] = False
    _, cols = np.nonzero(mask)
    return cols.mean()


@pytest.mark.parametrize("theta, side", [(+0.2, "right"), (-0.2, "left")])
def test_render_positive_angle_means_pole_leans_right(theta, side):
    env = gym.make(ENV_ID, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        set_raw_state(env, [0.0, 0.0, theta, 0.0])
        frame = env.render()
        pole_x = _pixel_centroid_x(frame, POLE_RGB)
        cart_x = _pixel_centroid_x(frame, CART_RGB)
    finally:
        env.close()
    assert (pole_x > cart_x) if side == "right" else (pole_x < cart_x)


def test_render_positive_position_means_cart_right_of_center():
    env = gym.make(ENV_ID, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        set_raw_state(env, [1.0, 0.0, 0.0, 0.0])
        frame = env.render()
    finally:
        env.close()
    assert _pixel_centroid_x(frame, CART_RGB) > frame.shape[1] / 2


def test_observation_is_float32_but_internal_state_is_float64(raw_env):
    obs, _ = raw_env.reset(seed=0)
    assert obs.dtype == np.float32
    assert raw_env.unwrapped.state.dtype == np.float64


def test_reset_options_cannot_set_an_exact_initial_state(raw_env):
    """Justifica por qué el wrapper sobrescribe `unwrapped.state` al fijar la CI."""
    with pytest.raises((TypeError, ValueError)):
        raw_env.reset(seed=0, options={"low": np.zeros(4), "high": np.zeros(4)})
