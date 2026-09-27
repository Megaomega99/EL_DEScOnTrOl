"""Tests del controlador fuzzy Takagi-Sugeno de orden 0 (paso 6).

Bloques:
1. Funciones de membresía y tabla de reglas: propiedades que las hacen explicables.
2. Inferencia: qué calcula realmente el controlador (interpolación de la tabla; PD cerca de 0).
3. Sintonización y sanity checks con los factores de escala sintonizados.
"""

import json

import numpy as np
import pytest

from cartpole_lab.controllers.fuzzy import (
    LABELS,
    RULE_TABLE,
    CascadeFuzzy,
    FuzzyGains,
    fuzzy_force,
    load_tuned_fuzzy_gains,
    memberships,
)
from cartpole_lab.controllers.fuzzy_tuning import (
    FuzzyTuningConfig,
    fuzzy_closed_loop_spectral_radius,
    simulate_fuzzy_batch,
    tune_fuzzy,
)
from cartpole_lab.env import CartPoleTask
from cartpole_lab.paths import TUNING_DIR
from cartpole_lab.rollout import run_episode
from cartpole_lab.sanity import REFERENCE_HARD_INITIAL_STATE, UNRECOVERABLE_INITIAL_STATES, is_stabilized

# Escalas de prueba fijas (no sintonizadas): solo para verificar la mecánica.
GAINS = FuzzyGains(theta_scale=0.1, theta_dot_scale=0.5, force_scale=8.0, kp_x=0.03, kd_x=0.07)
LIMIT = 0.05


def law(params, state, gains=GAINS):
    return fuzzy_force(state, gains, force_limit=params.force_mag, theta_ref_limit=LIMIT)


# --- 1. Membresías y reglas -------------------------------------------------------------------


def test_five_linguistic_labels():
    assert LABELS == ("NG", "NP", "Z", "PP", "PG")


def test_memberships_form_a_partition_of_unity(rng):
    """En cada punto las pertenencias suman 1: nunca hay un hueco sin regla ni doble conteo."""
    z = rng.uniform(-3, 3, size=1000)
    np.testing.assert_allclose(memberships(z).sum(axis=-1), 1.0, atol=1e-12)


def test_each_label_is_fully_true_at_its_center_and_only_there():
    np.testing.assert_array_equal(memberships(np.linspace(-1, 1, 5)), np.eye(5))


def test_memberships_are_triangular_between_centers():
    mu = memberships(np.array(0.25))  # a mitad de camino entre Z (0) y PP (0.5)
    np.testing.assert_allclose(mu, [0, 0, 0.5, 0.5, 0])


def test_extreme_labels_saturate_beyond_the_universe():
    """Hombros: todo lo que supera 'grande' sigue siendo 'grande' (no deja de pertenecer)."""
    np.testing.assert_array_equal(memberships(np.array([-7.0, 7.0])), [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])


def test_rule_table_is_the_documented_macvicar_whelan_table():
    """Consecuente = sat(i + j): 'la fuerza sigue a hacia dónde cae más lo rápido que cae'."""
    levels = np.arange(-2, 3)
    expected = np.clip(levels[:, None] + levels[None, :], -2, 2) / 2
    np.testing.assert_array_equal(RULE_TABLE, expected)


def test_rule_table_encodes_the_physical_intuition():
    Z, PG, NG = 2, 4, 0
    assert RULE_TABLE[Z, Z] == 0.0  # vertical y quieto: no empujes
    assert RULE_TABLE[PG, PG] == 1.0  # cae a la derecha y deprisa: empuja fuerte a la derecha
    assert RULE_TABLE[PG, NG] == 0.0  # inclinado a la derecha pero ya vuelve rápido: no hagas nada
    np.testing.assert_array_equal(RULE_TABLE, -RULE_TABLE[::-1, ::-1])  # simetría izquierda-derecha
    assert np.all(np.diff(RULE_TABLE, axis=0) >= 0) and np.all(np.diff(RULE_TABLE, axis=1) >= 0)


def test_rule_table_is_immutable():
    with pytest.raises(ValueError):
        RULE_TABLE[0, 0] = 1.0


# --- 2. Inferencia ------------------------------------------------------------------------------


@pytest.mark.parametrize("i, j", [(0, 0), (1, 3), (4, 2), (2, 2), (4, 0)])
def test_at_grid_nodes_a_single_rule_speaks(params, i, j):
    centers = np.linspace(-1, 1, 5)
    theta, theta_dot = centers[i] * GAINS.theta_scale, centers[j] * GAINS.theta_dot_scale
    force = law(params, [0.0, 0.0, theta, theta_dot])  # x = 0 -> θ_ref = 0 -> e = θ
    assert force == pytest.approx(min(max(GAINS.force_scale * RULE_TABLE[i, j], -10), 10))


def test_near_the_origin_the_fuzzy_controller_is_exactly_a_pd(params, rng):
    """Con reglas sat(i+j), membresías triangulares y producto, la inferencia en la celda
    central es F = U·(e/E + θ̇/D): un PD con Kp = U/E y Kd = U/D."""
    for _ in range(50):
        e_n, d_n = rng.uniform(0, 0.5, size=2) * rng.choice([-1, 1], size=2)
        if abs(e_n + d_n) > 0.5:  # dentro del rombo lineal
            continue
        theta, theta_dot = e_n * GAINS.theta_scale, d_n * GAINS.theta_dot_scale
        expected = GAINS.force_scale * (theta / GAINS.theta_scale + theta_dot / GAINS.theta_dot_scale)
        assert law(params, [0.0, 0.0, theta, theta_dot]) == pytest.approx(expected, abs=1e-12)


def test_far_from_the_origin_the_output_saturates_at_the_output_scale(params):
    assert law(params, [0.0, 0.0, 0.2, 3.0]) == pytest.approx(GAINS.force_scale)
    assert law(params, [0.0, 0.0, -0.2, -3.0]) == pytest.approx(-GAINS.force_scale)


def test_signs_and_cascade(params):
    assert law(params, [0.0, 0.0, 0.05, 0.0]) > 0  # cae a la derecha -> empujar a la derecha
    assert law(params, [0.0, 0.0, 0.0, 0.5]) > 0
    assert law(params, [0.5, 0.0, 0.0, 0.0]) > 0  # fase no mínima, igual que el PID


def test_law_is_mirror_symmetric(params, rng):
    for state in rng.uniform(-0.2, 0.2, size=(30, 4)):
        assert law(params, -state) == pytest.approx(-law(params, state), abs=1e-12)


def test_vectorized_law_matches_scalar_calls(params, rng):
    states = rng.uniform(-0.2, 0.2, size=(20, 4))
    forces = law(params, states)
    for s, f in zip(states, forces):
        assert law(params, s) == f


@pytest.mark.parametrize(
    "kwargs",
    [dict(theta_scale=0.0), dict(theta_dot_scale=-1.0), dict(force_scale=0.0), dict(kp_x=np.nan)],
)
def test_invalid_gains_raise(kwargs):
    values = dict(theta_scale=0.1, theta_dot_scale=0.5, force_scale=5.0, kp_x=0.0, kd_x=0.0) | kwargs
    with pytest.raises(ValueError):
        FuzzyGains(**values)


def test_controller_rejects_invalid_reference_limit(params):
    with pytest.raises(ValueError):
        CascadeFuzzy(GAINS, params, theta_ref_limit=0.3)


# --- 3. Sintonización y sanity checks -----------------------------------------------------------


@pytest.mark.parametrize("initial_state", [[0.3, 0.0, 0.08, 0.0], [0.0, 0.0, 0.15, 2.0], [1.5, 1.0, 0.0, 0.0]])
def test_batch_simulator_reproduces_the_gymnasium_rollout(params, initial_state):
    batch = simulate_fuzzy_batch(GAINS, np.array(initial_state), params, n_steps=500, theta_ref_limit=LIMIT, record=True)
    task = CartPoleTask(action_mode="continuous")
    try:
        traj = run_episode(task, CascadeFuzzy(GAINS, params, theta_ref_limit=LIMIT), initial_state=initial_state)
    finally:
        task.close()
    n = traj.n_steps
    assert int(batch.steps_alive) == n
    np.testing.assert_array_equal(batch.states[: n + 1].astype(np.float32).astype(np.float64), traj.states)
    np.testing.assert_array_equal(batch.forces[:n], traj.forces)


def test_tuning_config_matches_pid_except_for_the_bounds():
    """Comparación justa: mismas CI, criterio, horizonte y presupuesto que el PID."""
    from cartpole_lab.controllers.pid_tuning import TuningConfig

    fuzzy, pid = FuzzyTuningConfig().to_dict(), TuningConfig().to_dict()
    assert {k: v for k, v in fuzzy.items() if k != "bounds"} == {k: v for k, v in pid.items() if k != "bounds"}


def test_tuning_is_deterministic_given_the_seed(params):
    tiny = FuzzyTuningConfig(n_initial_states=4, n_steps=100, de_seeds=(0,), popsize=5, maxiter=3)
    assert tune_fuzzy(tiny, params).best.gains == tune_fuzzy(tiny, params).best.gains


@pytest.fixture(scope="module")
def tuned():
    return load_tuned_fuzzy_gains()


def test_tuned_record_matches_the_current_config(tuned):
    record = json.loads((TUNING_DIR / "fuzzy.json").read_text(encoding="utf-8"))
    assert record["tuning"]["config"] == FuzzyTuningConfig().to_dict()
    assert FuzzyGains(**record["gains"]) == tuned


def test_tuned_fuzzy_is_locally_asymptotically_stable(tuned, params):
    assert fuzzy_closed_loop_spectral_radius(tuned, params, FuzzyTuningConfig().theta_ref_limit(params)) < 0.99


@pytest.fixture
def task():
    t = CartPoleTask(action_mode="continuous")
    yield t
    t.close()


def test_tuned_fuzzy_stabilizes_from_near_equilibrium(task, tuned, params):
    ctrl = CascadeFuzzy(tuned, params)
    for seed in range(1000, 1020):
        assert is_stabilized(run_episode(task, ctrl, seed=seed), params), f"semilla {seed}"


def test_tuned_fuzzy_recovers_the_reference_hard_initial_state(task, tuned, params):
    assert is_stabilized(run_episode(task, CascadeFuzzy(tuned, params), initial_state=REFERENCE_HARD_INITIAL_STATE), params)


@pytest.mark.parametrize("name", sorted(UNRECOVERABLE_INITIAL_STATES))
def test_extreme_initial_states_fail_loudly(task, tuned, params, name):
    initial_state, reasons = UNRECOVERABLE_INITIAL_STATES[name]
    traj = run_episode(task, CascadeFuzzy(tuned, params), initial_state=initial_state)
    assert traj.terminated and traj.termination_reason in reasons
    assert traj.n_steps > 1 and np.all(np.isfinite(traj.states))


def test_equivalent_pd_matches_the_central_cell_behaviour(params):
    pd = GAINS.equivalent_pd()
    assert pd["kp_theta"] == pytest.approx(GAINS.force_scale / GAINS.theta_scale)
    assert pd["kd_theta"] == pytest.approx(GAINS.force_scale / GAINS.theta_dot_scale)
    small = np.array([0.0, 0.0, 0.001, 0.002])
    assert law(params, small) == pytest.approx(pd["kp_theta"] * small[2] + pd["kd_theta"] * small[3])


def test_gains_round_trip_through_dict():
    assert FuzzyGains(**GAINS.to_dict()) == GAINS


def test_missing_tuning_file_raises_with_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="tune_fuzzy.py"):
        load_tuned_fuzzy_gains(tmp_path / "no_existe.json")
