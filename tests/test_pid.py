"""Tests del PID en cascada (paso 3).

Tres bloques:
1. Ley de control: signos, límite de la referencia de ángulo, saturación y anti-windup.
2. Sintonización: el simulador por lotes reproduce Gymnasium y el optimizador es determinista.
3. Sanity checks con las ganancias sintonizadas (results/tuning/pid.json).
"""

import dataclasses

import numpy as np
import pytest

from cartpole_lab.controllers.pid import (
    CascadePID,
    PIDGains,
    load_tuned_gains,
    load_tuning_record,
    pid_force,
)
from cartpole_lab.disturbances import ForcePulse
from cartpole_lab.controllers.pid_tuning import (
    TuningConfig,
    closed_loop_spectral_radius,
    p_only_angle_spectral_radius,
    sample_tuning_initial_states,
    simulate_pid_batch,
    tune_pid,
)
from cartpole_lab.env import CartPoleTask
from cartpole_lab.rollout import run_episode

# Ganancias de prueba fijas (no sintonizadas): solo para verificar la mecánica de la ley.
GAINS = PIDGains(kp_theta=30.0, ki_theta=1.0, kd_theta=8.0, kp_x=0.03, kd_x=0.07)
LIMIT = 0.05  # rad, límite de la referencia de ángulo en los tests unitarios


def law(params, state, integral=0.0, gains=GAINS):
    return pid_force(
        state, integral, gains, dt=params.tau, force_limit=params.force_mag, theta_ref_limit=LIMIT
    )


# --- 1. Ley de control --------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value", [("kp_theta", -1.0), ("kd_x", np.nan), ("ki_theta", np.inf)]
)
def test_gains_must_be_finite_and_non_negative(field, value):
    kwargs = dict(kp_theta=1.0, ki_theta=0.0, kd_theta=1.0, kp_x=0.0, kd_x=0.0)
    kwargs[field] = value
    with pytest.raises(ValueError):
        PIDGains(**kwargs)


@pytest.mark.parametrize("state", [[0, 0, 0.05, 0], [0, 0, 0, 0.5]])
def test_pole_leaning_or_falling_right_commands_push_right(params, state):
    force, _ = law(params, state)
    assert force > 0.0


def test_cart_right_of_target_first_pushes_right(params):
    """Fase no mínima: para volver a la izquierda, primero hay que inclinar el poste
    hacia la izquierda, y eso exige empujar el carro hacia la DERECHA."""
    force, _ = law(params, [0.5, 0.0, 0.0, 0.0])
    assert force > 0.0


def test_law_is_mirror_symmetric(params, rng):
    for state in rng.uniform(-0.1, 0.1, size=(20, 4)):
        f_pos, i_pos = law(params, state)
        f_neg, i_neg = law(params, -state)
        assert f_neg == -f_pos and i_neg == -i_pos


def test_angle_reference_is_clamped(params):
    # |kp_x · x| = 0.06 y 0.6 rad superan el límite de 0.05 rad -> misma fuerza.
    f_far, _ = law(params, [2.0, 0.0, 0.0, 0.0])
    f_farther, _ = law(params, [20.0, 0.0, 0.0, 0.0])
    assert f_far == f_farther == pytest.approx(GAINS.kp_theta * LIMIT)


def test_force_saturates_at_actuator_limit(params):
    force, _ = law(params, [0.0, 0.0, 0.0, 5.0])  # kd·θ̇ = 40 N
    assert force == params.force_mag


def test_integral_accumulates_error_times_dt(params):
    state = [0.1, 0.0, 0.02, 0.0]
    theta_ref = -(GAINS.kp_x * 0.1)
    _, integral = law(params, state, integral=0.3)
    assert integral == pytest.approx(0.3 + (0.02 - theta_ref) * params.tau)


def test_anti_windup_freezes_integral_when_saturated_in_error_direction(params):
    _, integral = law(params, [0.0, 0.0, 0.1, 5.0], integral=0.3)  # satura con e > 0
    assert integral == 0.3


def test_integral_still_unwinds_while_saturated_against_the_error(params):
    # ki·I = 20 N satura hacia +, pero e < 0: integrar ayuda a salir de la saturación.
    _, integral = law(params, [0.0, 0.0, -0.01, 0.0], integral=20.0)
    assert integral < 20.0


def test_vectorized_law_matches_scalar_calls(params, rng):
    states = rng.uniform(-0.2, 0.2, size=(30, 4))
    integrals = rng.uniform(-1, 1, size=30)
    forces, new_integrals = law(params, states, integrals)
    for s, i, f, ni in zip(states, integrals, forces, new_integrals):
        f1, ni1 = law(params, s, i)
        assert (f1, ni1) == (f, ni)


def test_controller_keeps_integral_between_calls_and_reset_clears_it(params):
    ctrl = CascadePID(GAINS, params, theta_ref_limit=LIMIT)
    ctrl.reset()
    first = ctrl([0.0, 0.0, 0.05, 0.0])
    second = ctrl([0.0, 0.0, 0.05, 0.0])
    assert second > first  # el término integral creció
    ctrl.reset()
    assert ctrl([0.0, 0.0, 0.05, 0.0]) == first


def test_default_reference_limit_is_a_quarter_of_the_failure_angle(params):
    ctrl = CascadePID(GAINS, params)
    assert ctrl.theta_ref_limit == pytest.approx(0.25 * params.theta_threshold_radians)


def test_gains_round_trip_through_dict():
    assert PIDGains(**GAINS.to_dict()) == GAINS


def test_missing_tuning_file_raises_with_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="tune_pid.py"):
        load_tuned_gains(tmp_path / "no_existe.json")


@pytest.mark.parametrize("limit", [0.0, -0.01, 0.3])
def test_controller_rejects_invalid_reference_limit(params, limit):
    # 0.3 rad > 12°: la cascada podría pedir un ángulo que ya termina el episodio.
    with pytest.raises(ValueError):
        CascadePID(GAINS, params, theta_ref_limit=limit)


# --- 2. Por qué no Ziegler-Nichols, y la sintonización ---------------------------------------


def test_p_only_angle_control_is_never_asymptotically_stable(params):
    """No existe una "ganancia última" Ku: con P puro sobre θ el radio espectral es ≥ 1
    para TODA Kp (diverge si Kp < a/b; oscilador que Euler amplifica si Kp > a/b)."""
    for kp in np.linspace(0.0, 300.0, 61):
        assert p_only_angle_spectral_radius(kp, params) >= 1.0 - 1e-9


@pytest.mark.parametrize(
    "initial_state",
    [
        [0.3, 0.0, 0.08, 0.0],  # sobrevive los 500 pasos
        [0.0, 0.0, 0.15, 2.0],  # falla en el paso 2
        [1.5, 1.0, 0.0, 0.0],  # falla a mitad de horizonte (paso 51): transición vivo -> congelado
    ],
)
def test_batch_simulator_reproduces_the_gymnasium_rollout(params, initial_state):
    """El optimizador evalúa miles de candidatos con el simulador por lotes; este test
    garantiza que optimiza el sistema REAL (misma física, misma cuantización float32)."""
    batch = simulate_pid_batch(
        GAINS, np.array(initial_state), params, n_steps=500, theta_ref_limit=LIMIT, record=True
    )
    task = CartPoleTask(action_mode="continuous")
    try:
        traj = run_episode(task, CascadePID(GAINS, params, theta_ref_limit=LIMIT), initial_state=initial_state)
    finally:
        task.close()
    n = traj.n_steps
    assert int(batch.steps_alive) == n
    observed = batch.states[: n + 1].astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(observed, traj.states)
    np.testing.assert_array_equal(batch.forces[:n], traj.forces)


def test_tuning_initial_states_are_reproducible_and_inside_the_box():
    config = TuningConfig()
    a, b = sample_tuning_initial_states(config), sample_tuning_initial_states(config)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (config.n_initial_states, 4)
    assert np.all(np.abs(a) <= np.array(config.initial_state_box))


def test_tuning_is_deterministic_given_the_seed(params):
    tiny = TuningConfig(n_initial_states=4, n_steps=100, de_seeds=(0,), popsize=5, maxiter=3)
    first, second = tune_pid(tiny, params), tune_pid(tiny, params)
    assert first.best.gains == second.best.gains
    assert first.best.cost == second.best.cost


def test_objective_penalizes_failure(params):
    config = TuningConfig()
    states = sample_tuning_initial_states(config)
    no_control = PIDGains(0.0, 0.0, 0.0, 0.0, 0.0)
    batch = simulate_pid_batch(
        no_control, states, params, n_steps=config.n_steps, theta_ref_limit=LIMIT
    )
    assert not np.any(batch.alive)  # sin control, todas las condiciones iniciales fallan
    # El error imputado tras el fallo (2 por paso) acota el ITAE por el caso "fallo inmediato".
    horizon = config.n_steps * params.tau
    assert np.all(batch.itae <= 2.0 * horizon**2 / 2 + 2.0 * horizon * params.tau)


# --- 3. Sanity checks con las ganancias sintonizadas -----------------------------------------


@pytest.fixture(scope="module")
def tuned():
    record = load_tuning_record()
    return PIDGains(**record["gains"]), record


def test_tuned_record_matches_the_current_tuning_config(tuned):
    """Si alguien cambia TuningConfig sin re-sintonizar, las ganancias guardadas son obsoletas."""
    gains, record = tuned
    assert record["tuning"]["config"] == TuningConfig().to_dict()
    assert load_tuned_gains() == gains


def test_tuned_pid_has_no_divergent_mode_and_physical_modes_decay(tuned, params):
    """Se comprueban por separado: con Ki ≈ 0 el modo del integrador queda en λ ≈ 1, y el
    radio espectral global (≈ 1) no informaría de la convergencia real de x y θ."""
    gains, _ = tuned
    limit = TuningConfig().theta_ref_limit(params)
    assert closed_loop_spectral_radius(gains, params, limit) <= 1.0 + 1e-9
    physical = closed_loop_spectral_radius(gains, params, limit, include_integral=False)
    assert physical < 0.99  # constante de tiempo dominante -τ/ln(ρ) < 2 s


def test_spectral_radius_of_a_truly_unstable_loop_is_detected(params):
    """Control del propio análisis: ganancias que no vencen a la gravedad deben dar ρ > 1."""
    weak = PIDGains(kp_theta=5.0, ki_theta=0.0, kd_theta=1.0, kp_x=0.0, kd_x=0.0)
    assert closed_loop_spectral_radius(weak, params, LIMIT) > 1.0


# --- Por qué el optimizador anula la I: no hay nada que rechazar sin perturbación constante ---


def _final_x_under_constant_force(gains, params, force):
    task = CartPoleTask(action_mode="continuous", disturbance=ForcePulse(0, 500, force))
    try:
        traj = run_episode(
            task,
            CascadePID(gains, params, theta_ref_limit=TuningConfig().theta_ref_limit(params)),
            initial_state=np.zeros(4),
        )
    finally:
        task.close()
    assert traj.termination_reason == "time_limit"
    return traj.states[-1, 0]


def test_without_integral_a_constant_force_leaves_the_predicted_position_offset(tuned, params):
    """Equilibrio PD con fuerza externa d: F = -d = Kp_θ·Kp_x·x  =>  x_ss = -d / (Kp_θ·Kp_x)."""
    gains, _ = tuned
    predicted = -1.0 / (gains.kp_theta * gains.kp_x)
    assert _final_x_under_constant_force(gains, params, 1.0) == pytest.approx(predicted, rel=0.02)


def test_integral_action_removes_the_offset(tuned, params):
    """Con I, el integrador solo se detiene si e = θ - θ_ref = 0 -> θ_ref = 0 -> x = 0.
    Ki = 80 es un valor ILUSTRATIVO (no sintonizado) elegido para que se vea en 10 s."""
    gains, _ = tuned
    pd_offset = abs(1.0 / (gains.kp_theta * gains.kp_x))
    with_integral = dataclasses.replace(gains, ki_theta=80.0)
    assert abs(_final_x_under_constant_force(with_integral, params, 1.0)) < 0.1 * pd_offset


def test_tuned_pid_stabilizes_from_near_equilibrium(tuned, params):
    """Sanity check positivo: 20 condiciones iniciales de la distribución oficial de CartPole-v1."""
    gains, _ = tuned
    task = CartPoleTask(action_mode="continuous")
    ctrl = CascadePID(gains, params, theta_ref_limit=TuningConfig().theta_ref_limit(params))
    last_second = int(round(1.0 / params.tau))
    try:
        for seed in range(1000, 1020):
            traj = run_episode(task, ctrl, seed=seed)
            assert traj.termination_reason == "time_limit", f"semilla {seed}"
            tail = traj.states[-last_second:]
            assert np.max(np.abs(tail[:, 2])) < np.radians(0.5), f"semilla {seed}: θ no converge"
            assert np.max(np.abs(tail[:, 0])) < 0.05, f"semilla {seed}: x no converge"
    finally:
        task.close()


def test_tuned_pid_recovers_the_worst_corner_of_the_tuning_box(tuned, params):
    gains, _ = tuned
    corner = np.array(TuningConfig().initial_state_box)
    task = CartPoleTask(action_mode="continuous")
    try:
        traj = run_episode(
            task,
            CascadePID(gains, params, theta_ref_limit=TuningConfig().theta_ref_limit(params)),
            initial_state=corner,
        )
    finally:
        task.close()
    assert traj.termination_reason == "time_limit"
    assert abs(traj.states[-1, 0]) < 0.05 and abs(traj.states[-1, 2]) < np.radians(0.5)


@pytest.mark.parametrize(
    "extreme, reasons",
    [
        ([0.0, 0.0, 0.15, 2.0], {"pole_angle_limit"}),  # cae demasiado rápido
        ([2.2, 2.0, 0.0, 0.0], {"cart_position_limit", "pole_angle_limit"}),  # carro lanzado al borde
    ],
)
def test_extreme_initial_states_fail_loudly(tuned, params, extreme, reasons):
    """Sanity check negativo: el fallo es explícito (motivo registrado), no silencioso."""
    gains, _ = tuned
    task = CartPoleTask(action_mode="continuous")
    try:
        traj = run_episode(
            task,
            CascadePID(gains, params, theta_ref_limit=TuningConfig().theta_ref_limit(params)),
            initial_state=extreme,
        )
    finally:
        task.close()
    assert traj.terminated and traj.termination_reason in reasons
    assert traj.n_steps > 1 and np.all(np.isfinite(traj.states))


def test_falling_pole_is_unrecoverable_for_any_admissible_force(params):
    """Justifica el caso anterior: ni la fuerza máxima constante hacia el lado de la caída
    lo evita (distancia de frenado θ̇²/(2|θ̈_max|) ≈ 0.165 rad > 0.06 rad de margen)."""

    class FullPushRight:
        action_mode = "continuous"

        def reset(self):
            pass

        def __call__(self, state):
            return params.force_mag

    task = CartPoleTask(action_mode="continuous")
    try:
        traj = run_episode(task, FullPushRight(), initial_state=[0.0, 0.0, 0.15, 2.0])
    finally:
        task.close()
    assert traj.termination_reason == "pole_angle_limit"
