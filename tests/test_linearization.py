"""Linealización en el equilibrio vertical: las matrices analíticas deben coincidir con el simulador."""

import dataclasses

import numpy as np
import pytest

from cartpole_lab.dynamics import continuous_dynamics, euler_step, finite_difference_jacobian
from cartpole_lab.linearization import (
    continuous_linearization,
    controllability_matrix,
    euler_discretization,
    zoh_discretization,
)


def test_analytic_continuous_jacobians_match_finite_differences(params):
    A_c, B_c = continuous_linearization(params)
    A_fd = finite_difference_jacobian(lambda s: continuous_dynamics(s, 0.0, params), np.zeros(4))
    B_fd = finite_difference_jacobian(lambda f: continuous_dynamics(np.zeros(4), f[0], params), np.zeros(1))
    np.testing.assert_allclose(A_c, A_fd, rtol=0, atol=1e-8)
    np.testing.assert_allclose(B_c, B_fd[:, 0], rtol=0, atol=1e-8)


def test_euler_discretization_is_the_jacobian_of_the_simulator_step(params):
    """El modelo discreto del LQR es la linealización EXACTA del mapa que ejecuta Gymnasium."""
    A_d, B_d = euler_discretization(*continuous_linearization(params), params.tau)
    A_fd = finite_difference_jacobian(lambda s: euler_step(s, 0.0, params), np.zeros(4))
    B_fd = finite_difference_jacobian(lambda f: euler_step(np.zeros(4), f[0], params), np.zeros(1))
    np.testing.assert_allclose(A_d, A_fd, rtol=0, atol=1e-9)
    np.testing.assert_allclose(B_d, B_fd[:, 0], rtol=0, atol=1e-9)


def test_documented_numerical_values(params):
    """Valores que aparecen en docs/04_lqr.md (4 decimales)."""
    A_c, B_c = continuous_linearization(params)
    assert A_c[3, 2] == pytest.approx(15.7756, abs=1e-4)  # a = g / D
    assert A_c[1, 2] == pytest.approx(-0.7171, abs=1e-4)  # −(m l / M) · a
    assert B_c[1] == pytest.approx(0.9756, abs=1e-4)
    assert B_c[3] == pytest.approx(-1.4634, abs=1e-4)  # −b


def test_open_loop_eigenvalues_show_one_unstable_mode(params):
    A_c, _ = continuous_linearization(params)
    eig = np.sort(np.linalg.eigvals(A_c).real)
    lam = np.sqrt(A_c[3, 2])
    np.testing.assert_allclose(eig, [-lam, 0.0, 0.0, lam], atol=1e-9)  # carro: doble integrador


def test_system_is_controllable(params):
    A_d, B_d = euler_discretization(*continuous_linearization(params), params.tau)
    assert np.linalg.matrix_rank(controllability_matrix(A_d, B_d)) == 4


def test_zoh_discretization_matches_fine_integration_of_the_linear_ode(params):
    """Verificación independiente de la fórmula de Van Loan: se integra ṡ = A s + B F con
    10 000 subpasos (F constante durante τ) y se compara con A_zoh s0 + B_zoh F."""
    A_c, B_c = continuous_linearization(params)
    A_z, B_z = zoh_discretization(A_c, B_c, params.tau)
    s0, force, n = np.array([0.1, -0.2, 0.05, 0.3]), 2.0, 10_000
    s, h = s0.copy(), params.tau / n
    for _ in range(n):  # RK4 sobre el sistema lineal
        k1 = A_c @ s + B_c * force
        k2 = A_c @ (s + h / 2 * k1) + B_c * force
        k3 = A_c @ (s + h / 2 * k2) + B_c * force
        k4 = A_c @ (s + h * k3) + B_c * force
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    np.testing.assert_allclose(A_z @ s0 + B_z * force, s, rtol=0, atol=1e-12)


def test_euler_and_zoh_differ_at_second_order_in_tau(params):
    """||e^{Aτ} − (I + Aτ)|| ≈ τ²/2·||A²||: al dividir τ entre 2, la discrepancia cae ~4×."""

    def discrepancy(tau):
        A_c, B_c = continuous_linearization(params)
        A_e, _ = euler_discretization(A_c, B_c, tau)
        A_z, _ = zoh_discretization(A_c, B_c, tau)
        return np.linalg.norm(A_z - A_e)

    ratio = discrepancy(params.tau) / discrepancy(params.tau / 2)
    assert 3.8 < ratio < 4.2


def test_linearization_follows_the_params_it_is_given(params):
    heavier = dataclasses.replace(params, masspole=0.5)
    assert not np.allclose(continuous_linearization(heavier)[0], continuous_linearization(params)[0])
