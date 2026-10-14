"""Utilidades de los notebooks (teaching.py): las dos versiones del notebook dependen de ellas."""

import matplotlib

matplotlib.use("Agg")  # sin ventana: los tests corren también en servidores
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cartpole_lab import teaching as tl
from cartpole_lab.controllers.lqr import LQRController, design_default_lqr
from cartpole_lab.evaluation.methods import PolicySpec, rl_policy_path
from cartpole_lab.evaluation.protocol import MAIN_METHODS


def test_force_map_of_the_lqr_is_its_saturated_linear_law(params):
    K = design_default_lqr(params).K
    theta_deg, theta_dot, force = tl.force_map(LQRController(K, params), params, n=7)
    for i, th in enumerate(np.radians(theta_deg)):
        for j, w in enumerate(theta_dot):
            expected = np.clip(-(K @ np.array([0.0, 0.0, th, w])).item(), -params.force_mag, params.force_mag)
            assert force[i, j] == pytest.approx(expected, abs=1e-12)


def test_force_map_resets_stateful_policies_before_each_state(params):
    class Accumulator:
        """Devuelve cuántas veces se ha llamado desde el último reset: con reset por estado, siempre 1."""

        def __init__(self):
            self.calls = 0

        def reset(self):
            self.calls = 0

        def __call__(self, state):
            self.calls += 1
            return float(self.calls)

    _, _, force = tl.force_map(Accumulator(), params, n=5)
    assert np.all(force == 1.0)


def test_compare_with_record_reports_agreement_difference_and_shape_mismatch():
    assert tl.compare_with_record([1.0, 2.0], [1.0, 2.0]) == "coincide con el registro"
    assert "difiere del registro en 1 de 3" in tl.compare_with_record([1, 2, 3], [1, 2, 4])
    assert "no comparable" in tl.compare_with_record([1, 2], [1, 2, 3])


def test_exported_seed_points_to_a_saved_policy_of_each_rl_method():
    for method in tl.EXPORTED_WEIGHTS:
        seed = tl.exported_seed(method)
        assert isinstance(seed, int)
        assert rl_policy_path(PolicySpec(method, seed)).exists()


def test_method_controllers_build_the_eight_methods_and_simulate_from_the_same_state(params):
    controllers = tl.method_controllers(params)
    assert tuple(controllers) == MAIN_METHODS
    s0 = (0.0, 0.0, 0.02, 0.0)
    subset = {k: controllers[k] for k in ("LQR", "Q-learning", "Actor-crítico")}
    trajectories = tl.simulate(subset, initial_state=s0)
    for traj in trajectories.values():
        # La observación inicial es float32, como la oficial de Gymnasium (env.py): igualdad a precisión float32.
        np.testing.assert_allclose(traj.states[0], s0, rtol=0, atol=1e-7)


def test_plot_trajectories_labels_failures_explicitly(params):
    class Push:
        action_mode = "continuous"

        def reset(self):
            pass

        def __call__(self, state):
            return 10.0  # empujar siempre a la derecha: el poste cae

    trajectories = tl.simulate({"LQR": LQRController(design_default_lqr(params).K, params), "Empuje": Push()},
                               initial_state=(0.0, 0.0, 0.02, 0.0))
    fig = tl.plot_trajectories(trajectories, params, title="prueba", impulse_time=5.0)
    try:
        assert len(fig.axes) == 3
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert labels[0] == "LQR"
        assert labels[1].startswith("Empuje  → falla")
    finally:
        plt.close(fig)


def test_load_result_and_figure_path_fail_explicitly_when_missing():
    with pytest.raises(FileNotFoundError):
        tl.load_result("no_existe/registro.json")
    with pytest.raises(FileNotFoundError):
        tl.figure_path("no_existe.png")
    assert tl.load_result("tuning/lqr.json")["K"]
