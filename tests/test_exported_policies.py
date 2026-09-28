"""Los pesos exportados (results/weights) se cargan y actúan sin PyTorch: contrato para el navegador."""

import json

import numpy as np
import pytest

from cartpole_lab.paths import WEIGHTS_DIR
from cartpole_lab.rl.actor_critic import ACPolicy
from cartpole_lab.rl.dqn import DQNPolicy, load_network

NETWORKS = ["dqn", "double_dqn", "actor_critic", "actor_critic_final"]


@pytest.mark.parametrize("name", NETWORKS)
def test_exported_network_follows_the_documented_format(name, params):
    net = load_network(WEIGHTS_DIR / f"{name}.json")
    assert net["format"] == "mlp-json-v1"
    np.testing.assert_allclose(net["input_scales"], [params.x_threshold, 1.0, params.theta_threshold_radians, 1.5])
    assert net["layers"][-1]["activation"] == "linear" and all(l["activation"] == "relu" for l in net["layers"][:-1])
    assert "VALIDACIÓN" in net["metadata"]["selected_by"] or "FINAL" in net["metadata"]["selected_by"]
    if "actor_critic" in name:
        assert isinstance(net["log_std"], float)  # no null: el formato documentado lo incluye
    for layer in net["layers"]:
        assert np.asarray(layer["W"]).shape[0] == np.asarray(layer["b"]).shape[0]


@pytest.mark.parametrize("name", NETWORKS)
def test_exported_policy_returns_an_admissible_force(name, params, rng):
    net = load_network(WEIGHTS_DIR / f"{name}.json")
    if "force_levels" in net:
        policy = DQNPolicy(net["layers"], tuple(net["input_scales"]), tuple(net["force_levels"]))
    else:
        policy = ACPolicy(net["layers"], tuple(net["input_scales"]), net["force_limit"])
    for state in rng.uniform(-0.2, 0.2, size=(20, 4)):
        assert abs(policy(state)) <= params.force_mag


@pytest.mark.parametrize("method", ["q_learning", "sarsa"])
def test_exported_q_tables_have_the_documented_shape(method):
    w = json.loads((WEIGHTS_DIR / f"tabular_{method}.json").read_text(encoding="utf-8"))
    n_states = int(np.prod([len(e) + 1 for e in w["edges"]]))
    assert np.asarray(w["q_table"]).shape == (n_states, len(w["force_levels"])) == (162, 5)
