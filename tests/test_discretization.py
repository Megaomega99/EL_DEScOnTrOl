"""Discretización del estado continuo en "cajas" para los métodos tabulares (paso 7)."""

import numpy as np
import pytest

from cartpole_lab.rl.discretization import BOXES_1983, DISCRETIZATIONS, BoxDiscretizer


def test_boxes_1983_has_the_historical_162_boxes():
    assert BOXES_1983.shape == (3, 3, 6, 3)
    assert BOXES_1983.n_states == 162


def test_boxes_1983_thresholds(params):
    one, six, fifty = np.radians(1.0), np.radians(6.0), np.radians(50.0)
    assert BOXES_1983.edges == ((-0.8, 0.8), (-0.5, 0.5), (-six, -one, 0.0, one, six), (-fifty, fifty))


def test_value_on_an_edge_goes_to_the_upper_box():
    """Misma convención que pole.c: `if (x < -0.8) box = 0; else if (x < 0.8) ...`."""
    d = BoxDiscretizer(edges=((0.0,), (), (), ()))
    assert d.box([-1e-12, 0, 0, 0])[0] == 0 and d.box([0.0, 0, 0, 0])[0] == 1


def test_every_box_is_reachable_and_indices_are_a_bijection(rng):
    """El índice plano recorre 0..n-1 sin huecos: la tabla Q no tiene filas inalcanzables."""
    d = BOXES_1983
    samples = rng.uniform([-2.4, -2, -0.21, -2], [2.4, 2, 0.21, 2], size=(200_000, 4))
    indices = {d.index(s) for s in samples}
    assert indices == set(range(d.n_states))


def test_index_and_box_are_consistent():
    d = BOXES_1983
    for flat in (0, 17, 80, 161):
        box = np.unravel_index(flat, d.shape)
        centre = [np.mean(_bounds(d.edges[v], box[v])) for v in range(4)]
        assert d.index(centre) == flat


def _bounds(edges, k):
    lower = edges[k - 1] if k > 0 else edges[0] - 1.0 if edges else -1.0
    upper = edges[k] if k < len(edges) else (edges[-1] + 1.0 if edges else 1.0)
    return lower, upper


def test_values_beyond_the_outer_edges_fall_in_the_extreme_boxes():
    assert BOXES_1983.box([100.0, -100.0, 1.0, 50.0]) == (2, 0, 5, 2)


def test_symmetric_states_map_to_mirrored_boxes():
    """Las cajas de BOXES son simétricas: s y −s caen en cajas espejo (salvo en los bordes)."""
    d = BOXES_1983
    state = np.array([1.0, 0.7, 0.03, 0.5])
    mirrored = tuple(n - 1 - b for n, b in zip(d.shape, d.box(state)))
    assert d.box(-state) == mirrored


@pytest.mark.parametrize(
    "edges",
    [((0.5, 0.1), (), (), ()), ((np.nan,), (), (), ()), ((), (), ())],
)
def test_invalid_edges_raise(edges):
    with pytest.raises(ValueError):
        BoxDiscretizer(edges=edges)


def test_rejects_state_with_wrong_shape():
    with pytest.raises(ValueError):
        BOXES_1983.index([0.0, 0.0, 0.0])


def test_named_discretizations_for_the_sensitivity_study():
    assert set(DISCRETIZATIONS) == {"sin_carro", "boxes_1983", "fina"}
    assert DISCRETIZATIONS["boxes_1983"] is BOXES_1983
    sizes = [DISCRETIZATIONS[k].n_states for k in ("sin_carro", "boxes_1983", "fina")]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3
    assert DISCRETIZATIONS["sin_carro"].shape[:2] == (1, 1)  # no ve ni x ni ẋ
