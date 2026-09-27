import random

import numpy as np
import pytest

from cartpole_lab.seeding import set_global_seeds


def _draw():
    return random.random(), np.random.rand()


def test_global_seeds_make_draws_reproducible():
    rng_a = set_global_seeds(123)
    draws_a = _draw(), rng_a.random()
    rng_b = set_global_seeds(123)
    draws_b = _draw(), rng_b.random()
    assert draws_a == draws_b


def test_torch_is_seeded_when_installed():
    torch = pytest.importorskip("torch")
    set_global_seeds(7)
    a = torch.rand(3)
    set_global_seeds(7)
    assert torch.equal(a, torch.rand(3))


@pytest.mark.parametrize("bad", [-1, 1.5, "3", None, 2**32])
def test_invalid_seed_raises(bad):
    with pytest.raises((TypeError, ValueError)):
        set_global_seeds(bad)
