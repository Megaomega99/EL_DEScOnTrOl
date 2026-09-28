"""Métricas de aprendizaje y regla de selección (sin simular: datos sintéticos)."""

import numpy as np

from cartpole_lab.rl.evaluation import (
    episodes_to_solve_greedy,
    episodes_to_solve_moving_average,
    moving_average,
    select_by_parsimony,
)


def test_greedy_solve_point_is_the_first_evaluation_at_or_above_threshold():
    assert episodes_to_solve_greedy([100, 200, 300, 400], [10, 480, 300, 500]) == 200
    assert episodes_to_solve_greedy([100, 200], [474.9, 100]) is None


def test_moving_average_window_semantics():
    np.testing.assert_allclose(moving_average([1, 2, 3, 4], window=2), [1.5, 2.5, 3.5])


def test_moving_average_solve_point_counts_episodes_from_one():
    """Con ventana 3 y retornos [0,0,500,500,500], la media llega a 500 en los episodios 3..5,
    es decir, tras el episodio 5 (1-indexado)."""
    assert episodes_to_solve_moving_average([0, 0, 500, 500, 500], window=3, threshold=475) == 5
    assert episodes_to_solve_moving_average([500] * 3, window=3, threshold=475) == 3
    assert episodes_to_solve_moving_average([100] * 10, window=3, threshold=475) is None


def test_parsimony_prefers_the_smaller_candidate_within_one_std():
    candidates = {
        "grande": {"mean": 330.0, "std": 90.0, "n_states": 360},
        "media": {"mean": 320.0, "std": 85.0, "n_states": 162},
        "pequeña": {"mean": 200.0, "std": 100.0, "n_states": 18},
    }
    result = select_by_parsimony(candidates)
    assert result["best_by_mean"] == "grande"
    assert set(result["within_one_std"]) == {"grande", "media"}
    assert result["chosen"] == "media"


def test_parsimony_keeps_the_best_when_it_is_clearly_ahead():
    candidates = {"a": {"mean": 400.0, "std": 10.0, "n_states": 500}, "b": {"mean": 300.0, "std": 5.0, "n_states": 10}}
    assert select_by_parsimony(candidates)["chosen"] == "a"


def test_first_crossing_returns_the_index_not_the_value():
    from cartpole_lab.rl.evaluation import first_crossing

    assert first_crossing([100, 480, 490]) == 1 and first_crossing([1, 2]) is None
