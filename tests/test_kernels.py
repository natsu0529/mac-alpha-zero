import mlx.core as mx
import numpy as np
import pytest

from mac_alpha_zero import games, kernels
from mac_alpha_zero.mcts import random_rollout_evaluator, search

from .test_batched_mcts import CENTRALITY, random_positions

pytestmark = pytest.mark.skipif(not kernels.is_available(), reason="needs the Metal GPU")

game = games.Connect4()


def evaluator(state):
    board = state["board"].astype(mx.float32)
    player = state["player"].astype(mx.float32)[:, None, None]
    empty = (board == 0).sum(axis=1)
    raw = 1 + empty + 3 * (mx.arange(7) == 3)
    priors = raw / raw.sum(axis=-1, keepdims=True)
    centrality = mx.array(CENTRALITY, dtype=mx.float32)
    value = mx.clip((board * player * centrality).sum(axis=(1, 2)) / 32, -0.9, 0.9)
    return priors, value


@pytest.mark.parametrize("num_simulations", [1, 2, 10, 64, 300])
def test_kernels_match_pure_mlx(num_simulations):
    root = game.from_reference(random_positions(128, seed=num_simulations))
    expected = search(game, root, evaluator, num_simulations)
    result = search(game, root, evaluator, num_simulations, use_kernels=True)
    np.testing.assert_array_equal(np.array(result.visit_counts), np.array(expected.visit_counts))
    np.testing.assert_allclose(
        np.array(result.root_value), np.array(expected.root_value), rtol=1e-5, atol=1e-6
    )


def test_kernels_reach_terminal_nodes():
    # Near the end of the game most simulations hit terminal nodes again.
    root = game.from_reference(random_positions(32, seed=9, min_moves=30))
    expected = search(game, root, evaluator, 200)
    result = search(game, root, evaluator, 200, use_kernels=True)
    np.testing.assert_array_equal(np.array(result.visit_counts), np.array(expected.visit_counts))


def test_kernels_with_random_rollouts_find_the_win():
    from mac_alpha_zero import reference

    ref = reference.Connect4()
    root = game.from_reference([ref.from_moves("121212"), ref.from_moves("12121")])
    result = search(game, root, random_rollout_evaluator(game), 400, use_kernels=True)
    assert np.array(mx.argmax(result.visit_counts, axis=1)).tolist() == [0, 0]


def test_kernels_reject_float64():
    with pytest.raises(ValueError):
        search(game, game.init(1), evaluator, 4, dtype=mx.float64, use_kernels=True)
