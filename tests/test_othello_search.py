import mlx.core as mx
import numpy as np
import pytest

from mac_alpha_zero import games, kernels, reference
from mac_alpha_zero.gumbel import gumbel_search
from mac_alpha_zero.mcts import search

from .test_batched_mcts import cpu  # noqa: F401

ref = reference.Othello()
game = games.Othello()

# Priors and values are multiples of a power of two, so every sum over the legal actions
# is exact and NumPy and MLX agree bit for bit in float64 whatever order they add in.
# (With 65 actions the two libraries do add in different orders.)
PRIORS = (1 + np.arange(ref.num_actions) % 5) / 256


def ref_evaluator(state):
    value = float(np.clip((state.board * state.player).sum() / 16, -0.9, 0.9))
    return PRIORS, value


def batched_evaluator(state):
    relative = game.boards(state) * state["player"][:, None, None]
    margin = relative.astype(mx.float64).sum(axis=(1, 2))
    priors = mx.broadcast_to(mx.array(PRIORS, dtype=mx.float64), (margin.shape[0], ref.num_actions))
    return priors, mx.clip(margin / 16, -0.9, 0.9)


def random_positions(count, seed, max_moves=58):
    """Non-terminal positions from random play, including ones where the mover must pass."""
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < count:
        state = ref.initial_state()
        for _ in range(rng.integers(0, max_moves + 1)):
            if ref.terminal_value(state) is not None:
                break
            state = ref.step(state, int(rng.choice(np.flatnonzero(ref.legal_mask(state)))))
        if ref.terminal_value(state) is None:
            positions.append(state)
    return positions


@pytest.mark.parametrize("num_simulations", [1, 10, 64, 200])
def test_search_matches_reference_exactly_in_float64(cpu, num_simulations):  # noqa: F811
    positions = [ref.initial_state()] + random_positions(11, seed=num_simulations)
    result = search(
        game, game.from_reference(positions), batched_evaluator, num_simulations, dtype=mx.float64
    )
    for i, state in enumerate(positions):
        expected = reference.run_mcts(ref, state, ref_evaluator, num_simulations)
        np.testing.assert_array_equal(np.array(result.visit_counts[i]), expected.visit_counts)
        assert float(result.root_value[i]) == expected.root_value


def test_search_handles_forced_passes_and_game_ends(cpu):  # noqa: F811
    # Late positions: the trees reach passes and finished games within a few plies.
    positions = [
        s for s in random_positions(200, seed=3, max_moves=60) if (s.board != 0).sum() > 54
    ]
    assert len(positions) >= 8
    assert any(ref.legal_mask(s)[-1] for s in positions)
    result = search(game, game.from_reference(positions), batched_evaluator, 150, dtype=mx.float64)
    for i, state in enumerate(positions):
        expected = reference.run_mcts(ref, state, ref_evaluator, 150)
        np.testing.assert_array_equal(np.array(result.visit_counts[i]), expected.visit_counts)
        assert float(result.root_value[i]) == expected.root_value


def test_gumbel_matches_reference_in_float64(cpu):  # noqa: F811
    positions = random_positions(8, seed=5)
    rng = np.random.default_rng(0)
    gumbel = rng.gumbel(size=(len(positions), ref.num_actions))
    result = gumbel_search(
        game,
        game.from_reference(positions),
        batched_evaluator,
        48,
        gumbel=mx.array(gumbel, dtype=mx.float64),
        dtype=mx.float64,
    )
    for i, state in enumerate(positions):
        expected = reference.run_gumbel_mcts(ref, state, ref_evaluator, 48, gumbel[i])
        assert int(result.action[i]) == expected.action
        np.testing.assert_array_equal(np.array(result.visit_counts[i]), expected.visit_counts)
        np.testing.assert_allclose(
            np.array(result.improved_policy[i]), expected.improved_policy, rtol=1e-6, atol=1e-9
        )


@pytest.mark.skipif(not kernels.is_available(), reason="needs the Metal GPU")
def test_kernels_match_pure_mlx():
    def evaluator(state):
        relative = game.boards(state) * state["player"][:, None, None]
        margin = relative.astype(mx.float32).sum(axis=(1, 2))
        priors = mx.array(PRIORS.astype(np.float32))
        return mx.broadcast_to(priors, (margin.shape[0], ref.num_actions)), mx.clip(
            margin / 16, -0.9, 0.9
        )

    root = game.from_reference(random_positions(32, seed=7))
    plain = search(game, root, evaluator, 100)
    fused = search(game, root, evaluator, 100, use_kernels=True)
    np.testing.assert_array_equal(np.array(fused.visit_counts), np.array(plain.visit_counts))
