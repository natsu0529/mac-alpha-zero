import mlx.core as mx
import numpy as np
import pytest

from mac_alpha_zero import games, reference
from mac_alpha_zero.mcts import random_rollout_evaluator, search

ref = reference.Connect4()
game = games.Connect4()

# A deterministic evaluator built only from integer arithmetic and one division, so NumPy
# and MLX produce bit-identical float64 priors and values.
CENTRALITY = np.array([0, 1, 2, 3, 2, 1, 0])


def ref_evaluator(state):
    empty = (state.board == 0).sum(axis=0)
    raw = 1 + empty + 3 * (np.arange(7) == 3)
    priors = raw / raw.sum()
    value = float(np.clip((state.board * state.player * CENTRALITY).sum() / 32, -0.9, 0.9))
    return priors, value


def batched_evaluator(state):
    board = state["board"].astype(mx.float64)
    player = state["player"].astype(mx.float64)[:, None, None]
    empty = (board == 0).sum(axis=1)
    raw = 1 + empty + 3 * (mx.arange(7) == 3)
    priors = raw / raw.sum(axis=-1, keepdims=True)
    centrality = mx.array(CENTRALITY, dtype=mx.float64)
    value = mx.clip((board * player * centrality).sum(axis=(1, 2)) / 32, -0.9, 0.9)
    return priors, value


def random_positions(count, seed):
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < count:
        state = ref.initial_state()
        for _ in range(rng.integers(0, 20)):
            if ref.terminal_value(state) is not None:
                break
            state = ref.step(state, int(rng.choice(np.flatnonzero(ref.legal_mask(state)))))
        if ref.terminal_value(state) is None:
            positions.append(state)
    return positions


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize("num_simulations", [1, 2, 10, 64, 300])
def test_matches_reference_exactly_in_float64(cpu, num_simulations):
    positions = [ref.initial_state(), ref.from_moves("121212"), ref.from_moves("12121")]
    positions += random_positions(13, seed=num_simulations)
    result = search(
        game,
        game.from_reference(positions),
        batched_evaluator,
        num_simulations,
        dtype=mx.float64,
    )
    for i, state in enumerate(positions):
        expected = reference.run_mcts(ref, state, ref_evaluator, num_simulations)
        np.testing.assert_array_equal(np.array(result.visit_counts[i]), expected.visit_counts)
        assert float(result.root_value[i]) == expected.root_value


def test_trees_in_a_batch_are_independent(cpu):
    positions = random_positions(8, seed=1)
    together = search(
        game, game.from_reference(positions), batched_evaluator, 100, dtype=mx.float64
    )
    for i, state in enumerate(positions):
        alone = search(game, game.from_reference([state]), batched_evaluator, 100, dtype=mx.float64)
        np.testing.assert_array_equal(np.array(together.visit_counts[i]), alone.visit_counts[0])


def test_float32_picks_the_reference_move():
    positions = random_positions(16, seed=2)

    def evaluator32(state):
        priors, value = batched_evaluator(state)
        return priors.astype(mx.float32), value.astype(mx.float32)

    with mx.stream(mx.cpu):
        result = search(game, game.from_reference(positions), evaluator32, 200)
    for i, state in enumerate(positions):
        expected = reference.run_mcts(ref, state, ref_evaluator, 200)
        assert int(mx.argmax(result.visit_counts[i])) == expected.action


def test_visit_counts_add_up():
    result = search(game, game.init(4), random_rollout_evaluator(game), 50)
    assert np.array(result.visit_counts.sum(axis=1)).tolist() == [49] * 4


def test_random_rollouts_find_wins_and_blocks():
    # Take the win, block the loss, and two forced wins (see test_reference_mcts.py).
    positions = [ref.from_moves(m) for m in ["121212", "12121", "4455", "121556556"]]
    winning = [{0}, {0}, {2, 5}, {3}]
    result = search(game, game.from_reference(positions), random_rollout_evaluator(game), 1500)
    for i, allowed in enumerate(winning):
        assert int(mx.argmax(result.visit_counts[i])) in allowed


def test_rejects_terminal_root():
    with pytest.raises(ValueError):
        search(
            game,
            game.from_reference([ref.from_moves("1212121")]),
            random_rollout_evaluator(game),
            10,
        )
