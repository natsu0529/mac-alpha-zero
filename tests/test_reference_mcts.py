import numpy as np
import pytest

from mac_alpha_zero.reference import Connect4, random_rollout_evaluator, run_mcts, solve

game = Connect4()


def search(state, num_simulations=400, seed=0):
    evaluator = random_rollout_evaluator(game, np.random.default_rng(seed))
    return run_mcts(game, state, evaluator, num_simulations)


def test_takes_immediate_win():
    state = game.from_moves("121212")
    assert search(state).action == 0


def test_blocks_immediate_loss():
    state = game.from_moves("12121")
    assert search(state).action == 0


def test_visit_counts_add_up():
    # The first simulation only expands the root.
    result = search(game.initial_state(), num_simulations=100)
    assert result.visit_counts.sum() == 99


def test_same_seed_same_result():
    state = game.from_moves("4433")
    a, b = search(state, seed=7), search(state, seed=7)
    np.testing.assert_array_equal(a.visit_counts, b.visit_counts)
    assert a.root_value == b.root_value


def test_root_value_of_won_position_is_positive():
    assert search(game.from_moves("121212")).root_value > 0.5


@pytest.mark.parametrize(
    ("moves", "winning_actions"),
    [
        ("4455", {2, 5}),  # open two on the bottom row: play 3 or 6 to make an open three
        # Positions found by random search. Winning actions are every move that forces a
        # win within 9 plies, precomputed with solve(depth=9) because that takes seconds.
        ("53141626746", {3, 4, 5}),
        ("3752662446", {2, 4}),
        ("121556556", {3}),
    ],
)
def test_finds_forced_win(moves, winning_actions):
    state = game.from_moves(moves)
    value, _ = solve(game, state, depth=5)
    assert value == 1.0
    assert search(state, num_simulations=1500).action in winning_actions


def test_rejects_terminal_state():
    with pytest.raises(ValueError):
        search(game.from_moves("1212121"))
