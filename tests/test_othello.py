import numpy as np
import pytest

from mac_alpha_zero import reference
from mac_alpha_zero.reference.othello import PASS

game = reference.Othello()

# Number of move sequences of each length from the starting position.
PERFT = [1, 4, 12, 56, 244, 1396, 8200]


def perft(state, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(
        perft(game.step(state, a), depth - 1) for a in np.flatnonzero(game.legal_mask(state))
    )


def position(rows: list[str], player: int) -> reference.OthelloState:
    """Rows of 'X' (first player), 'O' (second player) and '.', row 0 first."""
    board = np.array([[{"X": 1, "O": -1, ".": 0}[ch] for ch in row] for row in rows], dtype=np.int8)
    return reference.OthelloState(board=board, player=player)


def test_initial_moves():
    state = game.initial_state()
    assert game.terminal_value(state) is None
    assert sorted(np.flatnonzero(game.legal_mask(state))) == [19, 26, 37, 44]


def test_perft():
    state = game.initial_state()
    for depth, expected in enumerate(PERFT):
        assert perft(state, depth) == expected


def test_move_flips_in_every_direction():
    state = position(
        [
            "X..X..X.",
            ".O.O.O..",
            "..OOO...",
            "XOO.OOOX",
            "..OOO...",
            ".O.O.O..",
            "X..X..X.",
            "........",
        ],
        player=1,
    )
    after = game.step(state, 3 * 8 + 3)
    assert after.player == -1
    assert (after.board == -1).sum() == 0
    assert (after.board == 1).sum() == 8 + 17 + 1


def test_line_without_own_disc_at_the_end_is_not_flipped():
    state = position(["XO.OO..."] + ["........"] * 7, player=1)
    after = game.step(state, 2)
    np.testing.assert_array_equal(after.board[0], [1, 1, 1, -1, -1, 0, 0, 0])


def test_pass_is_the_only_move_when_blocked():
    # X has no placement, O can still play at (0, 2).
    state = position(["OX......"] + ["........"] * 7, player=1)
    mask = game.legal_mask(state)
    assert list(np.flatnonzero(mask)) == [PASS]
    after = game.step(state, PASS)
    assert after.player == -1
    np.testing.assert_array_equal(after.board, state.board)
    assert list(np.flatnonzero(game.legal_mask(after))) == [2]


def test_pass_is_illegal_when_a_placement_exists():
    with pytest.raises(ValueError):
        game.step(game.initial_state(), PASS)


@pytest.mark.parametrize(
    ("rows", "player", "expected"),
    [
        (["XX......"] + ["........"] * 6 + [".......O"], 1, 1.0),
        (["XX......"] + ["........"] * 6 + [".......O"], -1, -1.0),
        (["X......."] + ["........"] * 6 + [".......O"], 1, 0.0),
        (["XXXXXXXX"] * 4 + ["OOOOOOOO"] * 4, -1, 0.0),
    ],
)
def test_game_ends_when_neither_player_can_move(rows, player, expected):
    state = position(rows, player)
    assert game.terminal_value(state) == expected
    assert not game.legal_mask(state).any()


def test_random_games_end_with_a_consistent_result():
    rng = np.random.default_rng(0)
    for _ in range(5):
        state = game.initial_state()
        plies = 0
        while game.terminal_value(state) is None:
            state = game.step(state, int(rng.choice(np.flatnonzero(game.legal_mask(state)))))
            plies += 1
        assert 9 <= plies <= 120
        assert game.terminal_value(state) == np.sign(state.board.sum() * state.player)
