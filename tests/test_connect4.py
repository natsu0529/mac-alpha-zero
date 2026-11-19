import numpy as np
import pytest

from mac_alpha_zero.reference import Connect4, Connect4State

game = Connect4()


def test_initial_state():
    state = game.initial_state()
    assert state.player == 1
    assert game.legal_mask(state).all()
    assert game.terminal_value(state) is None


def test_pieces_stack_and_players_alternate():
    state = game.from_moves("444")
    assert state.board[:3, 3].tolist() == [1, -1, 1]
    assert state.player == -1


def test_full_column_is_illegal():
    state = game.from_moves("444444")
    assert not game.legal_mask(state)[3]
    with pytest.raises(ValueError):
        game.step(state, 3)


@pytest.mark.parametrize(
    "moves",
    [
        "1213141",  # horizontal
        "1212121",  # vertical
        "12234334544",  # diagonal up-right
        "76654554344",  # diagonal up-left
    ],
)
def test_first_player_wins(moves):
    state = game.from_moves(moves)
    assert state.winner == 1
    assert game.terminal_value(state) == -1.0
    assert not game.legal_mask(state).any()


def test_win_one_move_earlier_is_not_terminal():
    state = game.from_moves("121314")
    assert state.winner == 0
    assert game.terminal_value(state) is None


def test_full_board_without_four_is_a_draw():
    a = [1, 1, -1, -1, 1, 1, -1]
    b = [-1, -1, 1, 1, -1, -1, 1]
    board = np.array([a, a, b, a, b, b], dtype=np.int8)
    state = Connect4State(board=board, player=1)
    assert game.terminal_value(state) == 0.0
