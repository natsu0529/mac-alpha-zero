import numpy as np
import pytest

from mac_alpha_zero import arena, games, reference
from mac_alpha_zero.edax import DEFAULT_PATH, edax_agent, parse_moves, to_obf

ref = reference.Othello()
game = games.Othello()

OUTPUT = """
 # | depth|score|       time   |  nodes (N)  |   N/s    | principal variation
---+------+-----+--------------+-------------+----------+---------------------
  1|    1   -04        0:00.000            14            d3
WARNING: missing best score in something X;
  2|    4   +04        0:00.000            11            C3 b4 H8
---+------+-----+--------------+-------------+----------+---------------------
positions.obf: 25 nodes in  0:00.000
"""


def test_obf_of_the_starting_position():
    state = ref.initial_state()
    assert to_obf(state.board, state.player) == "-" * 27 + "OX" + "-" * 6 + "XO" + "-" * 27 + " X;"
    assert to_obf(state.board, -1).endswith(" O;")


def test_parse_moves():
    assert parse_moves(OUTPUT, 2) == [2 * 8 + 3, 2 * 8 + 2]


def test_parse_moves_rejects_missing_rows():
    with pytest.raises(ValueError):
        parse_moves(OUTPUT, 3)


def test_start_positions_are_played_from_both_sides():
    start = game.from_reference([ref.step(ref.initial_state(), a) for a in (19, 26, 37, 44)])
    first, second = arena.random_agent(game, 1), arena.random_agent(game, 2)
    results = arena.play_match(game, first, second, 4, start)
    assert set(np.unique(results)) <= {-1.0, 0.0, 1.0}
    result = arena.evaluate(game, first, second, 8, start)
    assert result.wins + result.draws + result.losses == 8


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="needs Edax (scripts/build_edax.sh)")
def test_edax_plays_legal_moves_and_beats_random():
    agent = edax_agent(level=1)
    actions = np.array(agent(game.init(3)))
    assert all(a in (19, 26, 37, 44) for a in actions)

    # A position where the mover must pass is answered without asking Edax.
    board = np.zeros((8, 8), dtype=np.int8)
    board[0, 0], board[0, 1] = -1, 1
    blocked = game.from_reference([reference.OthelloState(board=board, player=1)])
    assert int(agent(blocked)[0]) == 64

    result = arena.evaluate(game, agent, arena.random_agent(game, 0), 20)
    assert result.score > 0.8
