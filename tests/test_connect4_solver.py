import json
from pathlib import Path

import numpy as np

from mac_alpha_zero import reference
from mac_alpha_zero.games.connect4_solver import move_values

from .test_batched_mcts import random_positions

ref = reference.Connect4()
POSITIONS = Path(__file__).parent.parent / "benchmarks" / "data" / "connect4_positions.json"


def test_agrees_with_reference_solver_on_proven_positions():
    checked = 0
    for state in random_positions(40, seed=4, min_moves=26):
        value, best = reference.solve(ref, state, depth=5)
        if value == 0.0:
            continue  # the depth-limited reference could not prove a result
        values = move_values(state.board, state.player)
        assert max(values.values()) == value
        assert set(best) <= {a for a, v in values.items() if v == value}
        checked += 1
    assert checked >= 10


def test_legal_moves_and_immediate_wins():
    # Early positions take far too long to solve, so use late ones.
    for state in random_positions(30, seed=6, min_moves=28):
        values = move_values(state.board, state.player)
        assert sorted(values) == np.flatnonzero(ref.legal_mask(state)).tolist()
        for action in values:
            if ref.step(state, action).winner:
                assert values[action] == 1


def test_stored_positions_match_the_solver():
    entries = json.loads(POSITIONS.read_text())
    assert len(entries) == 1000
    # Late positions solve in milliseconds; early ones are too slow to recheck here.
    late = [e for e in entries if len(e["moves"]) >= 24][:50]
    for entry in late:
        state = ref.from_moves(entry["moves"])
        assert ref.terminal_value(state) is None
        values = move_values(state.board, state.player)
        assert {str(a): v for a, v in values.items()} == entry["values"]
        assert len(set(values.values())) > 1
    assert np.all([len(e["moves"]) >= 14 for e in entries])
