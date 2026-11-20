"""Play Othello against Edax (https://github.com/abulmo/edax-reversi).

Edax is a separate GPL-3.0 program and is not part of this package. Build it with
`scripts/build_edax.sh`; it is only ever run as a subprocess.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from mac_alpha_zero.arena import Agent
from mac_alpha_zero.games.base import State
from mac_alpha_zero.games.othello import PASS, SIZE, Othello

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "third_party" / "edax" / "edax"

# A result row of `edax -solve`: "  12|   3   -04   0:00.000   14   d3 C4 ...".
_RESULT_ROW = re.compile(r"^\s*(\d+)\|(.*)$")
_SQUARE = re.compile(r"\b([a-hA-H])([1-8])\b")


def to_obf(board: np.ndarray, player: int) -> str:
    """One position in Edax's OBF format. `board` is (8, 8) with +1 black and -1 white."""
    symbols = {1: "X", -1: "O", 0: "-"}
    squares = "".join(symbols[int(v)] for v in board.reshape(-1))
    return f"{squares} {symbols[player]};"


def parse_moves(output: str, count: int) -> list[int]:
    """The move Edax chose for each of `count` positions, as squares `row * 8 + col`."""
    moves: dict[int, int] = {}
    for line in output.splitlines():
        row = _RESULT_ROW.match(line)
        if row is None:
            continue
        square = _SQUARE.search(row.group(2))
        if square is None:
            raise ValueError(f"no move in Edax output line: {line!r}")
        column = ord(square.group(1).lower()) - ord("a")
        moves[int(row.group(1))] = (int(square.group(2)) - 1) * SIZE + column
    if sorted(moves) != list(range(1, count + 1)):
        raise ValueError(f"expected {count} moves from Edax, got {len(moves)}:\n{output}")
    return [moves[i] for i in range(1, count + 1)]


def edax_agent(level: int, path: Path | str = DEFAULT_PATH, threads: int = 1) -> Agent:
    """Agent that plays Edax's move at search `level`, without its opening book.

    Each call runs Edax once on all positions of the batch. Edax is deterministic, so
    vary the starting positions to get different games.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build Edax with scripts/build_edax.sh")
    game = Othello()

    def act(state: State) -> mx.array:
        boards = np.array(game.boards(state))
        players = np.array(state["player"])
        legal = np.array(game.legal_mask(state))
        actions = np.full(len(boards), PASS, dtype=np.int32)
        # Edax is only asked where there is a real choice to make.
        asked = np.flatnonzero(~legal[:, PASS])
        if len(asked):
            with tempfile.TemporaryDirectory() as directory:
                problems = Path(directory) / "positions.obf"
                problems.write_text(
                    "".join(to_obf(boards[i], int(players[i])) + "\n" for i in asked)
                )
                command = [str(path), "-solve", str(problems), "-l", str(level), "-n", str(threads)]
                done = subprocess.run(
                    command, cwd=path.parent, capture_output=True, text=True, check=True
                )
            moves = parse_moves(done.stdout, len(asked))
            for i, move in zip(asked, moves, strict=True):
                if not legal[i, move]:
                    position = to_obf(boards[i], int(players[i]))
                    raise ValueError(f"Edax played illegal move {move} in {position}")
                actions[i] = move
        return mx.array(actions)

    return act
