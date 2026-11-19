from dataclasses import dataclass

import numpy as np

ROWS = 6
COLS = 7
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass(frozen=True, eq=False)
class Connect4State:
    # board[row, col], row 0 is the bottom. +1 is the first player, -1 the second.
    board: np.ndarray
    player: int
    # The player who just completed four in a row, or 0.
    winner: int = 0


class Connect4:
    num_actions = COLS

    def initial_state(self) -> Connect4State:
        return Connect4State(board=np.zeros((ROWS, COLS), dtype=np.int8), player=1)

    def legal_mask(self, state: Connect4State) -> np.ndarray:
        if state.winner != 0:
            return np.zeros(COLS, dtype=bool)
        return state.board[ROWS - 1] == 0

    def step(self, state: Connect4State, action: int) -> Connect4State:
        if not self.legal_mask(state)[action]:
            raise ValueError(f"illegal action {action}")
        row = int(np.count_nonzero(state.board[:, action]))
        board = state.board.copy()
        board[row, action] = state.player
        winner = state.player if _makes_four(board, row, action) else 0
        return Connect4State(board=board, player=-state.player, winner=winner)

    def terminal_value(self, state: Connect4State) -> float | None:
        if state.winner != 0:
            # Only the player who just moved can have won, so the player to move lost.
            return -1.0
        if np.all(state.board[ROWS - 1] != 0):
            return 0.0
        return None

    def from_moves(self, moves: str) -> Connect4State:
        """Build a position from 1-indexed column digits, e.g. "4453"."""
        state = self.initial_state()
        for ch in moves:
            state = self.step(state, int(ch) - 1)
        return state


def _makes_four(board: np.ndarray, row: int, col: int) -> bool:
    color = board[row, col]
    for dr, dc in _DIRECTIONS:
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == color:
                count += 1
                r, c = r + sign * dr, c + sign * dc
        if count >= 4:
            return True
    return False
