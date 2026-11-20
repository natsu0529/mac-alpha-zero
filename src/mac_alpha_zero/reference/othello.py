from dataclasses import dataclass

import numpy as np

SIZE = 8
PASS = SIZE * SIZE
_DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass(frozen=True, eq=False)
class OthelloState:
    # board[row, col]. +1 is the first player (black), -1 the second (white).
    board: np.ndarray
    player: int


class Othello:
    """Othello with an explicit pass action, so the players always alternate.

    Actions 0..63 place a disc on square `row * 8 + col`; action 64 is the pass, legal
    only when the player to move has no placement. The game ends when neither player
    has a placement, and the player with more discs wins.
    """

    num_actions = SIZE * SIZE + 1

    def initial_state(self) -> OthelloState:
        board = np.zeros((SIZE, SIZE), dtype=np.int8)
        board[3, 3] = board[4, 4] = -1
        board[3, 4] = board[4, 3] = 1
        return OthelloState(board=board, player=1)

    def legal_mask(self, state: OthelloState) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=bool)
        mask[:PASS] = _placements(state.board, state.player)
        if not mask.any() and _placements(state.board, -state.player).any():
            mask[PASS] = True
        return mask

    def step(self, state: OthelloState, action: int) -> OthelloState:
        if not self.legal_mask(state)[action]:
            raise ValueError(f"illegal action {action}")
        board = state.board.copy()
        if action != PASS:
            row, col = divmod(action, SIZE)
            board[row, col] = state.player
            for r, c in _flips(state.board, state.player, row, col):
                board[r, c] = state.player
        return OthelloState(board=board, player=-state.player)

    def terminal_value(self, state: OthelloState) -> float | None:
        if _placements(state.board, 1).any() or _placements(state.board, -1).any():
            return None
        return float(np.sign(int(state.board.sum()) * state.player))


def _flips(board: np.ndarray, player: int, row: int, col: int) -> list[tuple[int, int]]:
    """Opponent discs that `player` would flip by placing on the empty square (row, col)."""
    flipped = []
    for dr, dc in _DIRECTIONS:
        line = []
        r, c = row + dr, col + dc
        while 0 <= r < SIZE and 0 <= c < SIZE and board[r, c] == -player:
            line.append((r, c))
            r, c = r + dr, c + dc
        if line and 0 <= r < SIZE and 0 <= c < SIZE and board[r, c] == player:
            flipped.extend(line)
    return flipped


def _placements(board: np.ndarray, player: int) -> np.ndarray:
    """(64,) bool: empty squares where `player` would flip at least one disc."""
    mask = np.zeros(SIZE * SIZE, dtype=bool)
    for row in range(SIZE):
        for col in range(SIZE):
            if board[row, col] == 0 and _flips(board, player, row, col):
                mask[row * SIZE + col] = True
    return mask
