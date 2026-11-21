import mlx.core as mx
import numpy as np

from mac_alpha_zero.games.base import State, Symmetries

ROWS = 6
COLS = 7


class Connect4:
    """Batched Connect4 with the same conventions as `reference.Connect4`.

    State keys: board (B, ROWS, COLS) int8 with row 0 at the bottom, +1 for the first
    player and -1 for the second; player (B,) int8, the player to move; winner (B,) int8,
    the player who just completed four in a row, or 0.
    """

    num_actions = COLS
    observation_shape = (ROWS, COLS, 2)

    def init(self, batch_size: int) -> State:
        return {
            "board": mx.zeros((batch_size, ROWS, COLS), dtype=mx.int8),
            "player": mx.ones((batch_size,), dtype=mx.int8),
            "winner": mx.zeros((batch_size,), dtype=mx.int8),
        }

    def observe(self, state: State) -> mx.array:
        """Planes: the player to move's stones, then the opponent's."""
        relative = state["board"] * state["player"][:, None, None]
        return mx.stack([relative == 1, relative == -1], axis=-1).astype(mx.float32)

    def legal_mask(self, state: State) -> mx.array:
        """(B, COLS) bool. Empty for finished games."""
        top_empty = state["board"][:, ROWS - 1, :] == 0
        return top_empty & (state["winner"] == 0)[:, None]

    def step(self, state: State, actions: mx.array) -> State:
        """Play `actions` (B,) in every game. A full column leaves the board unchanged."""
        board, player = state["board"], state["player"]
        heights = (board != 0).sum(axis=1)
        row = mx.take_along_axis(heights, actions[:, None], axis=1)
        rows = mx.arange(ROWS)[None, :, None]
        cols = mx.arange(COLS)[None, None, :]
        placed = (rows == row[:, :, None]) & (cols == actions[:, None, None])
        board = mx.where(placed, player[:, None, None], board)
        # No four existed before this move, so any four of the mover's colour is new.
        wins = _has_four(board == player[:, None, None])
        winner = mx.where(wins, player, mx.zeros_like(player))
        return {"board": board, "player": -player, "winner": winner}

    def terminal(self, state: State) -> tuple[mx.array, mx.array]:
        """(done, value) per game. Value is for the player to move and 0 if not done."""
        won = state["winner"] != 0
        full = mx.all(state["board"][:, ROWS - 1, :] != 0, axis=1)
        value = mx.where(won, -1.0, 0.0)
        return won | full, value

    def symmetries(self) -> Symmetries:
        """The identity and the left-right mirror."""
        squares = np.arange(ROWS * COLS).reshape(ROWS, COLS)
        squares = np.stack([squares.reshape(-1), squares[:, ::-1].reshape(-1)])
        return squares, np.stack([np.arange(COLS), np.arange(COLS)[::-1]])

    def from_reference(self, states) -> State:
        """Stack `reference.Connect4State`s into a batched state."""
        return {
            "board": mx.array(np.stack([s.board for s in states]).astype(np.int8)),
            "player": mx.array(np.array([s.player for s in states], dtype=np.int8)),
            "winner": mx.array(np.array([s.winner for s in states], dtype=np.int8)),
        }


def _has_four(stones: mx.array) -> mx.array:
    """(B, ROWS, COLS) bool -> (B,) whether any four in a row exists."""
    s = stones
    horizontal = s[:, :, :-3] & s[:, :, 1:-2] & s[:, :, 2:-1] & s[:, :, 3:]
    vertical = s[:, :-3, :] & s[:, 1:-2, :] & s[:, 2:-1, :] & s[:, 3:, :]
    up_right = s[:, :-3, :-3] & s[:, 1:-2, 1:-2] & s[:, 2:-1, 2:-1] & s[:, 3:, 3:]
    up_left = s[:, :-3, 3:] & s[:, 1:-2, 2:-1] & s[:, 2:-1, 1:-2] & s[:, 3:, :-3]
    return (
        mx.any(horizontal, axis=(1, 2))
        | mx.any(vertical, axis=(1, 2))
        | mx.any(up_right, axis=(1, 2))
        | mx.any(up_left, axis=(1, 2))
    )
