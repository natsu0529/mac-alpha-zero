"""Exact win/draw/loss solver for Connect4 on Python-int bitboards.

Used to grade moves, not to play: it only answers who wins with perfect play, not how
fast. Fast enough for positions with roughly 16 or more stones on the board.

Bit `col * 7 + row` holds the cell at (row, col); the 7th bit of each column stays empty
so that shifts never wrap into the next column.
"""

import numpy as np

ROWS = 6
COLS = 7
_HEIGHT = ROWS + 1
_COLUMN_ORDER = (3, 2, 4, 1, 5, 0, 6)
_BOTTOM = [1 << (col * _HEIGHT) for col in range(COLS)]
_TOP = [1 << (col * _HEIGHT + ROWS - 1) for col in range(COLS)]
_LOWER, _EXACT, _UPPER = 0, 1, 2


def move_values(board: np.ndarray, player: int) -> dict[int, int]:
    """Game-theoretic value of every legal move for `player` to move: +1, 0 or -1.

    `board` is (ROWS, COLS) with row 0 at the bottom, +1 / -1 stones, and no four in a
    row yet.
    """
    current, mask = _to_bitboards(board, player)
    table: dict[tuple[int, int], tuple[int, int]] = {}
    values = {}
    for col in range(COLS):
        if mask & _TOP[col]:
            continue
        move = (mask + _BOTTOM[col]) & ~mask
        if _has_four(current | move):
            values[col] = 1
        else:
            values[col] = -_negamax(mask & ~current, mask | move, -1, 1, table)
    return values


def _negamax(current: int, mask: int, alpha: int, beta: int, table: dict) -> int:
    """Value for the player to move, who owns `current`. The opponent has no four."""
    moves = []
    for col in _COLUMN_ORDER:
        if mask & _TOP[col]:
            continue
        move = (mask + _BOTTOM[col]) & ~mask
        if _has_four(current | move):
            return 1
        moves.append(move)
    if not moves:
        return 0

    key = (current, mask)
    entry = table.get(key)
    if entry is not None:
        value, flag = entry
        if flag == _EXACT:
            return value
        if flag == _LOWER:
            alpha = max(alpha, value)
        else:
            beta = min(beta, value)
        if alpha >= beta:
            return value

    original_alpha = alpha
    best = -1
    opponent = mask & ~current
    for move in moves:
        value = -_negamax(opponent, mask | move, -beta, -alpha, table)
        if value > best:
            best = value
            alpha = max(alpha, value)
            if alpha >= beta:
                break

    if best <= original_alpha:
        flag = _UPPER
    elif best >= beta:
        flag = _LOWER
    else:
        flag = _EXACT
    table[key] = (best, flag)
    return best


def _has_four(stones: int) -> bool:
    # Shifts: 1 vertical, 7 horizontal, 6 and 8 the two diagonals.
    for shift in (1, _HEIGHT, _HEIGHT - 1, _HEIGHT + 1):
        pairs = stones & (stones >> shift)
        if pairs & (pairs >> (2 * shift)):
            return True
    return False


def _to_bitboards(board: np.ndarray, player: int) -> tuple[int, int]:
    current = 0
    mask = 0
    for row, col in zip(*np.nonzero(board), strict=True):
        bit = 1 << (int(col) * _HEIGHT + int(row))
        mask |= bit
        if board[row, col] == player:
            current |= bit
    return current, mask
