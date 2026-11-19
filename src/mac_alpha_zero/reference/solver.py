import numpy as np

from mac_alpha_zero.reference.game import Game


def solve(game: Game, state, depth: int) -> tuple[float, list[int]]:
    """Depth-limited negamax with alpha-beta.

    Returns the value for the player to move and the actions that achieve it.
    Positions not resolved within `depth` plies count as 0, so a value of +1 or -1
    is a proven result while 0 means "draw or unknown".
    """
    legal = np.flatnonzero(game.legal_mask(state))
    values = {
        int(a): -_negamax(game, game.step(state, int(a)), depth - 1, -1.0, 1.0) for a in legal
    }
    best = max(values.values())
    return best, [a for a, v in values.items() if v == best]


def _negamax(game: Game, state, depth: int, alpha: float, beta: float) -> float:
    terminal = game.terminal_value(state)
    if terminal is not None:
        return terminal
    if depth == 0:
        return 0.0
    best = -1.0
    for a in _center_first(game.legal_mask(state)):
        value = -_negamax(game, game.step(state, a), depth - 1, -beta, -alpha)
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return best


def _center_first(mask: np.ndarray) -> list[int]:
    center = (len(mask) - 1) / 2
    return sorted(np.flatnonzero(mask).tolist(), key=lambda a: abs(a - center))
