from typing import Protocol, TypeVar

import numpy as np

S = TypeVar("S")


class Game(Protocol[S]):
    """Two-player, zero-sum, perfect-information game with a fixed action space.

    Values are always from the perspective of the player to move in the given state.
    """

    num_actions: int

    def initial_state(self) -> S: ...

    def legal_mask(self, state: S) -> np.ndarray:
        """Boolean array of shape (num_actions,)."""
        ...

    def step(self, state: S, action: int) -> S:
        """Return the state after the player to move plays `action`."""
        ...

    def terminal_value(self, state: S) -> float | None:
        """None if the game is not over, otherwise +1 / 0 / -1 for the player to move."""
        ...
