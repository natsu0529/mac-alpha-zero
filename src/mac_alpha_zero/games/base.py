from typing import Protocol

import mlx.core as mx
import numpy as np

# A batched state is a dict of arrays that all share a leading batch dimension.
State = dict[str, mx.array]

# (squares (S, H * W), actions (S, num_actions)) int arrays, one row per symmetry, the
# identity first. Under symmetry k, square j of the transformed observation is square
# `squares[k, j]` of the original (all planes alike), and action j of the transformed
# policy is action `actions[k, j]` of the original.
Symmetries = tuple[np.ndarray, np.ndarray]


class BatchedGame(Protocol):
    """Batched version of `reference.Game`. Values are for the player to move.

    `step` may be called with illegal actions or on finished games; callers discard
    those results, so it must not raise.

    A game may also define `symmetries()` (see `Symmetries`), the transformations that
    leave its rules unchanged. Training then augments its samples with them.
    """

    num_actions: int
    # (height, width, planes) of one observation.
    observation_shape: tuple[int, int, int]

    def init(self, batch_size: int) -> State:
        """`batch_size` games at the starting position."""
        ...

    def observe(self, state: State) -> mx.array:
        """(B, *observation_shape) float32 network input, from the player to move's view."""
        ...

    def legal_mask(self, state: State) -> mx.array:
        """(B, num_actions) bool."""
        ...

    def step(self, state: State, actions: mx.array) -> State: ...

    def terminal(self, state: State) -> tuple[mx.array, mx.array]:
        """(done, value): (B,) bool and (B,) value for the player to move, 0 if not done."""
        ...
