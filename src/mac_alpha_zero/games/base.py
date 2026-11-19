from typing import Protocol

import mlx.core as mx

# A batched state is a dict of arrays that all share a leading batch dimension.
State = dict[str, mx.array]


class BatchedGame(Protocol):
    """Batched version of `reference.Game`. Values are for the player to move.

    `step` may be called with illegal actions or on finished games; callers discard
    those results, so it must not raise.
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
