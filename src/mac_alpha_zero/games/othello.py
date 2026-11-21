import mlx.core as mx
import numpy as np

from mac_alpha_zero.games.base import State, Symmetries

SIZE = 8
PASS = SIZE * SIZE


def _u64(value: int) -> mx.array:
    # Built through NumPy: MLX cannot make a uint64 array from a large Python int.
    return mx.array(np.array(value, dtype=np.uint64))


_ZERO = _u64(0)
_ONE = _u64(1)
_SQUARES = mx.array(np.arange(PASS, dtype=np.uint64))
_ALL = 0xFFFFFFFFFFFFFFFF
_INNER_COLUMNS = 0x7E7E7E7E7E7E7E7E
# (shift towards higher bits?, distance, squares a line may pass through). Lines that
# change column may not pass through the edge columns, which stops them wrapping around.
_DIRECTIONS = [
    (left, _u64(distance), _u64(passable))
    for distance, passable in (
        (1, _INNER_COLUMNS),
        (7, _INNER_COLUMNS),
        (9, _INNER_COLUMNS),
        (8, _ALL),
    )
    for left in (True, False)
]


class Othello:
    """Batched Othello on bitboards, with the same rules as `reference.Othello`.

    State keys: own and opp (B,) uint64, the discs of the player to move and of the
    opponent, with bit `row * 8 + col` for each square; player (B,) int8, +1 if the first
    player (black) is to move and -1 otherwise.
    """

    num_actions = PASS + 1
    observation_shape = (SIZE, SIZE, 2)

    def init(self, batch_size: int) -> State:
        black = (1 << (3 * SIZE + 4)) | (1 << (4 * SIZE + 3))
        white = (1 << (3 * SIZE + 3)) | (1 << (4 * SIZE + 4))
        return {
            "own": mx.broadcast_to(_u64(black), (batch_size,)),
            "opp": mx.broadcast_to(_u64(white), (batch_size,)),
            "player": mx.ones((batch_size,), dtype=mx.int8),
        }

    def observe(self, state: State) -> mx.array:
        """Planes: the player to move's discs, then the opponent's."""
        planes = mx.stack([_bits(state["own"]), _bits(state["opp"])], axis=-1)
        return planes.reshape(-1, SIZE, SIZE, 2).astype(mx.float32)

    def legal_mask(self, state: State) -> mx.array:
        """(B, 65) bool. The pass is legal only if it is the only move. Empty when done."""
        own, opp = state["own"], state["opp"]
        placements = _placements(own, opp)
        can_pass = (placements == _ZERO) & (_placements(opp, own) != _ZERO)
        return mx.concatenate([_bits(placements), can_pass[:, None]], axis=1)

    def step(self, state: State, actions: mx.array) -> State:
        """Play `actions` (B,) in every game. An illegal action gives a meaningless state."""
        own, opp = state["own"], state["opp"]
        is_pass = actions >= PASS
        square = mx.where(is_pass, 0, actions).astype(mx.uint64)
        disc = mx.where(is_pass, _ZERO, mx.left_shift(_ONE, square))
        flips = _flips(own, opp, disc)
        return {
            "own": opp & ~flips,
            "opp": own | disc | flips,
            "player": -state["player"],
        }

    def terminal(self, state: State) -> tuple[mx.array, mx.array]:
        """(done, value) per game. Value is for the player to move and 0 if not done."""
        own, opp = state["own"], state["opp"]
        done = (_placements(own, opp) == _ZERO) & (_placements(opp, own) == _ZERO)
        margin = _bits(own).sum(axis=1) - _bits(opp).sum(axis=1)
        return done, mx.where(done, mx.sign(margin), 0).astype(mx.float32)

    def boards(self, state: State) -> mx.array:
        """(B, 8, 8) int8 boards as in `reference.OthelloState`: +1 black, -1 white."""
        relative = _bits(state["own"]).astype(mx.int8) - _bits(state["opp"]).astype(mx.int8)
        return relative.reshape(-1, SIZE, SIZE) * state["player"][:, None, None]

    def symmetries(self) -> Symmetries:
        """The 8 rotations and reflections of the board, as `reference.Othello` numbers
        them. The pass (action 64) maps to itself."""
        squares = np.arange(PASS).reshape(SIZE, SIZE)
        rotations = [np.rot90(squares, k) for k in range(4)]
        squares = np.stack([board.reshape(PASS) for board in rotations + [r.T for r in rotations]])
        passes = np.full((len(squares), 1), PASS)
        return squares, np.concatenate([squares, passes], axis=1)

    def from_reference(self, states) -> State:
        """Stack `reference.OthelloState`s into a batched state."""
        weights = np.uint64(1) << np.arange(PASS, dtype=np.uint64)

        def pack(discs: list[np.ndarray]) -> mx.array:
            flat = np.stack(discs).reshape(len(discs), PASS)
            return mx.array(np.where(flat, weights, np.uint64(0)).sum(axis=1, dtype=np.uint64))

        return {
            "own": pack([s.board == s.player for s in states]),
            "opp": pack([s.board == -s.player for s in states]),
            "player": mx.array(np.array([s.player for s in states], dtype=np.int8)),
        }


def _shift(bits: mx.array, left: bool, distance: mx.array) -> mx.array:
    return mx.left_shift(bits, distance) if left else mx.right_shift(bits, distance)


def _line(start: mx.array, opp: mx.array, left: bool, distance: mx.array, passable: mx.array):
    """Unbroken run of opponent discs next to `start` in one direction (at most 6)."""
    through = opp & passable
    run = through & _shift(start, left, distance)
    for _ in range(SIZE - 3):
        run = run | (through & _shift(run, left, distance))
    return run


def _placements(own: mx.array, opp: mx.array) -> mx.array:
    """(B,) uint64: empty squares where the owner of `own` would flip at least one disc."""
    empty = ~(own | opp)
    result = mx.zeros_like(own)
    for left, distance, passable in _DIRECTIONS:
        run = _line(own, opp, left, distance, passable)
        result = result | (_shift(run, left, distance) & empty)
    return result


def _flips(own: mx.array, opp: mx.array, disc: mx.array) -> mx.array:
    """(B,) uint64: opponent discs flipped when the owner of `own` places `disc`."""
    result = mx.zeros_like(own)
    for left, distance, passable in _DIRECTIONS:
        run = _line(disc, opp, left, distance, passable)
        closed = (_shift(run, left, distance) & own) != _ZERO
        result = result | mx.where(closed, run, _ZERO)
    return result


def _bits(bitboard: mx.array) -> mx.array:
    """(B,) uint64 -> (B, 64) bool, one entry per square."""
    return (mx.right_shift(bitboard[:, None], _SQUARES) & _ONE) != _ZERO
