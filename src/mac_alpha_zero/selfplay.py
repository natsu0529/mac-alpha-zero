from collections.abc import Callable

import mlx.core as mx
import numpy as np

from mac_alpha_zero.games.base import BatchedGame, State

# Maps a batch of non-terminal states to (actions (B,), policy targets (B, A)).
Actor = Callable[[State], tuple[mx.array, mx.array]]

# (observations (n, ...), policy targets (n, A), outcomes (n,)) of finished games.
Samples = tuple[np.ndarray, np.ndarray, np.ndarray]


class SelfPlay:
    """Plays `batch_size` games in parallel and restarts each one as it finishes.

    Positions are held back until their game ends, then released with the final outcome
    from the perspective of the player to move in that position.
    """

    def __init__(self, game: BatchedGame, batch_size: int):
        self.game = game
        self.batch_size = batch_size
        self.state = game.init(batch_size)
        self.games_finished = 0
        self._pending: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(batch_size)]

    def step(self, actor: Actor) -> Samples:
        """Play one move in every game. Returns the samples of games that just ended."""
        game = self.game
        observations = np.array(game.observe(self.state))
        actions, policies = actor(self.state)
        policies = np.array(policies)
        for i in range(self.batch_size):
            self._pending[i].append((observations[i], policies[i]))

        state = game.step(self.state, actions)
        done, value = game.terminal(state)
        fresh = game.init(self.batch_size)
        self.state = {
            key: mx.where(_expand(done, array.ndim), fresh[key], array)
            for key, array in state.items()
        }
        mx.eval(*self.state.values())

        out_obs, out_policy, out_z = [], [], []
        values = np.array(value)
        for i in np.flatnonzero(np.array(done)):
            # `values[i]` is for the player to move after the last move, so the player who
            # made the last move gets the opposite, and it alternates going backwards.
            outcome = -float(values[i])
            for observation, policy in reversed(self._pending[i]):
                out_obs.append(observation)
                out_policy.append(policy)
                out_z.append(outcome)
                outcome = -outcome
            self._pending[i] = []
            self.games_finished += 1

        if not out_obs:
            empty_obs = np.zeros((0, *observations.shape[1:]), dtype=observations.dtype)
            empty_policy = np.zeros((0, policies.shape[1]), dtype=policies.dtype)
            return empty_obs, empty_policy, np.zeros((0,), dtype=np.float32)
        return np.stack(out_obs), np.stack(out_policy), np.array(out_z, dtype=np.float32)


def _expand(mask: mx.array, ndim: int) -> mx.array:
    return mask.reshape(-1, *([1] * (ndim - 1)))
