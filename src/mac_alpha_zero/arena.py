from collections.abc import Callable
from typing import NamedTuple

import mlx.core as mx
import numpy as np

from mac_alpha_zero import kernels
from mac_alpha_zero.games.base import BatchedGame, State
from mac_alpha_zero.gumbel import gumbel_search
from mac_alpha_zero.mcts import Evaluator, search

# Maps a batch of non-terminal states to actions (B,).
Agent = Callable[[State], mx.array]


class MatchResult(NamedTuple):
    wins: int
    draws: int
    losses: int

    @property
    def score(self) -> float:
        """Mean result in [0, 1]: a win counts 1 and a draw 0.5."""
        return (self.wins + 0.5 * self.draws) / (self.wins + self.draws + self.losses)


def play_match(game: BatchedGame, first: Agent, second: Agent, num_games: int) -> np.ndarray:
    """Play `num_games` games in parallel. Returns +1 / 0 / -1 per game for `first`."""
    state = game.init(num_games)
    fresh = game.init(num_games)
    results = np.zeros(num_games)
    finished = np.zeros(num_games, dtype=bool)
    ply = 0
    while not finished.all():
        mask = mx.array(finished)
        # Agents cannot act on finished games, so those slots see a fresh board instead.
        shown = {k: mx.where(_expand(mask, v.ndim), fresh[k], v) for k, v in state.items()}
        actions = (first if ply % 2 == 0 else second)(shown)
        stepped = game.step(state, actions)
        state = {k: mx.where(_expand(mask, v.ndim), state[k], v) for k, v in stepped.items()}
        done, value = game.terminal(state)
        mx.eval(*state.values())
        # `value` is for the player to move next, i.e. the opponent of whoever just moved.
        mover_sign = 1.0 if ply % 2 == 0 else -1.0
        just_finished = np.array(done) & ~finished
        results[just_finished] = -mover_sign * np.array(value)[just_finished]
        finished |= just_finished
        ply += 1
    return results


def evaluate(game: BatchedGame, agent: Agent, opponent: Agent, num_games: int) -> MatchResult:
    """`agent` against `opponent`, half the games with each side moving first."""
    half = num_games // 2
    results = np.concatenate(
        [play_match(game, agent, opponent, half), -play_match(game, opponent, agent, half)]
    )
    return MatchResult(
        wins=int((results > 0).sum()),
        draws=int((results == 0).sum()),
        losses=int((results < 0).sum()),
    )


def random_agent(game: BatchedGame, seed: int = 0) -> Agent:
    key = mx.random.key(seed)

    def act(state: State) -> mx.array:
        nonlocal key
        key, subkey = mx.random.split(key)
        logits = mx.where(game.legal_mask(state), 0.0, -mx.inf)
        return mx.random.categorical(logits, key=subkey).astype(mx.int32)

    return act


def puct_agent(game: BatchedGame, evaluator: Evaluator, num_simulations: int) -> Agent:
    """Plays the most visited action of a PUCT search."""

    def act(state: State) -> mx.array:
        result = search(game, state, evaluator, num_simulations, use_kernels=kernels.is_available())
        return mx.argmax(result.visit_counts, axis=-1).astype(mx.int32)

    return act


def gumbel_agent(game: BatchedGame, evaluator: Evaluator, num_simulations: int) -> Agent:
    """Plays the Gumbel search action without noise, i.e. deterministically."""

    def act(state: State) -> mx.array:
        batch = next(iter(state.values())).shape[0]
        result = gumbel_search(
            game,
            state,
            evaluator,
            num_simulations,
            gumbel=mx.zeros((batch, game.num_actions)),
            use_kernels=kernels.is_available(),
        )
        return result.action

    return act


def _expand(mask: mx.array, ndim: int) -> mx.array:
    return mask.reshape(-1, *([1] * (ndim - 1)))
