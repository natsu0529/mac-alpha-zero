"""Batched Gumbel root action selection. Semantics match `reference.gumbel`.

The root uses Gumbel noise with Sequential Halving. Nodes below the root can use PUCT
or the improved-policy rule.
"""

from typing import NamedTuple

import mlx.core as mx
import numpy as np

from mac_alpha_zero.games.base import BatchedGame, State
from mac_alpha_zero.mcts import Evaluator, _child_stats, _new_tree, _root_child_stats, _simulate
from mac_alpha_zero.sequential_halving import considered_visit_sequence

_TINY = 1e-30
_LOW_SCORE = -1e9


class GumbelResult(NamedTuple):
    action: mx.array  # (B,) the action to play
    improved_policy: mx.array  # (B, A) policy training target
    visit_counts: mx.array  # (B, A)
    root_value: mx.array  # (B,) mean root value for the player to move at the root


def gumbel_search(
    game: BatchedGame,
    root: State,
    evaluator: Evaluator,
    num_simulations: int,
    gumbel: mx.array | None = None,
    key: mx.array | None = None,
    max_considered: int = 16,
    c_visit: float = 50.0,
    c_scale: float = 0.1,
    c_puct: float = 1.25,
    dtype: mx.Dtype = mx.float32,
    use_kernels: bool = False,
    improved_nonroot: bool = False,
) -> GumbelResult:
    """Search with `num_simulations` evaluations per tree; the first expands the root.

    `gumbel` (B, A) is the root noise. If omitted it is sampled using `key`.
    """
    tree, rows = _new_tree(game, root, num_simulations, dtype, use_kernels)
    batch = rows.shape[0]
    if gumbel is None:
        gumbel = mx.random.gumbel((batch, game.num_actions), key=key)
    gumbel = gumbel.astype(dtype)

    raw_values = mx.zeros_like(tree.value_sum) if improved_nonroot else None

    def interior_action(node):
        visits, q = _child_stats(tree, rows, tree.children[rows, node])
        prior = tree.prior[rows, node]
        legal = tree.legal[rows, node]
        sigma = _sigma_completed_q(prior, visits, q, raw_values[rows, node], c_visit, c_scale)
        shift = mx.max(mx.where(legal, sigma, -mx.inf), axis=-1, keepdims=True)
        weights = mx.where(legal, mx.maximum(prior, _TINY) * mx.exp(sigma - shift), 0)
        policy = weights / mx.maximum(weights.sum(axis=-1, keepdims=True), _TINY)
        deficit = policy - visits / (1 + visits.sum(axis=-1, keepdims=True))
        return mx.argmax(mx.where(legal, deficit, -mx.inf), axis=-1).astype(mx.int32)

    def simulate(sim, root_action=None):
        value = _simulate(
            game,
            tree,
            rows,
            sim,
            evaluator,
            c_puct,
            dtype,
            use_kernels,
            root_action,
            interior_action if improved_nonroot else None,
        )
        if raw_values is not None:
            raw_values[:, sim] = value
        return value

    raw_value = simulate(0)
    legal = tree.legal[:, 0]
    prior = tree.prior[:, 0]
    logits = mx.log(mx.maximum(prior, _TINY))

    def scores(considered_visit: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        visits, q = _root_child_stats(tree, rows)
        sigma = _sigma_completed_q(prior, visits, q, raw_value, c_visit, c_scale)
        score = mx.maximum(_LOW_SCORE, gumbel + logits + sigma)
        allowed = legal & (visits == considered_visit[:, None])
        return mx.where(allowed, score, -mx.inf), sigma, visits

    # Row m is the schedule for trees with m considered actions.
    table = mx.array(
        np.array(
            [considered_visit_sequence(m, num_simulations - 1) for m in range(max_considered + 1)],
            dtype=np.int32,
        ).reshape(max_considered + 1, num_simulations - 1)
    )
    num_considered = mx.minimum(max_considered, legal.sum(axis=-1))
    for sim in range(1, num_simulations):
        score, _, _ = scores(table[num_considered, sim - 1])
        simulate(sim, mx.argmax(score, axis=-1).astype(mx.int32))

    visits, _ = _root_child_stats(tree, rows)
    score, sigma, _ = scores(visits.max(axis=-1))
    improved_policy = mx.softmax(mx.where(legal, logits + sigma, -mx.inf), axis=-1)
    return GumbelResult(
        action=mx.argmax(score, axis=-1).astype(mx.int32),
        improved_policy=improved_policy,
        visit_counts=visits.astype(mx.int32),
        root_value=-tree.value_sum[:, 0] / tree.visits[:, 0],
    )


def _sigma_completed_q(
    prior: mx.array,
    visits: mx.array,
    q: mx.array,
    raw_value: mx.array,
    c_visit: float,
    c_scale: float,
) -> mx.array:
    # Same operation order as reference.gumbel.sigma_completed_q.
    visited = visits > 0
    sum_visits = visits.sum(axis=-1)
    sum_probs = mx.where(visited, prior, 0).sum(axis=-1)
    weighted_q = mx.where(visited, prior * q, 0).sum(axis=-1) / mx.where(
        sum_probs > 0, sum_probs, 1
    )
    mixed_value = (raw_value + sum_visits * weighted_q) / (sum_visits + 1)
    completed = mx.where(visited, q, mixed_value[:, None])
    low = completed.min(axis=-1, keepdims=True)
    high = completed.max(axis=-1, keepdims=True)
    normalized = (completed - low) / mx.maximum(high - low, 1e-8)
    return (c_visit + visits.max(axis=-1, keepdims=True)) * c_scale * normalized
