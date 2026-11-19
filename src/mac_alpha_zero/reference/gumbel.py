"""Gumbel root action selection (Danihelka et al., 2022) on a single tree.

The root uses Gumbel noise with Sequential Halving and returns the improved policy from
completed Q-values. Nodes below the root still use PUCT; the paper's deterministic
non-root selection is not implemented yet.
"""

from dataclasses import dataclass

import numpy as np

from mac_alpha_zero.reference.game import Game
from mac_alpha_zero.reference.mcts import Evaluator, _Node, _root_visits, _simulate
from mac_alpha_zero.sequential_halving import considered_visit_sequence

_TINY = 1e-30
_LOW_SCORE = -1e9


@dataclass
class GumbelResult:
    # The action to play: best of the most-visited actions by gumbel + logit + sigma(q).
    action: int
    # Training target for the policy: softmax(logits + sigma(completed q)).
    improved_policy: np.ndarray
    visit_counts: np.ndarray
    root_value: float


def run_gumbel_mcts(
    game: Game,
    state,
    evaluator: Evaluator,
    num_simulations: int,
    gumbel: np.ndarray,
    max_considered: int = 16,
    c_visit: float = 50.0,
    c_scale: float = 0.1,
    c_puct: float = 1.25,
) -> GumbelResult:
    """Search with `num_simulations` evaluations; the first one expands the root.

    `gumbel` is one Gumbel(0, 1) sample per action. Passing it in keeps the search
    deterministic and lets the batched implementation be compared on the same noise.
    """
    if game.terminal_value(state) is not None:
        raise ValueError("cannot search from a terminal state")
    root = _Node(prior=1.0, state=state)
    _simulate(game, root, evaluator, c_puct)
    raw_value = -root.value_sum

    legal = game.legal_mask(state)
    prior = np.zeros(game.num_actions)
    for action, child in root.children.items():
        prior[action] = child.prior
    logits = np.log(np.maximum(prior, _TINY))

    def scores(considered_visit: int) -> tuple[np.ndarray, np.ndarray]:
        visits = _root_visits(game, root)
        q = np.array([root.children[a].q if legal[a] else 0.0 for a in range(len(legal))])
        sigma = sigma_completed_q(prior, visits, q, raw_value, c_visit, c_scale)
        score = np.maximum(_LOW_SCORE, gumbel + logits + sigma)
        return np.where(legal & (visits == considered_visit), score, -np.inf), sigma

    num_considered = min(max_considered, int(legal.sum()))
    for considered_visit in considered_visit_sequence(num_considered, num_simulations - 1):
        score, _ = scores(considered_visit)
        _simulate(game, root, evaluator, c_puct, root_action=int(np.argmax(score)))

    visits = _root_visits(game, root)
    score, sigma = scores(int(visits.max()))
    policy_logits = np.where(legal, logits + sigma, -np.inf)
    policy = np.exp(policy_logits - policy_logits.max())
    return GumbelResult(
        action=int(np.argmax(score)),
        improved_policy=policy / policy.sum(),
        visit_counts=visits,
        root_value=-root.q,
    )


def sigma_completed_q(
    prior: np.ndarray,
    visits: np.ndarray,
    q: np.ndarray,
    raw_value: float,
    c_visit: float,
    c_scale: float,
) -> np.ndarray:
    """sigma(completed q): unvisited actions get the mixed value, then min-max rescale."""
    visited = visits > 0
    sum_visits = visits.sum()
    sum_probs = np.where(visited, prior, 0.0).sum()
    weighted_q = np.where(visited, prior * q, 0.0).sum() / (sum_probs if sum_probs > 0 else 1.0)
    mixed_value = (raw_value + sum_visits * weighted_q) / (sum_visits + 1)
    completed = np.where(visited, q, mixed_value)
    low, high = completed.min(), completed.max()
    normalized = (completed - low) / max(high - low, 1e-8)
    return (c_visit + visits.max()) * c_scale * normalized
