import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from mac_alpha_zero.reference.game import Game

# Maps a non-terminal state to (prior over all actions, value for the player to move).
Evaluator = Callable[[object], tuple[np.ndarray, float]]


@dataclass
class _Node:
    prior: float
    state: object = None
    visit_count: int = 0
    # Sum of values from the perspective of the player who chose the action leading here,
    # so a parent can rank its children by mean value directly.
    value_sum: float = 0.0
    raw_value: float = 0.0
    children: dict[int, "_Node"] = field(default_factory=dict)

    @property
    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass
class MCTSResult:
    visit_counts: np.ndarray
    # Mean value of the root for the player to move at the root.
    root_value: float

    @property
    def action(self) -> int:
        return int(np.argmax(self.visit_counts))


def run_mcts(
    game: Game,
    state,
    evaluator: Evaluator,
    num_simulations: int,
    c_puct: float = 1.25,
) -> MCTSResult:
    """AlphaZero-style PUCT search on a single tree.

    Deterministic for a deterministic evaluator: ties go to the lowest action index.
    """
    if game.terminal_value(state) is not None:
        raise ValueError("cannot search from a terminal state")
    root = _Node(prior=1.0, state=state)
    for _ in range(num_simulations):
        _simulate(game, root, evaluator, c_puct)
    return MCTSResult(visit_counts=_root_visits(game, root), root_value=-root.q)


def _simulate(
    game: Game,
    root: _Node,
    evaluator: Evaluator,
    c_puct: float,
    root_action: int | None = None,
    interior_action: Callable[[_Node], int] | None = None,
) -> None:
    """One simulation. `root_action` overrides PUCT selection at the root."""
    node = root
    path = [node]
    while node.children:
        if node is root and root_action is not None:
            action, child = root_action, node.children[root_action]
        elif interior_action is not None and node is not root:
            action = interior_action(node)
            child = node.children[action]
        else:
            action, child = _select(node, c_puct)
        if child.state is None:
            child.state = game.step(node.state, action)
        node = child
        path.append(node)

    value = game.terminal_value(node.state)
    if value is None:
        priors, value = evaluator(node.state)
        node.raw_value = value
        _expand(node, priors, game.legal_mask(node.state))

    for n in reversed(path):
        n.value_sum -= value
        n.visit_count += 1
        value = -value


def _root_visits(game: Game, root: _Node) -> np.ndarray:
    visits = np.zeros(game.num_actions, dtype=np.int64)
    for action, child in root.children.items():
        visits[action] = child.visit_count
    return visits


def _select(node: _Node, c_puct: float) -> tuple[int, _Node]:
    sqrt_n = math.sqrt(node.visit_count)
    best_score = -math.inf
    best = None
    for action in sorted(node.children):
        child = node.children[action]
        score = child.q + c_puct * child.prior * sqrt_n / (1 + child.visit_count)
        if score > best_score:
            best_score = score
            best = (action, child)
    return best


def _expand(node: _Node, priors: np.ndarray, legal: np.ndarray) -> None:
    masked = np.where(legal, priors, 0.0)
    total = masked.sum()
    masked = masked / total if total > 0 else legal / legal.sum()
    for action in np.flatnonzero(legal):
        node.children[int(action)] = _Node(prior=float(masked[action]))


def random_rollout_evaluator(game: Game, rng: np.random.Generator) -> Evaluator:
    """Uniform priors and the result of one uniformly random playout."""

    def evaluate(state) -> tuple[np.ndarray, float]:
        priors = np.full(game.num_actions, 1.0 / game.num_actions)
        current = state
        sign = 1.0
        while (value := game.terminal_value(current)) is None:
            action = rng.choice(np.flatnonzero(game.legal_mask(current)))
            current = game.step(current, int(action))
            sign = -sign
        return priors, sign * value

    return evaluate
