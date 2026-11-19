"""Batched AlphaZero-style PUCT search in MLX.

Trees are fixed-shape arrays: node i of every tree is created by simulation i, so a
search with `num_simulations` simulations needs at most `num_simulations` nodes.
Semantics match `reference.mcts.run_mcts` exactly, including tie-breaking.
"""

from collections.abc import Callable
from typing import NamedTuple

import mlx.core as mx

from mac_alpha_zero import kernels
from mac_alpha_zero.games.base import BatchedGame, State

# Maps a batch of states to (priors (B, A), values (B,) for the player to move).
# It is also called on finished games; those outputs are ignored.
Evaluator = Callable[[State], tuple[mx.array, mx.array]]


class SearchResult(NamedTuple):
    visit_counts: mx.array  # (B, A) visits of the root's children
    root_value: mx.array  # (B,) mean root value for the player to move at the root


class _Tree:
    def __init__(self, root: State, batch: int, nodes: int, actions: int, dtype: mx.Dtype):
        self.states = {}
        for key, value in root.items():
            arr = mx.zeros((batch, nodes, *value.shape[1:]), dtype=value.dtype)
            arr[:, 0] = value
            self.states[key] = arr
        self.children = mx.full((batch, nodes, actions), -1, dtype=mx.int32)
        self.legal = mx.zeros((batch, nodes, actions), dtype=mx.bool_)
        self.prior = mx.zeros((batch, nodes, actions), dtype=dtype)
        self.parent = mx.full((batch, nodes), -1, dtype=mx.int32)
        self.expanded = mx.zeros((batch, nodes), dtype=mx.bool_)
        self.visits = mx.zeros((batch, nodes), dtype=dtype)
        # Sum of values for the player who chose the action leading to the node.
        self.value_sum = mx.zeros((batch, nodes), dtype=dtype)

    def arrays(self) -> list[mx.array]:
        return [
            *self.states.values(),
            self.children,
            self.legal,
            self.prior,
            self.parent,
            self.expanded,
            self.visits,
            self.value_sum,
        ]


def search(
    game: BatchedGame,
    root: State,
    evaluator: Evaluator,
    num_simulations: int,
    c_puct: float = 1.25,
    dtype: mx.Dtype = mx.float32,
    use_kernels: bool = False,
) -> SearchResult:
    """Run PUCT search on a batch of root states, one tree per state.

    `dtype` is the precision of the tree statistics. float64 only works on the CPU and is
    used to compare against the reference bit for bit.

    `use_kernels` runs tree descent and backup as fused Metal kernels. It needs the GPU
    and float32.
    """
    if use_kernels and (dtype != mx.float32 or not kernels.is_available()):
        raise ValueError("use_kernels needs float32 and the Metal GPU device")
    done, _ = game.terminal(root)
    if mx.any(done).item():
        raise ValueError("cannot search from a terminal state")
    batch = next(iter(root.values())).shape[0]
    tree = _Tree(root, batch, num_simulations, game.num_actions, dtype)
    rows = mx.arange(batch)

    for sim in range(num_simulations):
        if sim == 0:
            leaf = mx.zeros((batch,), dtype=mx.int32)
        else:
            walk = _descend_kernel if use_kernels else _descend
            node, new_parent, new_action = walk(tree, rows, sim, c_puct)
            leaf = _add_node(game, tree, rows, sim, node, new_parent, new_action)
        leaf_state = {k: v[rows, leaf] for k, v in tree.states.items()}
        value = _evaluate_and_expand(game, tree, rows, leaf, leaf_state, evaluator, dtype)
        if use_kernels:
            tree.visits, tree.value_sum = kernels.backup(
                tree.parent, tree.visits, tree.value_sum, leaf, value
            )
        else:
            _backup(tree, rows, leaf, value)
        mx.eval(*tree.arrays())

    children = tree.children[:, 0]
    child_visits = tree.visits[rows[:, None], mx.maximum(children, 0)]
    visit_counts = mx.where(children >= 0, child_visits, 0).astype(mx.int32)
    root_value = -tree.value_sum[:, 0] / tree.visits[:, 0]
    return SearchResult(visit_counts=visit_counts, root_value=root_value)


def _select(tree: _Tree, rows: mx.array, node: mx.array, c_puct: float) -> mx.array:
    children = tree.children[rows, node]
    exists = children >= 0
    index = mx.maximum(children, 0)
    child_visits = mx.where(exists, tree.visits[rows[:, None], index], 0)
    child_value = mx.where(exists, tree.value_sum[rows[:, None], index], 0)
    q = mx.where(child_visits > 0, child_value / mx.maximum(child_visits, 1), 0)
    sqrt_n = mx.sqrt(tree.visits[rows, node])[:, None]
    # Same operation order as the reference so float64 results match exactly.
    score = q + c_puct * tree.prior[rows, node] * sqrt_n / (1 + child_visits)
    score = mx.where(tree.legal[rows, node], score, -mx.inf)
    # argmax returns the first maximum, matching the reference's lowest-index tie-break.
    return mx.argmax(score, axis=-1).astype(mx.int32)


def _descend(
    tree: _Tree, rows: mx.array, sim: int, c_puct: float
) -> tuple[mx.array, mx.array, mx.array]:
    """Walk down every tree until the walk leaves it or reaches a terminal node.

    Returns (node, new_parent, new_action). Trees that need a new node have
    new_parent >= 0; the others stopped at the terminal node `node`.
    """
    batch = rows.shape[0]
    node = mx.zeros((batch,), dtype=mx.int32)
    active = tree.expanded[rows, node]
    new_parent = mx.full((batch,), -1, dtype=mx.int32)
    new_action = mx.zeros((batch,), dtype=mx.int32)
    # MLX has no while_loop inside graphs, so the depth loop lives in Python.
    while True:
        action = _select(tree, rows, node, c_puct)
        child = tree.children[rows, node, action]
        leaves = active & (child < 0)
        new_parent = mx.where(leaves, node, new_parent)
        new_action = mx.where(leaves, action, new_action)
        descend = active & (child >= 0)
        node = mx.where(descend, child, node)
        active = descend & tree.expanded[rows, node]
        if not mx.any(active).item():
            break
    return node, new_parent, new_action


def _descend_kernel(
    tree: _Tree, rows: mx.array, sim: int, c_puct: float
) -> tuple[mx.array, mx.array, mx.array]:
    return kernels.descend(
        tree.children,
        tree.legal,
        tree.prior,
        tree.expanded,
        tree.visits,
        tree.value_sum,
        sim,
        c_puct,
    )


def _add_node(
    game: BatchedGame,
    tree: _Tree,
    rows: mx.array,
    sim: int,
    node: mx.array,
    new_parent: mx.array,
    new_action: mx.array,
) -> mx.array:
    """Create node `sim` in the trees that left the tree. Returns the leaf per tree."""
    created = new_parent >= 0
    parent = mx.maximum(new_parent, 0)
    parent_state = {k: v[rows, parent] for k, v in tree.states.items()}
    # Trees that did not create a node leave slot `sim` unused, so writing it is harmless.
    child_state = game.step(parent_state, new_action)
    for key, value in child_state.items():
        tree.states[key][:, sim] = value
    current = tree.children[rows, parent, new_action]
    tree.children[rows, parent, new_action] = mx.where(created, sim, current)
    tree.parent[:, sim] = mx.where(created, new_parent, -1)
    return mx.where(created, sim, node)


def _evaluate_and_expand(
    game: BatchedGame,
    tree: _Tree,
    rows: mx.array,
    leaf: mx.array,
    leaf_state: State,
    evaluator: Evaluator,
    dtype: mx.Dtype,
) -> mx.array:
    done, terminal_value = game.terminal(leaf_state)
    priors, values = evaluator(leaf_state)
    value = mx.where(done, terminal_value.astype(dtype), values.astype(dtype))

    legal = game.legal_mask(leaf_state)
    masked = mx.where(legal, priors.astype(dtype), 0)
    total = masked.sum(axis=-1, keepdims=True)
    uniform = legal.astype(dtype) / mx.maximum(legal.sum(axis=-1, keepdims=True), 1)
    normalized = mx.where(total > 0, masked / mx.where(total > 0, total, 1), uniform)

    expand = ~done
    tree.prior[rows, leaf] = mx.where(expand[:, None], normalized, tree.prior[rows, leaf])
    tree.legal[rows, leaf] = mx.where(expand[:, None], legal, tree.legal[rows, leaf])
    tree.expanded[rows, leaf] = tree.expanded[rows, leaf] | expand
    return value


def _backup(tree: _Tree, rows: mx.array, leaf: mx.array, value: mx.array) -> None:
    node = leaf
    while True:
        valid = node >= 0
        index = mx.maximum(node, 0)
        tree.value_sum = tree.value_sum.at[rows, index].add(mx.where(valid, -value, 0))
        tree.visits = tree.visits.at[rows, index].add(valid.astype(tree.visits.dtype))
        value = -value
        node = mx.where(valid, tree.parent[rows, index], -1)
        if not mx.any(node >= 0).item():
            break


def random_rollout_evaluator(game: BatchedGame, seed: int = 0) -> Evaluator:
    """Uniform priors and the result of one uniformly random playout per state."""
    key = mx.random.key(seed)

    def evaluate(state: State) -> tuple[mx.array, mx.array]:
        nonlocal key
        batch = next(iter(state.values())).shape[0]
        priors = mx.full((batch, game.num_actions), 1.0 / game.num_actions)
        result = mx.zeros((batch,))
        finished = mx.zeros((batch,), dtype=mx.bool_)
        sign = mx.ones((batch,))
        current = state
        while True:
            done, value = game.terminal(current)
            result = mx.where(done & ~finished, sign * value, result)
            finished = finished | done
            if mx.all(finished).item():
                return priors, result
            # Finished games sample uniformly so every row has a valid distribution.
            allowed = game.legal_mask(current) | done[:, None]
            key, subkey = mx.random.split(key)
            actions = mx.random.categorical(mx.where(allowed, 0.0, -mx.inf), key=subkey)
            current = game.step(current, actions.astype(mx.int32))
            sign = -sign

    return evaluate
