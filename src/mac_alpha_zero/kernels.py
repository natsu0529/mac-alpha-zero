"""Fused Metal kernels for the data-dependent loops of batched MCTS.

One GPU thread handles one tree, so walking down a tree and walking back up are a single
dispatch each instead of one group of MLX ops per tree level. float32 only.
"""

import mlx.core as mx

_DESCEND_SOURCE = """
    uint b = thread_position_in_grid.x;
    int num_nodes = visits_shape[1];
    int num_actions = children_shape[2];
    int row = b * num_nodes;
    float c = c_puct[0];

    int node = 0;
    int parent = -1;
    int chosen = 0;
    while (expanded[row + node]) {
        int base = (row + node) * num_actions;
        float sqrt_n = metal::sqrt(visits[row + node]);
        float best = -INFINITY;
        int best_action = 0;
        // Strict '>' keeps the lowest action index on ties, like argmax.
        for (int a = 0; a < num_actions; a++) {
            if (!legal[base + a]) continue;
            int child = children[base + a];
            float n = 0.0f;
            float q = 0.0f;
            if (child >= 0) {
                n = visits[row + child];
                if (n > 0.0f) q = value_sum[row + child] / n;
            }
            float score = q + c * prior[base + a] * sqrt_n / (1.0f + n);
            if (score > best) {
                best = score;
                best_action = a;
            }
        }
        int next = children[base + best_action];
        if (next < 0) {
            parent = node;
            chosen = best_action;
            break;
        }
        node = next;
    }
    leaf[b] = parent >= 0 ? sim[0] : node;
    new_parent[b] = parent;
    new_action[b] = chosen;
"""

_BACKUP_SOURCE = """
    uint b = thread_position_in_grid.x;
    int num_nodes = visits_shape[1];
    int row = b * num_nodes;
    for (int i = 0; i < num_nodes; i++) {
        visits_out[row + i] = visits[row + i];
        value_sum_out[row + i] = value_sum[row + i];
    }
    float v = value[b];
    int node = leaf[b];
    while (node >= 0) {
        value_sum_out[row + node] -= v;
        visits_out[row + node] += 1.0f;
        v = -v;
        node = parent[row + node];
    }
"""

_descend = None
_backup = None


def is_available() -> bool:
    return mx.metal.is_available() and mx.default_device() == mx.gpu


def descend(
    children: mx.array,
    legal: mx.array,
    prior: mx.array,
    expanded: mx.array,
    visits: mx.array,
    value_sum: mx.array,
    sim: int,
    c_puct: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """PUCT walk from every root to where it leaves the tree.

    Returns (leaf, new_parent, new_action), each (B,) int32. Trees that need a new node
    have leaf == sim and new_parent >= 0. Trees that reached a terminal node again have
    that node as the leaf and new_parent == -1.
    """
    global _descend
    if _descend is None:
        _descend = mx.fast.metal_kernel(
            name="mcts_descend",
            input_names=[
                "children",
                "legal",
                "prior",
                "expanded",
                "visits",
                "value_sum",
                "sim",
                "c_puct",
            ],
            output_names=["leaf", "new_parent", "new_action"],
            source=_DESCEND_SOURCE,
        )
    batch = visits.shape[0]
    return _descend(
        inputs=[
            children,
            legal,
            prior,
            expanded,
            visits,
            value_sum,
            mx.array([sim], dtype=mx.int32),
            mx.array([c_puct], dtype=mx.float32),
        ],
        grid=(batch, 1, 1),
        threadgroup=(min(batch, 256), 1, 1),
        output_shapes=[(batch,)] * 3,
        output_dtypes=[mx.int32] * 3,
    )


def backup(
    parent: mx.array,
    visits: mx.array,
    value_sum: mx.array,
    leaf: mx.array,
    value: mx.array,
) -> tuple[mx.array, mx.array]:
    """Add one visit along every leaf-to-root path. Returns new (visits, value_sum)."""
    global _backup
    if _backup is None:
        _backup = mx.fast.metal_kernel(
            name="mcts_backup",
            input_names=["parent", "visits", "value_sum", "leaf", "value"],
            output_names=["visits_out", "value_sum_out"],
            source=_BACKUP_SOURCE,
        )
    batch = visits.shape[0]
    return _backup(
        inputs=[parent, visits, value_sum, leaf, value],
        grid=(batch, 1, 1),
        threadgroup=(min(batch, 256), 1, 1),
        output_shapes=[visits.shape, value_sum.shape],
        output_dtypes=[mx.float32, mx.float32],
    )
