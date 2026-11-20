# mac-alpha-zero

Game-agnostic batched MCTS and Gumbel AlphaZero–style self-play training on MLX for Apple Silicon. The plan and milestones are in ROADMAP.md; keep it current when scope or measurements change.

## Commands

```sh
uv sync                    # install (Python >= 3.10, managed by uv)
uv run pytest              # tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run python benchmarks/search_speed.py --backend metal              # search speed (or mlx)
uv run --group bench python benchmarks/search_speed.py --backend mctx # JAX/CPU baseline
uv run python benchmarks/search_speed.py --backend metal --network 64x4 # search speed with a ResNet
uv run python benchmarks/solver_agreement.py --checkpoint checkpoints/connect4.safetensors # optimal-move rate
uv run python examples/train_connect4.py --minutes 10                 # train Connect4 end to end
```

CI (GitHub Actions, macos-latest) runs lint, format check, and tests. Hosted runners may not expose the Metal GPU, so tests must pass on the MLX CPU device. Speed is measured locally on the M5, never in CI.

## Layout

- `src/mac_alpha_zero/reference/` — slow, single-game NumPy implementations: the correctness oracle. Clarity over speed. Do not optimize these.
  - `game.py` — the `Game` protocol. Values are always for the player to move.
  - `connect4.py`, `mcts.py` (PUCT), `solver.py` (depth-limited negamax).
- `src/mac_alpha_zero/games/` — batched MLX environments (`BatchedGame` protocol in `base.py`). States are dicts of arrays with a leading batch dimension.
- `src/mac_alpha_zero/mcts.py` — batched PUCT search. With `dtype=mx.float64` on the CPU it matches `reference.run_mcts` bit for bit; keep it that way (same operation order, first-index argmax tie-break).
- `src/mac_alpha_zero/kernels.py` — fused Metal kernels (tree descent, backup), one GPU thread per tree. `search(..., use_kernels=True)` must give the same visit counts as the pure-MLX path; tests skip when Metal is unavailable.
- `src/mac_alpha_zero/gumbel.py` — batched Gumbel root selection with Sequential Halving (`reference/gumbel.py` is its oracle; `sequential_halving.py` is the shared schedule). Nodes below the root still use PUCT; the paper's non-root selection rule is not implemented.
- `network.py` (ResNet + `network_evaluator`), `selfplay.py` (parallel games with auto-restart), `replay.py`, `train.py` (loss and one gradient step), `arena.py` (batched matches between agents). `examples/train_connect4.py` ties them together; checkpoints go to `checkpoints/` (git-ignored).

## Design rules

- Core code is game-agnostic. Each game supplies rules, observation encoding, network config, and its own weights.
- Trees are fixed-shape arrays (`[batch, num_simulations + 1, ...]`), mctx-style. No per-node Python objects outside `reference/`.
- MLX has no `while_loop` / `scan` inside compiled graphs. Data-dependent loops (tree descent) are Python loops or Metal kernels.
- Write a custom Metal kernel (`mx.fast.metal_kernel`) only after profiling shows the bottleneck, and test it against the pure-MLX version.
- Games are done strictly in order: Connect4 (test env) → Othello → chess → 9×9 Go. Don't start the next one until the previous one trains end to end.

## Testing conventions

- Every fast implementation gets an equivalence test against the reference: same inputs, same visit counts / values.
- The reference MCTS is deterministic for a deterministic evaluator (ties go to the lowest action index). Keep it that way.
- Random-rollout MCTS can miss some tactics even with many simulations. Tactical tests use positions whose winning moves were precomputed with `solve` at a deeper depth.
- MLX 0.32.4 evaluates `mx.exp` at float32 precision even for float64 inputs. Anything that goes through exp or softmax can only be compared to the reference with a tolerance (~1e-6).
- `games/connect4_solver.py` (exact bitboard solver) is only fast for positions with roughly 16+ stones; with fewer it can run for minutes or hours. Never call it on early positions in tests.
- Solver calls deeper than ~5 plies take seconds in Python. Precompute and hard-code their results in tests.
