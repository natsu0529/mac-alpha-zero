# mac-alpha-zero

**AlphaZero-style self-play training that runs end to end on your Mac.**

`mac-alpha-zero` is a game-agnostic library for batched Monte Carlo Tree Search and AlphaZero / Gumbel MuZero–style training, built on [MLX](https://github.com/ml-explore/mlx) for Apple Silicon.

> **Status: pre-alpha.** Nothing is runnable yet. The plan, milestones, and the measurements taken so far are in [ROADMAP.md](ROADMAP.md).

## Why

- **There is no MCTS / AlphaZero library for MLX.** [mctx](https://github.com/google-deepmind/mctx) is excellent but built on JAX, whose Mac GPU support is experimental. PyTorch-based frameworks aren't tuned for Apple Silicon.
- **Unified memory changes the trade-offs.** On a Mac, game rules, tree search, and network inference share one memory pool. Work that suits the CPU (like chess move generation) can feed the GPU without copies.
- **Compute efficiency is the interesting question.** We don't try to out-muscle Lc0 or KataGo. We measure how much playing strength each GPU-hour on a single Mac buys, and push that number up.

## Planned features

- Batched MCTS with fixed-shape tree arrays: thousands of games searched in parallel.
- PUCT and Gumbel root selection (strong policy improvement at low simulation counts).
- Fused Metal kernels for tree descent and backup.
- Self-play, replay buffer, training loop, and arena evaluation with Elo tracking.
- A small environment interface: bring your own game by supplying batched rules and an observation encoding.

## Games

The core is game-agnostic. Each game has its own rules, network config, and trained weights.

| Game | Approach | Target release |
|---|---|---|
| Connect4 | Internal test environment, checked against a solver | v0.1 |
| Othello | Trained from scratch, evaluated against Edax | v0.1 |
| Chess | Pretrained on public data, fine-tuned by self-play | v0.2 |
| 9×9 Go | Trained by self-play, evaluated against GNU Go / KataGo | v0.3 |

## Requirements (planned)

- Apple Silicon Mac (developed on an M5 with 24 GB)
- macOS with a recent Python and MLX

## Contributing

The project is at the design stage. Issues and discussion about the API and the roadmap are welcome.

## License

[MIT](LICENSE) — do whatever you want, just keep the notice.
