# Roadmap

## Goal

A game-agnostic library for batched MCTS and AlphaZero / Gumbel MuZero–style self-play training, built on [MLX](https://github.com/ml-explore/mlx) and tuned for Apple Silicon.

The claim we want to be able to make, with numbers: **"Train a strong game-playing agent end to end on a single Mac, and know exactly how much strength each GPU-hour buys."**

### Non-goals

- Beating Lc0 or Stockfish in raw playing strength. They are built on distributed compute that a laptop cannot match. We compete on compute efficiency and on running the whole loop on a Mac.
- 19×19 Go and shogi. Too heavy to train meaningfully on a 24 GB laptop.
- Supporting CUDA. Other libraries already cover that.

## Design principles

- **Game-agnostic core, per-game weights.** The core holds search, self-play, the replay buffer, losses, and evaluation. Each game supplies its own rules (batched state transition + legal-move mask), observation encoding, network config, and trained weights.
- **Batched, fixed-size trees.** Like [mctx](https://github.com/google-deepmind/mctx), every tree is a set of fixed-shape arrays (`[batch, num_simulations + 1, ...]`), so thousands of games are searched at once without Python-side tree objects.
- **Stay on the GPU, use unified memory.** Rules, search, and network inference run on the GPU where possible. Work the GPU is bad at (e.g. chess move generation) can run on the CPU and hand results over without copies.
- **Profile before writing kernels.** Custom Metal kernels (`mx.fast.metal_kernel`) are written only where profiling shows the bottleneck.

## What we measured so far (2026-11-18, Apple M5, 24 GB, MLX 0.32.4)

- One PUCT select + backup step over 1,024 trees (128 nodes, 64 actions): about **1.4 ms eager, 0.94 ms with `mx.compile`**. The arithmetic is tiny, so most of this is per-op dispatch overhead.
- `uint64` shifts and bitwise ops run on the GPU and give correct results, so bitboards are viable. One early run produced a value we could not explain, so bitboard ops get their own tests in M0.
- Search-only speed on Connect4 (trivial evaluator, 64 simulations, best of 5, `benchmarks/search_speed.py`), in simulations per second:

  | Batch | Pure MLX | MLX + Metal kernels | mctx (JAX, CPU) |
  |---:|---:|---:|---:|
  | 1 | 722 | 2,908 | 735,632 |
  | 64 | 46,109 | 182,349 | 946,714 |
  | 1,024 | 671,797 | 2,178,195 | 1,154,093 |
  | 4,096 | 1,933,470 | 4,126,741 | 1,004,938 |
  | 16,384 | 3,583,192 | 4,821,925 | 850,456 |

  - The fused descent and backup kernels give exactly the same visit counts as pure MLX and are 2–4× faster up to batch 4,096 (1.35× at 16,384).
  - Against mctx on the CPU we lose badly at small batches (a fixed ~0.35 ms of MLX dispatch per simulation step) and win from batch 1,024 up (1.9× at 1,024, 4.1× at 4,096, 5.7× at 16,384).
  - This is without a network. With one, mctx on a Mac has to run inference on the CPU; that comparison is still to be measured.
  - What limits us now: per-step MLX op dispatch at small batches, and whole-array copies when writing node states and statistics at large batches.
- MLX has no `while_loop` / `scan` inside compiled graphs. Tree descent has a data-dependent depth, so a pure-MLX descent becomes a Python loop that pays dispatch overhead per level. This is the main motivation for a fused Metal kernel.

## Milestones

Each milestone ends with a go/no-go check and something written up. Weeks are targets, not promises.

### M0 — Search core + Connect4 (weeks 1–2)

- Batched MCTS in MLX: PUCT selection and Gumbel root selection (sequential halving), expansion, backup.
- Connect4 environment, batched on the GPU.
- Correctness: compare search results against a Connect4 solver on a fixed position set.
- Profile, then write a fused Metal kernel for tree descent + backup (one thread per tree).
- Benchmark: simulations/sec vs mctx on CPU and vs our own pure-MLX Python-loop version.

**Go/no-go:** the fused kernel gives a clear, reproducible speedup over both baselines. If it doesn't, rethink the architecture before building training on top.

### M1 — Othello from scratch (weeks 3–6) → v0.1 release

- Bitboard Othello environment on the GPU, validated with perft and known positions.
- Configurable ResNet (board shape + action count), self-play loop, replay buffer, training loop, checkpointing.
- Arena evaluation and Elo tracking during training.
- Evaluation against Edax at fixed levels.
- Publish the curve: strength vs M5 GPU-hours.

**Go/no-go:** an agent trained from zero on one Mac beats Edax at some non-trivial level, with the training run reproducible from a single command.

**Release:** v0.1 on GitHub, write-ups on Zenn (Japanese) and a personal blog (English), posts on X / Hacker News / r/MachineLearning.

### M2 — Chess (v0.2)

- Move generation validated by perft. Decide between a CPU implementation (Rust/C++, zero-copy into MLX) and a GPU one based on profiling.
- Supervised pretraining on public data (Lichess database and/or Lc0 training data; check each license before use).
- Self-play fine-tuning with Gumbel search at low simulation counts, plus KataGo-style efficiency tricks (playout cap randomization, auxiliary targets).
- Evaluation against small Lc0 networks and Stockfish at limited strength.
- Report separately how much strength comes from pretraining and how much from self-play.

**Headline metric:** Elo gained per M5 GPU-hour, not absolute Elo.

### M3 — 9×9 Go (v0.3)

- Batched Go rules: captures, ko / superko, scoring. [Pgx](https://github.com/sotetsuk/pgx) is the reference for vectorizing these rules.
- Evaluation against GNU Go and KataGo.

Do not start a new game until the previous one trains end to end.

## Validation and baselines per game

| Game | Rules validated by | Strength measured against |
|---|---|---|
| Connect4 (test env) | Solver | Perfect play |
| Othello | Perft, known positions | Edax levels |
| Chess | Perft | Small Lc0 networks, Stockfish (limited) |
| 9×9 Go | Position tests for captures and ko | GNU Go, KataGo |

## Numbers we intend to publish

- Simulations/sec (search only, and search + network) by batch size.
- Self-play games/hour.
- Elo vs M5 GPU-hours for each game.
- Peak memory during self-play and training.

## Risks

| Risk | Mitigation |
|---|---|
| Fused Metal kernel doesn't beat baselines | M0 go/no-go before any training work |
| Network inference, not search, dominates time | Profile early; tune batch size and network size before writing more kernels |
| Chess move generation is too branchy for the GPU | Run it on the CPU and use unified memory |
| 24 GB limits batch × tree size | Measure peak memory in M0; offload heavy pretraining to Colab if needed |
| Scope creep across three games | Strictly sequential milestones |
