# Roadmap

## Goal

A game-agnostic library for batched MCTS and Gumbel AlphaZero–style self-play training, built on [MLX](https://github.com/ml-explore/mlx) and tuned for Apple Silicon.

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
- First end-to-end training run (2026-11-19, `examples/train_connect4.py --minutes 10`, defaults: 256 parallel games, 32 Gumbel simulations, 4 blocks × 64 channels, Metal kernels on). Score is the mean result over 128 games (win 1, draw 0.5), half with each side moving first:

  | Training minutes | Self-play games | Policy only vs random | Search vs random | Policy only vs rollout MCTS (100 sims) | Search vs rollout MCTS (100 sims) |
  |---:|---:|---:|---:|---:|---:|
  | 0 | 0 | 0.80 | 0.90 | 0.02 | 0.16 |
  | 3.0 | 6,321 | 1.00 | 1.00 | 0.60 | 0.80 |
  | 7.7 | 11,325 | 1.00 | 0.99 | 0.75 | 0.89 |
  | 10.2 | 12,476 | 1.00 | 1.00 | 0.70 | 0.90 |

  One run, one seed, weak opponents: this shows the loop learns, not how strong it gets. About 20 self-play games per second.
- Search speed with a network (2026-11-20, Connect4, random 64-channel 4-block ResNet at every leaf, 64 simulations, best of 3, `benchmarks/search_speed.py --network 64x4`), in simulations per second:

  | Batch | Pure MLX | MLX + Metal kernels | mctx (JAX, CPU) | Kernels vs mctx |
  |---:|---:|---:|---:|---:|
  | 1 | — | 1,167 | 6,353 | 0.18× |
  | 64 | 8,710 | 51,781 | 17,999 | 2.9× |
  | 256 | 28,307 | 70,485 | 18,899 | 3.7× |
  | 1,024 | 34,885 | 62,789 | 13,621 | 4.6× |
  | 4,096 | — | 50,367 | 9,899 | 5.1× |

  - With a network we beat mctx on the CPU from batch 64 up, by 3–5×. Only batch 1 is slower.
  - Network inference is now nearly the whole cost: the network alone runs about 31k–49k positions/s on the M5 GPU, peaking near batch 1,024 and dropping at 4,096. Search throughput peaks around batch 256.
  - So the next speedups have to come from inference (smaller or compiled network, half precision, batch size), not from more tree kernels.
  - Not measured here: JAX on the Mac GPU (jax-metal). It is covered in the 2026-11-27 comparison below. A trained rather than random network is still unmeasured.
- Search speed against mctx on the Mac GPU ([#31](https://github.com/natsu0529/mac-alpha-zero/issues/31), 2026-11-27). This used the same benchmark and random 64x4 network with 64 simulations, best of 5, at commit `65d84b5`. The Mac was on AC power with other apps closed, and `pmset -g therm` reported no thermal warning before any measurement.
  - **Versions.** jax-metal 0.1.1, the latest release, requires JAX 0.4.34. mctx on Metal therefore ran in a separate Python 3.12 environment with jax and jaxlib 0.4.34, jax-metal 0.1.1, mctx 0.0.5, chex 0.1.89 and pgx 2.6.0. mctx on the CPU ran in that environment and in the project environment (JAX 0.12.1, mctx 0.0.71). The CPU runs set `JAX_PLATFORMS=cpu`, and the Metal runs printed `Metal device set to: Apple M5`.

  Simulations per second, with the network:

  | Batch | Pure MLX (GPU) | MLX + Metal kernels (GPU) | mctx, JAX 0.12.1 (CPU) | mctx, JAX 0.4.34 (CPU) | mctx, JAX 0.4.34 (Metal) |
  |---:|---:|---:|---:|---:|---:|
  | 1 | 602 | 1,532 | 8,258 | 3,419 | 838 |
  | 64 | 29,187 | 54,096 | 23,749 | 14,242 | 14,870 |
  | 256 | 57,436 | 74,855 | 26,367 | 16,465 | 17,435 |
  | 1,024 | 73,366 | 79,804 | 25,018 | 15,230 | 18,273 |
  | 4,096 | 59,699 | 61,374 | 24,487 | 14,342 | Did not finish in 900 s |

  The same run without a network (search only):

  | Batch | Pure MLX (GPU) | MLX + Metal kernels (GPU) | mctx, JAX 0.12.1 (CPU) | mctx, JAX 0.4.34 (CPU) | mctx, JAX 0.4.34 (Metal) |
  |---:|---:|---:|---:|---:|---:|
  | 1 | 760 | 3,012 | 699,453 | 82,051 | 964 |
  | 64 | 48,752 | 197,733 | 962,887 | 409,080 | 18,172 |
  | 256 | 188,885 | 727,656 | 934,469 | 495,686 | 21,535 |
  | 1,024 | 692,857 | 2,355,969 | 1,149,006 | 489,894 | 22,367 |
  | 4,096 | 1,973,838 | 4,355,929 | 1,054,587 | 563,346 | Did not finish in 900 s |

  - **GPU against GPU.** With the network, the Metal kernels were 3.6–4.4× faster than mctx on Metal at batches 64–1,024. At JAX 0.4.34, moving mctx from the CPU to Metal gained only 1.0–1.2× at those batches and was 4× slower at batch 1. With jax-metal 0.1.1, the lead over mctx therefore does not come mainly from the GPU.
  - **GPU against CPU.** Against mctx 0.0.71 on the CPU, the kernels were 2.3–3.2× faster from batch 64 up and 5.4× slower at batch 1. This is below the 3–5× measured on November 20.
  - **Kernels against pure MLX** (same GPU, same visit counts): 2.5× at batch 1, 1.85× at 64, 1.30× at 256, 1.09× at 1,024 and 1.03× at 4,096. At large batches, network inference takes almost all of the time.
  - **jax-metal at batch 4,096.** Both mctx runs on Metal printed nothing more after batch 1,024. They were stopped after 900 s, with and without the network. The night before, the same run had hung for hours inside `BlockUntilReady`. No error was printed.
  - **Correctness on Metal is unverified.** We compared mctx action weights between the CPU and Metal devices at JAX 0.4.34. Without a network they matched at batches 64 and 1,024 but not at batch 1. With the network they differed by about as much as the CPU results of JAX 0.4.34 and 0.12.1 differ from each other, so the check is inconclusive.
  - **Change since November 20.** Pure MLX and mctx on the CPU ran faster than on November 20 with the same MLX, JAX and mctx versions: 57,436 against 28,307, and 26,367 against 18,899, at batch 256. That run used best of 3 without recording other load on the Mac, and the search code changed on November 22. We have not isolated the cause.
  - Not measured: earlier M-series chips, the second part of #31.

  Each cell comes from a command like the one below. Replace `--backend mctx` with `metal` or `mlx` for the MLX columns, and drop `--network 64x4` for search only. Run mctx on Metal with the separate environment's Python and without `JAX_PLATFORMS`.

  ```sh
  JAX_PLATFORMS=cpu uv run --group bench python benchmarks/search_speed.py --backend mctx --network 64x4 --batch 1 64 256 1024 4096 --sims 64 --repeats 5
  ```
- Agreement with the exact solver (2026-11-20, `benchmarks/solver_agreement.py`, 1,000 solved positions from random play with 14–30 stones; "quiet" = the 398 positions without an immediate win). Share of moves that keep the best achievable result:

  | Agent | All positions | Quiet only |
  |---|---:|---:|
  | Random | 0.326 | 0.319 |
  | Rollout MCTS, 100 sims | 0.895 | 0.736 |
  | Rollout MCTS, 400 sims | 0.943 | 0.857 |
  | Network (10-minute run), policy only | 0.741 | 0.671 |
  | Network + Gumbel search, 32 sims | 0.878 | 0.739 |
  | Network + Gumbel search, 128 sims | 0.920 | 0.817 |

  After 10 minutes the network with search is about level with 100-simulation rollout MCTS on these positions and below 400. The positions come from random play, which self-play rarely visits, so this understates in-distribution strength; it is still the honest absolute number we have.
- First Othello training run (2026-11-21, `examples/train_othello.py --minutes 10`, defaults: 256 parallel games, 32 Gumbel simulations, 4 blocks × 64 channels, Metal kernels on). Score over 64 games, half with each side moving first:

  | Training minutes | Self-play games | Policy only vs random | Search vs random | Policy only vs rollout MCTS (8 sims) | Search vs rollout MCTS (8 sims) |
  |---:|---:|---:|---:|---:|---:|
  | 0 | 0 | 0.45 | 0.53 | 0.34 | 0.13 |
  | 3.8 | 1,537 | 0.79 | 0.91 | 0.49 | 0.75 |
  | 7.5 | 3,329 | 0.75 | 0.98 | 0.68 | 0.90 |
  | 10.0 | 4,115 | 0.89 | 1.00 | 0.77 | 0.90 |

  One run, one seed, and an 8-simulation rollout MCTS is a very weak opponent: this only shows that the loop learns Othello. About 7 self-play games per second (Connect4: 20). Gumbel search with a random 64x4 network runs about 32k simulations/s at batch 256, against 70k for Connect4, so the bitboard rules now cost about as much as the network. Random rollouts are slow (8 simulations on 64 boards take about 1.1 s), which is why the baseline is so weak; Edax replaces it.
- First match against Edax 4.6 (2026-11-21, `benchmarks/edax_match.py`, the 10-minute network above, 100 games per cell from 50 random 6-ply openings played with both colours, Edax without its book). Score, then wins-draws-losses:

  | Agent | Edax level 1 | Edax level 2 | Edax level 3 |
  |---|---:|---:|---:|
  | Random | 0.020 (2-0-98) | 0.000 (0-0-100) | 0.000 (0-0-100) |
  | Network, policy only | 0.040 (4-0-96) | 0.000 (0-0-100) | 0.000 (0-0-100) |
  | Network + Gumbel search, 32 sims | 0.090 (9-0-91) | 0.000 (0-0-100) | 0.000 (0-0-100) |
  | Network + Gumbel search, 128 sims | 0.130 (12-2-86) | 0.000 (0-0-100) | 0.000 (0-0-100) |

  After 10 minutes the agent is far below even Edax level 1 and wins nothing at level 2. This is the starting point of the strength curve, and it shows how weak the random and rollout baselines are. The whole table takes about 4.5 minutes.
- First long Othello run (2026-11-21, `examples/train_othello.py --minutes 120`, same defaults: 256 parallel games, 32 Gumbel simulations, 4 blocks × 64 channels, constant learning rate 2e-3, 200k-sample buffer; one seed). Scores during training over 64 games per cell, search with 32 simulations:

  | Training minutes | Self-play games | Loss | Policy / search vs Edax level 1 | Policy / search vs Edax level 2 | Policy / search vs Edax level 3 |
  |---:|---:|---:|---:|---:|---:|
  | 0 | 0 | — | 0.00 / 0.00 | 0.00 / 0.03 | 0.00 / 0.02 |
  | 19.5 | 10,010 | 2.228 | 0.04 / 0.19 | 0.00 / 0.02 | 0.00 / 0.00 |
  | 44.5 | 20,231 | 2.189 | 0.16 / 0.22 | 0.03 / 0.05 | 0.00 / 0.00 |
  | 61.2 | 26,933 | 2.224 | 0.17 / 0.23 | 0.02 / 0.05 | 0.00 / 0.05 |
  | 90.5 | 40,491 | 2.189 | 0.19 / 0.17 | 0.02 / 0.02 | 0.00 / 0.03 |
  | 120.0 | 56,076 | 2.203 | 0.17 / 0.29 | 0.05 / 0.01 | 0.00 / 0.02 |

  The final network over 100 games per cell (`benchmarks/edax_match.py`), score and wins-draws-losses:

  | Agent | Edax level 1 | Edax level 2 |
  |---|---:|---:|
  | Random | 0.020 (2-0-98) | 0.000 (0-0-100) |
  | Network, policy only | 0.170 (16-2-82) | 0.005 (0-1-99) |
  | Network + Gumbel search, 32 sims | 0.230 (23-0-77) | 0.020 (2-0-98) |
  | Network + Gumbel search, 128 sims | 0.285 (28-1-71) | 0.100 (10-0-90) |
  | Network + Gumbel search, 512 sims | 0.405 (39-3-58) | 0.070 (6-2-92) |

  - Two hours do not reach Edax level 1. Strength and loss both stop improving after roughly 45–60 minutes (20k–27k games); the second hour buys almost nothing with these settings.
  - More search helps at level 1 (0.23 → 0.41 from 32 to 512 simulations), so the value head carries some signal, but not enough.
  - On 20,000 samples of the final replay buffer the network's value has the right sign 73% of the time (59% with fewer than 20 discs, 86% with 52 or more) and a squared error of 0.62 against outcomes with variance 0.96. The policy is 0.64 nats (KL) away from the search targets, whose own entropy is 0.91.
  - Candidates identified after this first run: the constant learning rate, the small network, the 200k buffer (only the last ~50 iterations), no use of the board's 8 symmetries, and targets from only 32 simulations.
  - The run finished 7.8 games per **training second** (self-play plus optimizer updates). The two slow stretches coincide with closed-lid maintenance wakes on AC power. We aligned all 18 evaluation bursts in the kernel's Edax process log with the 18 JSONL records, using `pmset -g log` to account for time spent asleep. Local times below are JST on November 21.

    | Iterations | Games per training second | Mac power state during training |
    |---|---:|---|
    | 0–200 | 8.2–8.9 per 50 iterations | Lid closed at 04:25, near the end of the interval |
    | 200–350 | 4.8–6.4 | Sleep and maintenance wakes. Silent Running stayed clamped even during the uninterrupted 07:24–07:36 training window. |
    | 350–450 | 11.2–11.3 | `powerd` logged Silent Running unclamped at 07:38:48 |
    | 500–650 | 6.0–6.4 | Silent Running clamped again at 07:54:52. The three training windows contained 11, 10, and 11 sleep entries. |
    | 650–831 | 8.9–11.0 | Lid opened and Silent Running unclamped at 13:04:06 |

    The populated replay buffer stayed at its 200k capacity across both recoveries. Evaluation runs synchronously outside the training timer. The 600–650 window alone took about 70 wall minutes for 8.7 training minutes. Sleep explains the missing wall time, and both changes out of the clamped power state coincide with recovery in awake throughput. The original run did not record GPU clocks, temperature, or swap activity, so it cannot isolate their individual contributions. For comparable throughput, keep the lid open on AC and prevent idle sleep for the duration of the command, e.g. `caffeinate -i uv run python examples/train_othello.py --minutes 120`.
- Ratings for the first 2-hour Othello run, fitted to all 18 evaluations with `uv run python benchmarks/checkpoint_elo.py checkpoints/othello.log.jsonl`. Edax level 1 is fixed at 0 Elo. A Bradley–Terry fit uses the 32-simulation search scores against random and Edax levels 1–3 (64 games per opponent, draws worth half a point), with an 800-Elo Gaussian prior on each free rating to keep undefeated players finite. These are estimates from one seed and repeated openings.

  | Training minutes | 0 | 6.8 | 13.2 | 19.5 | 25.8 | 35.5 | 44.5 | 56.2 | 61.2 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Elo | -1127 | -420 | -396 | -273 | -336 | -280 | -240 | -246 | -189 |

  | Training minutes | 66.1 | 72.1 | 81.6 | 90.5 | 99.2 | 104.3 | 110.1 | 116.1 | 120.0 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Elo | -246 | -210 | -228 | -266 | -189 | -189 | -174 | -210 | -194 |

  The original run overwrote intermediate weights. For a new Othello run, pass `--save-eval-weights` to training, then add `--match-games 64` to the rating command to include adjacent checkpoints playing the same openings as both colors. Redirect stdout to save the ratings as JSONL.
- MLX has no `while_loop` / `scan` inside compiled graphs. Tree descent has a data-dependent depth, so a pure-MLX descent becomes a Python loop that pays dispatch overhead per level. This is the main motivation for a fused Metal kernel.

## Two-hour Othello comparison

On November 22, 2026, two Othello agents trained from scratch on an Apple M5 with 24 GB, macOS 26.6.2 and MLX 0.32.4 at commit `5d22b64`. Both used seed 0, the eight board symmetries, a 4-block 64-channel ResNet, a 200k replay buffer, 256 parallel games and 32 Gumbel simulations with PUCT below the root. The first used a constant learning rate of 0.002. The second started at 0.002 and halved it every 60 training minutes. The [README](README.md#results-so-far) gives the training and match commands. To reproduce the second run, change the checkpoint basename from `sym` to `decay` and add `--lr-half-life-minutes 60` to training.

Each cell below gives the Edax 4.6 **level 1 / level 2** scores, with a draw worth half a point. Each score covers 100 games with opening seed 0, 50 six-ply openings and both colors. The baseline is the November 21 checkpoint.

| Final agent | Baseline, no augmentation | Symmetries, constant rate | Symmetries, decaying rate |
|---|---:|---:|---:|
| Policy only | 0.170 / 0.005 | 0.715 / 0.270 | 0.530 / 0.305 |
| Search, 32 simulations | 0.230 / 0.020 | 0.905 / 0.605 | 0.920 / 0.640 |
| Search, 128 simulations | 0.285 / 0.100 | 0.990 / 0.820 | 0.980 / 0.875 |
| Search, 512 simulations | 0.405 / 0.070 | 1.000 / 0.960 | 1.000 / 0.950 |

| Training run | Self-play + updates | Wall time including evaluations and saves | Self-play games | Training process peak footprint |
|---|---:|---:|---:|---:|
| Baseline | 120.0 min | Interrupted by sleep | 56,076 | Unmeasured |
| Symmetries, constant | 120.0 min | 148.6 min | 78,809 | 1,816,888,208 bytes (1.692 GiB) |
| Symmetries, decaying | 120.1 min | 148.3 min | 77,514 | 1,814,692,800 bytes (1.690 GiB) |

Peaks are macOS `/usr/bin/time -l` peak memory footprints for the timed Python process. The final match processes peaked at 0.122 and 0.123 GiB respectively. Each configuration ran once. The baseline experienced closed-lid slowdowns and generated fewer games, so the comparison cannot isolate the size of the augmentation effect. The [awake rerun below](#awake-baseline-and-a-second-seed) addresses this. Decay helped at some search budgets and hurt at others. We retain the constant rate.

The constant-rate agent scored 0.516 against level 1 at 14.1 training minutes in the 64-game checks. Its final 100-game test at 32 simulations yielded 90 wins, one draw and nine losses. The [strength curve](docs/othello-sym-strength.png) shows all 25 checks and the fitted Elo, ending at +354 relative to level 1. The archived evaluation logs cover both [constant](benchmarks/data/othello/sym-2h.log.jsonl) and [decaying](benchmarks/data/othello/decay-2h.log.jsonl) rates. SHA-256 of the constant-rate final weights: `7f9d0a671fccd67ba6ac7b29b5719a91b1d27cea216449919efcb04138f66bb6`.

### Awake baseline and a second seed

On November 26–27, 2026, two more two-hour runs trained at commit `65d84b5` on the same M5. Both ran on AC power with the lid open, under `caffeinate -is`, with other apps closed. The first repeated the November 21 baseline: seed 0 without augmentation. The second used the eight symmetries with seed 1. Training and match settings were otherwise the same as above, and both passed `--save-eval-weights`.

The run without augmentation was interrupted at 71.4 training minutes (iteration 687). It continued with `--resume` the next night for the remaining 48.6 minutes. Resuming restores the weights, optimizer, replay buffer and counters, but not the games in progress or the random state, so this run is not identical to an uninterrupted one.

Each cell gives the Edax 4.6 **level 1 / level 2** scores with the same 100 games per level as above:

| Final agent | No augmentation, seed 0, Nov 21 (sleep) | No augmentation, seed 0, Nov 27 (awake) | Symmetries, seed 0, Nov 22 | Symmetries, seed 1, Nov 27 |
|---|---:|---:|---:|---:|
| Policy only | 0.170 / 0.005 | 0.195 / 0.010 | 0.715 / 0.270 | 0.640 / 0.360 |
| Search, 32 simulations | 0.230 / 0.020 | 0.290 / 0.065 | 0.905 / 0.605 | 0.910 / 0.765 |
| Search, 128 simulations | 0.285 / 0.100 | 0.435 / 0.070 | 0.990 / 0.820 | 0.990 / 0.880 |
| Search, 512 simulations | 0.405 / 0.070 | 0.520 / 0.115 | 1.000 / 0.960 | 1.000 / 0.950 |

| Training run | Self-play + updates | Wall time including evaluations and saves | Self-play games | Training process peak footprint |
|---|---:|---:|---:|---:|
| No augmentation, seed 0, awake | 120.1 min | 151.6 min (88.7 + 62.9) | 74,273 | 1,820,083,160 bytes (1.695 GiB) before the interruption, 2,130,347,112 bytes (1.984 GiB) after resuming |
| Symmetries, seed 1 | 120.0 min | 149.8 min | 78,312 | 1,816,150,904 bytes (1.691 GiB) |

- **The augmentation effect does not come from sleep.** Awake, the unaugmented run completed 74,273 games instead of 56,076 and scored a little higher than on November 21. It still reached only 0.290 against level 1 at 32 simulations, against 0.905 and 0.910 with symmetries. After 20 training minutes, its 64-game checks at 32 simulations stayed between 0.13 and 0.32 against level 1. The plateau seen on November 21 was therefore not caused by sleep.
- **A second seed gave the same result at level 1.** The seed-1 agent scored 0.910, 0.990 and 1.000 against level 1 at 32, 128 and 512 simulations, within one point of seed 0. At level 2 it scored higher at 32 and 128 simulations (0.765 against 0.605, 0.880 against 0.820). Its policy alone was weaker against level 1 (0.640 against 0.715). With two seeds, level-2 scores varied by up to 0.16 between runs.
- **Memory after resuming.** We did not investigate the 0.29 GiB higher peak after resuming. Loading the saved 200k-entry replay buffer is a likely cause. Each final match process peaked at 0.125 GiB.

The evaluation logs are archived for the [awake run without augmentation](benchmarks/data/othello/nosym-awake-2h.log.jsonl) and the [seed-1 run](benchmarks/data/othello/sym-seed1-2h.log.jsonl).

**Non-root selection at low search budgets.** On the original, unaugmented checkpoint, PUCT below the Gumbel root scored 0.090 at 8 simulations and 0.180 at 16 against Edax level 1. On the same paired openings, `--improved-nonroot` scored 0.090 and 0.190. Each cell comprised 100 games. At 16 simulations this amounted to one extra point for the alternative rule. PUCT remains the default.

## Equal-time Othello comparison

For the network, replay, and search-target comparison, give each configuration 60 minutes of self-play and optimization on the same M5. Use seed 0, board symmetries, constant learning rate 0.002, and 256 parallel games. Evaluate only before and after the timed work. Run one configuration at a time with the lid open on AC.

| Run | Residual blocks × channels | Replay capacity | Self-play simulations |
|---|---:|---:|---:|
| Control | 4 × 64 | 200k | 32 |
| Larger network | 6 × 96 | 200k | 32 |
| Larger replay | 4 × 64 | 500k | 32 |
| More search | 4 × 64 | 200k | 64 |
| More search | 4 × 64 | 200k | 128 |

For the control:

```sh
uv run python examples/train_othello.py --minutes 60 --symmetries --eval-every 100000 --checkpoint checkpoints/control.safetensors
uv run python benchmarks/edax_match.py --checkpoint checkpoints/control.safetensors --games 100 --levels 1 2 --simulations 32 128 512
```

Use a fresh checkpoint path for each run. Change only `--blocks 6 --channels 96`, `--buffer 500000`, or `--simulations 64` / `128` in training. For the larger network, also pass its shape to the match command. The runner's clock excludes evaluation and checkpoint I/O. Record actual elapsed time separately. Each final match uses the same 50 six-ply openings, once per color. Compare the level-1 score at **128 evaluation simulations** across configurations, retaining the control in a tie. Report the remaining budgets and level 2 alongside that comparison, then check the leading configuration on fresh openings before changing defaults.

### Results on the M5

On November 23, 2026, each configuration trained once from scratch at commit `5d22b64` on the same 24 GB M5 with MLX 0.32.4. All five used symmetry augmentation. The elapsed times include the initial and final evaluations and saving the checkpoint. The peak is the macOS footprint of the Python training process.

| Configuration | Training minutes | Elapsed (min:sec) | Self-play games | Peak (GiB) |
|---|---:|---:|---:|---:|
| Control | 60.02 | 62:26 | 38,927 | 1.693 |
| 6 × 96 network | 60.03 | 62:54 | 13,761 | 2.692 |
| 500k replay | 60.06 | 62:26 | 40,322 | 1.906 |
| 64 self-play simulations | 60.06 | 63:14 | 24,572 | 1.702 |
| 128 self-play simulations | 60.10 | 64:43 | 13,569 | 1.728 |

Each cell gives **Edax level 1 / level 2** scores on opening seed 0, with 100 games per level and half a point per draw. The column headers give the evaluation search budget for all five agents.

| Configuration | Policy only | 32 simulations | 128 simulations | 512 simulations |
|---|---:|---:|---:|---:|
| Control | 0.570 / 0.215 | 0.875 / 0.615 | 0.965 / 0.775 | 0.975 / 0.900 |
| 6 × 96 network | 0.490 / 0.115 | 0.850 / 0.540 | 0.940 / 0.700 | 0.990 / 0.865 |
| 500k replay | 0.480 / 0.185 | 0.880 / 0.510 | 0.970 / 0.740 | 0.995 / 0.960 |
| 64 self-play simulations | 0.500 / 0.110 | 0.905 / 0.515 | 0.925 / 0.825 | 0.985 / 0.930 |
| 128 self-play simulations | 0.335 / 0.025 | 0.765 / 0.470 | 0.895 / 0.660 | 0.970 / 0.850 |

At 128 evaluation simulations the 500k buffer led the control by half a point over 100 games against level 1 (97–0–3 versus 96–1–3, wins–draws–losses). On a fresh set of 50 paired openings (`--seed 1 --levels 1 --simulations 128 --games 100 --search-only`), the control scored **0.940** (94–0–6) and the larger buffer **0.905** (89–3–8). The opening seed changed between tests. Both used the same saved weights for each model.

We retain the Othello defaults of **4 blocks × 64 channels, 200k replay entries and 32 self-play simulations**. The larger buffer's narrow lead reversed on fresh openings and it used more memory. The larger network and deeper self-play searches also completed fewer games in the hour and scored lower on the primary comparison. This decision covers these three parameters at the one-hour budget with a single training seed.

## Milestones

Each milestone ends with a go/no-go check and something written up. Weeks are targets, not promises.

### M0 — Search core + Connect4 (weeks 1–2)

- Batched MCTS in MLX: PUCT selection and Gumbel root selection (sequential halving), expansion, backup.
- Connect4 environment, batched on the GPU.
- Correctness: compare search results against a Connect4 solver on a fixed position set.
- Profile, then write a fused Metal kernel for tree descent + backup (one thread per tree).
- Benchmark: simulations/sec vs mctx on CPU and vs our own pure-MLX Python-loop version.

Status (2026-11-19): search core, Connect4, PUCT, Gumbel root selection (PUCT below the root), fused descent/backup kernels, and the benchmark are done. The network, self-play, replay buffer, training step, and arena planned for M1 are also in place and train Connect4 end to end. The solver check is done (`benchmarks/solver_agreement.py`). The optional `--improved-nonroot` rule selects the legal move with the largest difference between its improved-policy probability and its current visit count divided by one plus total child visits. The default uses PUCT. Use the same switch with `benchmarks/edax_match.py` to evaluate it.

**Go/no-go:** the fused kernel gives a clear, reproducible speedup over both baselines. If it doesn't, rethink the architecture before building training on top.

**Result so far:** mixed. Clear win over pure MLX (2–4×) and over mctx on the CPU from batch 1,024 up, but far behind mctx at small batches. With a network (the case that matters for training) we were 3–5× faster than mctx on the CPU from batch 64 up, so the decision is **go**. Re-measured on 2026-11-27 on an idle Mac, the lead was 2.3–3.2× over mctx on the CPU and 3.6–4.4× over mctx on the Mac GPU through jax-metal. Tree search without a network at small batches remains much slower than mctx.

### M1 — Othello from scratch (weeks 3–6) → v0.1 release

- Bitboard Othello environment on the GPU, validated with perft and known positions.
- Configurable ResNet (board shape + action count), self-play loop, replay buffer, training loop, checkpointing.
- Arena evaluation and Elo tracking during training.
- Evaluation against Edax at fixed levels.
- Publish the curve: strength vs M5 GPU-hours.

Status (2026-11-22): the bitboard environment is done and matches the reference rules and perft to depth 9 on both CPU and GPU. The ResNet, self-play, replay buffer, training loop (`runner.py`) and checkpoint saving work for Othello. Matches against Edax at fixed levels work (`benchmarks/edax_match.py`). Runs resume from their saved state and log their scores against Edax as they train. Ratings are available from the evaluation log and optional checkpoint matches. The [two-hour comparison](#two-hour-othello-comparison) meets the playing-strength criterion for v0.1.

**Go/no-go:** an agent trained from zero on one Mac beats Edax at some non-trivial level, with the training run reproducible from a single command.

**GitHub release.** [v0.1.0](https://github.com/natsu0529/mac-alpha-zero/releases/tag/v0.1.0) distributes the code and the constant-rate Othello weights used in the README game. [Download, play and reproduce the 100-game Edax match](README.md#play-the-pretrained-othello-agent). Launch write-ups and posts are tracked in [#12](https://github.com/natsu0529/mac-alpha-zero/issues/12).

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
