"""Train an Othello agent from scratch by Gumbel self-play.

    scripts/build_edax.sh   # once, to measure strength against Edax
    uv run python examples/train_othello.py --minutes 10
    uv run python examples/train_othello.py --minutes 60 --resume

Every few iterations the network plays a random player and Edax at fixed levels, both
with search and with the raw policy (no search). The scores go to the log next to the
checkpoint. Without Edax the opponents are a random player and plain rollout MCTS.
"""

import argparse
from pathlib import Path

from mac_alpha_zero import arena, games, runner
from mac_alpha_zero.edax import DEFAULT_PATH, edax_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    runner.add_arguments(
        parser,
        checkpoint="checkpoints/othello.safetensors",
        # Random rollouts are slow in Othello (about 60 plies each), so that baseline is
        # weak; it is only used when Edax is missing.
        rollout_simulations=8,
        eval_games=64,
        eval_every=50,
    )
    parser.add_argument("--edax", default=str(DEFAULT_PATH), help="path to the Edax binary")
    parser.add_argument("--edax-levels", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--opening-plies", type=int, default=6, help="random moves before Edax")
    args = parser.parse_args()

    game = games.Othello()
    opponents = None
    if Path(args.edax).exists():
        # Edax is deterministic, so its games start from fixed random openings.
        start = arena.random_openings(game, args.eval_games // 2, args.opening_plies, args.seed)
        opponents = {"random": (arena.random_agent(game, seed=args.seed + 1), None)}
        for level in args.edax_levels:
            opponents[f"Edax level {level}"] = (edax_agent(level, args.edax), start)
    else:
        print(f"{args.edax} not found: evaluating against rollout MCTS instead of Edax")
    runner.run(game, args, opponents)


if __name__ == "__main__":
    main()
