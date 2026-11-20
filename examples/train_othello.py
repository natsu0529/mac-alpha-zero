"""Train an Othello agent from scratch by Gumbel self-play.

    uv run python examples/train_othello.py --minutes 10

Prints, every few iterations, how the network fares against a random player and against
plain MCTS with random rollouts, both with search and with the raw policy (no search).
"""

import argparse

from mac_alpha_zero import games, runner


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    runner.add_arguments(
        parser,
        checkpoint="checkpoints/othello.safetensors",
        # Random rollouts are slow in Othello (about 60 plies each), so the baseline is weaker
        # and evaluated less often than in Connect4.
        rollout_simulations=8,
        eval_games=64,
        eval_every=25,
    )
    runner.run(games.Othello(), parser.parse_args())


if __name__ == "__main__":
    main()
