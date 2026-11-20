"""Train a Connect4 agent from scratch by Gumbel self-play.

    uv run python examples/train_connect4.py --minutes 10

Prints, every few iterations, how the network fares against a random player and against
plain MCTS with random rollouts, both with search and with the raw policy (no search).
"""

import argparse

from mac_alpha_zero import games, runner


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    runner.add_arguments(parser, checkpoint="checkpoints/connect4.safetensors")
    runner.run(games.Connect4(), parser.parse_args())


if __name__ == "__main__":
    main()
