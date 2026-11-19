"""How often agents pick a game-theoretically optimal move in Connect4.

    uv run python benchmarks/solver_agreement.py --checkpoint checkpoints/connect4.safetensors

Uses the solved positions in benchmarks/data/connect4_positions.json (random-play
positions with 14 to 30 stones where at least one move is worse than the best). A move
counts as optimal if it keeps the best achievable win/draw/loss result. "Quiet" positions
are those where the side to move has no immediate win.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mac_alpha_zero import arena, games, reference
from mac_alpha_zero.mcts import random_rollout_evaluator
from mac_alpha_zero.network import AlphaZeroNet, network_evaluator

POSITIONS = Path(__file__).parent / "data" / "connect4_positions.json"


def load_positions():
    ref = reference.Connect4()
    states, optimal, quiet = [], [], []
    for entry in json.loads(POSITIONS.read_text()):
        state = ref.from_moves(entry["moves"])
        values = {int(a): v for a, v in entry["values"].items()}
        best = max(values.values())
        states.append(state)
        optimal.append([values.get(a) == best for a in range(ref.num_actions)])
        quiet.append(not any(ref.step(state, a).winner for a in values))
    return states, np.array(optimal), np.array(quiet)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", help="network weights from examples/train_connect4.py")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--simulations", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--rollout-simulations", type=int, nargs="+", default=[100, 400])
    args = parser.parse_args()

    game = games.Connect4()
    states, optimal, quiet = load_positions()
    root = game.from_reference(states)

    agents = {"random": arena.random_agent(game, seed=0)}
    for sims in args.rollout_simulations:
        agents[f"rollout MCTS, {sims} sims"] = arena.puct_agent(
            game, random_rollout_evaluator(game, seed=1), sims
        )
    if args.checkpoint:
        net = AlphaZeroNet(game.observation_shape, game.num_actions, args.channels, args.blocks)
        net.load_weights(args.checkpoint)
        net.eval()

        def raw_policy(state):
            logits, _ = net(game.observe(state))
            masked = mx.where(game.legal_mask(state), logits, -mx.inf)
            return mx.argmax(masked, axis=-1).astype(mx.int32)

        agents["network, policy only"] = raw_policy
        for sims in args.simulations:
            agents[f"network + Gumbel search, {sims} sims"] = arena.gumbel_agent(
                game, network_evaluator(game, net), sims
            )

    print(f"{len(states)} positions, {int(quiet.sum())} quiet")
    print(f"{'agent':<38} {'optimal':>8} {'quiet only':>11}")
    for name, agent in agents.items():
        actions = np.array(agent(root))
        correct = optimal[np.arange(len(states)), actions]
        print(f"{name:<38} {correct.mean():>8.3f} {correct[quiet].mean():>11.3f}")


if __name__ == "__main__":
    main()
