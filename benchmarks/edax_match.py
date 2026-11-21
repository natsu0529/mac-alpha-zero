"""Play Othello agents against Edax at fixed search levels.

    scripts/build_edax.sh
    uv run python benchmarks/edax_match.py --checkpoint checkpoints/othello.safetensors

Edax plays without its opening book and is deterministic, so every game starts from a
different random opening (`--opening-plies` random moves), and each opening is played
twice, once with each side to move. The score is the mean result over all games (win 1,
draw 0.5). Without a checkpoint only the baselines play.
"""

import argparse

import mlx.core as mx

from mac_alpha_zero import arena, games
from mac_alpha_zero.edax import DEFAULT_PATH, edax_agent
from mac_alpha_zero.network import AlphaZeroNet, network_evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", help="network weights from examples/train_othello.py")
    parser.add_argument(
        "--search-only", action="store_true", help="only evaluate the checkpoint with search"
    )
    parser.add_argument("--improved-nonroot", action="store_true")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--simulations", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edax", default=str(DEFAULT_PATH), help="path to the Edax binary")
    args = parser.parse_args()

    if args.search_only and not args.checkpoint:
        parser.error("--search-only requires --checkpoint")
    if args.games < 2 or args.games % 2:
        parser.error("--games must be even and at least two for paired openings")

    game = games.Othello()
    start = arena.random_openings(game, args.games // 2, args.opening_plies, args.seed)

    agents = {} if args.search_only else {"random": arena.random_agent(game, seed=args.seed + 1)}
    if args.checkpoint:
        net = AlphaZeroNet(game.observation_shape, game.num_actions, args.channels, args.blocks)
        net.load_weights(args.checkpoint)
        net.eval()

        def raw_policy(state):
            logits, _ = net(game.observe(state))
            masked = mx.where(game.legal_mask(state), logits, -mx.inf)
            return mx.argmax(masked, axis=-1).astype(mx.int32)

        if not args.search_only:
            agents["network, policy only"] = raw_policy
        for sims in args.simulations:
            agents[f"network + Gumbel search, {sims} sims"] = arena.gumbel_agent(
                game, network_evaluator(game, net), sims, improved_nonroot=args.improved_nonroot
            )

    print(f"{args.games} games per cell, {args.opening_plies} random opening plies")
    print(f"{'agent':<38}" + "".join(f"{'Edax level ' + str(level):>18}" for level in args.levels))
    for name, agent in agents.items():
        cells = []
        for level in args.levels:
            result = arena.evaluate(game, agent, edax_agent(level, args.edax), args.games, start)
            cells.append(f"{result.score:.3f} ({result.wins}-{result.draws}-{result.losses})")
        print(f"{name:<38}" + "".join(f"{cell:>18}" for cell in cells), flush=True)


if __name__ == "__main__":
    main()
