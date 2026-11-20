"""The self-play training loop shared by the per-game scripts in `examples/`."""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from mac_alpha_zero import arena, kernels
from mac_alpha_zero.games.base import BatchedGame
from mac_alpha_zero.gumbel import gumbel_search
from mac_alpha_zero.mcts import random_rollout_evaluator
from mac_alpha_zero.network import AlphaZeroNet, network_evaluator
from mac_alpha_zero.replay import ReplayBuffer
from mac_alpha_zero.selfplay import SelfPlay
from mac_alpha_zero.train import train_step


def add_arguments(parser: argparse.ArgumentParser, **defaults) -> None:
    """Add the training options. `defaults` overrides them per game, e.g. `games=128`."""
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--games", type=int, default=256, help="parallel self-play games")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--steps-per-iteration", type=int, default=16, help="self-play moves")
    parser.add_argument("--train-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--buffer", type=int, default=200_000)
    parser.add_argument("--eval-every", type=int, default=10, help="iterations")
    parser.add_argument("--eval-games", type=int, default=128)
    parser.add_argument("--rollout-simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", required="checkpoint" not in defaults)
    parser.set_defaults(**defaults)


def run(game: BatchedGame, args: argparse.Namespace) -> None:
    """Train by Gumbel self-play for `args.minutes`, reporting strength along the way.

    The report is the score (win 1, draw 0.5) against a random player and against plain
    MCTS with random rollouts, both with search and with the raw policy (no search).
    """
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = AlphaZeroNet(game.observation_shape, game.num_actions, args.channels, args.blocks)
    mx.eval(net.parameters())
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=1e-4)
    evaluator = network_evaluator(game, net)
    use_kernels = kernels.is_available()
    key = mx.random.key(args.seed)

    def self_play_actor(state):
        nonlocal key
        key, subkey = mx.random.split(key)
        result = gumbel_search(
            game, state, evaluator, args.simulations, key=subkey, use_kernels=use_kernels
        )
        return result.action, result.improved_policy

    def raw_policy_agent(state):
        logits, _ = net(game.observe(state))
        return mx.argmax(mx.where(game.legal_mask(state), logits, -mx.inf), axis=-1).astype(
            mx.int32
        )

    random_opponent = arena.random_agent(game, seed=args.seed + 1)
    rollout_opponent = arena.puct_agent(
        game, random_rollout_evaluator(game, seed=args.seed + 2), args.rollout_simulations
    )

    def report(iteration, elapsed, loss):
        net.eval()
        search_agent = arena.gumbel_agent(game, evaluator, args.simulations)
        n = args.eval_games
        scores = [
            arena.evaluate(game, raw_policy_agent, random_opponent, n).score,
            arena.evaluate(game, search_agent, random_opponent, n).score,
            arena.evaluate(game, raw_policy_agent, rollout_opponent, n).score,
            arena.evaluate(game, search_agent, rollout_opponent, n).score,
        ]
        print(
            f"{iteration:>5} {elapsed / 60:>7.1f} {selfplay.games_finished:>8} {loss:>7.3f}  "
            + "  ".join(f"{s:>6.2f}" for s in scores),
            flush=True,
        )

    selfplay = SelfPlay(game, args.games)
    buffer = ReplayBuffer(args.buffer)
    print(f"kernels={use_kernels} games={args.games} simulations={args.simulations}")
    print(
        " iter minutes    games    loss  score vs random    "
        f"score vs rollout MCTS ({args.rollout_simulations} sims)"
    )
    print(f"{'':>30}  policy  search  policy  search")

    train_seconds = 0.0
    iteration = 0
    loss = float("nan")
    report(iteration, train_seconds, loss)
    while train_seconds < args.minutes * 60:
        start = time.perf_counter()
        net.eval()
        for _ in range(args.steps_per_iteration):
            buffer.add(*selfplay.step(self_play_actor))
        if buffer.size >= args.batch_size:
            net.train()
            for _ in range(args.train_steps):
                loss = train_step(net, optimizer, *buffer.sample(args.batch_size, rng))
        train_seconds += time.perf_counter() - start
        iteration += 1
        if iteration % args.eval_every == 0:
            report(iteration, train_seconds, loss)

    report(iteration, train_seconds, loss)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    net.save_weights(args.checkpoint)
    print(f"saved {args.checkpoint}")
