"""The self-play training loop shared by the per-game scripts in `examples/`."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from mac_alpha_zero import arena, kernels
from mac_alpha_zero.games.base import BatchedGame, State
from mac_alpha_zero.gumbel import gumbel_search
from mac_alpha_zero.mcts import random_rollout_evaluator
from mac_alpha_zero.network import AlphaZeroNet, network_evaluator
from mac_alpha_zero.replay import ReplayBuffer
from mac_alpha_zero.selfplay import SelfPlay
from mac_alpha_zero.train import train_step

# Evaluation opponents by name: the agent, and optionally the positions the games start
# from (`eval_games // 2` of them, each played with both sides to move).
Opponents = dict[str, tuple[arena.Agent, State | None]]


def add_arguments(parser: argparse.ArgumentParser, **defaults) -> None:
    """Add the training options. `defaults` overrides them per game, e.g. `games=128`."""
    parser.add_argument("--minutes", type=float, default=10.0, help="training time of this run")
    parser.add_argument("--games", type=int, default=256, help="parallel self-play games")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--steps-per-iteration", type=int, default=16, help="self-play moves")
    parser.add_argument("--train-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--buffer", type=int, default=200_000)
    parser.add_argument(
        "--symmetries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="train on a random symmetric copy of each sample, if the game defines them",
    )
    parser.add_argument("--eval-every", type=int, default=10, help="iterations")
    parser.add_argument("--eval-games", type=int, default=128)
    parser.add_argument("--rollout-simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        required="checkpoint" not in defaults,
        help="network weights; the training state and the log are written next to it",
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue from the state next to --checkpoint"
    )
    parser.set_defaults(**defaults)


def default_opponents(game: BatchedGame, args: argparse.Namespace) -> Opponents:
    """A random player and plain MCTS with random rollouts."""
    rollout = arena.puct_agent(
        game, random_rollout_evaluator(game, seed=args.seed + 2), args.rollout_simulations
    )
    return {
        "random": (arena.random_agent(game, seed=args.seed + 1), None),
        f"rollout MCTS {args.rollout_simulations}": (rollout, None),
    }


def run(game: BatchedGame, args: argparse.Namespace, opponents: Opponents | None = None) -> None:
    """Train by Gumbel self-play for `args.minutes`, measuring strength along the way.

    Every `args.eval_every` iterations the network plays each of `opponents` (default:
    `default_opponents`) with the raw policy and with search; the scores (win 1, draw
    0.5) are printed and appended to the log, and the training state is saved.

    Files, for `--checkpoint x.safetensors`: the weights, `x.optimizer.safetensors`,
    `x.buffer.npz`, `x.json` (counters) and `x.log.jsonl` (one line per evaluation).
    With `--resume` the run continues from them. Games in progress and the random
    streams are not saved, so a resumed run is not identical to an uninterrupted one.
    """
    if opponents is None:
        opponents = default_opponents(game, args)
    weights = Path(args.checkpoint)
    weights.parent.mkdir(parents=True, exist_ok=True)
    optimizer_path = weights.with_suffix(".optimizer.safetensors")
    buffer_path = weights.with_suffix(".buffer.npz")
    counters_path = weights.with_suffix(".json")
    log_path = weights.with_suffix(".log.jsonl")

    net = AlphaZeroNet(game.observation_shape, game.num_actions, args.channels, args.blocks)
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=1e-4)
    optimizer.init(net.trainable_parameters())
    symmetries = getattr(game, "symmetries", None) if args.symmetries else None
    buffer = ReplayBuffer(args.buffer, symmetries() if symmetries else None)
    counters = {"iteration": 0, "seconds": 0.0, "games": 0, "loss": float("nan")}
    if args.resume:
        net.load_weights(str(weights))
        saved = dict(mx.load(str(optimizer_path)))
        scalars = {k: v.item() for k, v in saved.items() if k in ("step", "learning_rate")}
        optimizer.state = tree_unflatten(list(saved.items()))
        buffer.load(str(buffer_path))
        counters = json.loads(counters_path.read_text())
        print(f"resumed {weights} at iteration {counters['iteration']} (step {scalars['step']})")
    elif counters_path.exists():
        raise SystemExit(f"{counters_path} exists: pass --resume or choose another --checkpoint")
    mx.eval(net.parameters(), optimizer.state)

    # A resumed run must not replay the random streams of the run it continues.
    seed = args.seed + counters["iteration"]
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    key = mx.random.key(seed)
    evaluator = network_evaluator(game, net)
    use_kernels = kernels.is_available()

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

    def save():
        net.save_weights(str(weights))
        mx.save_safetensors(str(optimizer_path), dict(tree_flatten(optimizer.state)))
        buffer.save(str(buffer_path))
        counters_path.write_text(json.dumps(counters))

    def report():
        net.eval()
        search_agent = arena.gumbel_agent(game, evaluator, args.simulations)
        scores = {}
        for name, (opponent, start) in opponents.items():
            for kind, agent in (("policy", raw_policy_agent), ("search", search_agent)):
                result = arena.evaluate(game, agent, opponent, args.eval_games, start)
                scores[f"{kind} vs {name}"] = result.score
        record = {
            "iteration": counters["iteration"],
            "minutes": counters["seconds"] / 60,
            "games": counters["games"],
            "loss": counters["loss"],
            "simulations": args.simulations,
            "eval_games": args.eval_games,
            "scores": scores,
        }
        with log_path.open("a") as log:
            log.write(json.dumps(record) + "\n")
        print(
            f"{record['iteration']:>5} {record['minutes']:>7.1f} {record['games']:>8} "
            f"{record['loss']:>7.3f}  " + "  ".join(f"{s:>6.2f}" for s in scores.values()),
            flush=True,
        )

    selfplay = SelfPlay(game, args.games)
    print(
        f"kernels={use_kernels} games={args.games} simulations={args.simulations} "
        f"symmetries={buffer.symmetries is not None}"
    )
    print("score columns (policy = no search, search = Gumbel search):")
    for name in opponents:
        print(f"  policy vs {name}, search vs {name}")
    print(" iter minutes    games    loss  scores")

    if counters["iteration"] == 0:
        report()
    deadline = counters["seconds"] + args.minutes * 60
    try:
        while counters["seconds"] < deadline:
            start = time.perf_counter()
            finished = selfplay.games_finished
            net.eval()
            for _ in range(args.steps_per_iteration):
                buffer.add(*selfplay.step(self_play_actor))
            if buffer.size >= args.batch_size:
                net.train()
                for _ in range(args.train_steps):
                    loss = train_step(net, optimizer, *buffer.sample(args.batch_size, rng))
                counters["loss"] = float(loss)
            counters["seconds"] += time.perf_counter() - start
            counters["games"] += selfplay.games_finished - finished
            counters["iteration"] += 1
            if counters["iteration"] % args.eval_every == 0:
                report()
                save()
    except KeyboardInterrupt:
        print("interrupted: saving")
        save()
        raise
    if counters["iteration"] % args.eval_every != 0:
        report()
    save()
    print(f"saved {weights}")
