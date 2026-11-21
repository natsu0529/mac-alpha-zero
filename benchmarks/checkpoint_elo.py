"""Rate Othello evaluation points against fixed opponents and optional checkpoint matches."""

import argparse
import json
from pathlib import Path

import numpy as np


def fit_elo(matches, anchor):
    players = sorted({p for left, right, _, _ in matches for p in (left, right)} - {anchor})
    if anchor not in {right for _, right, _, _ in matches}:
        raise ValueError(f"missing anchor {anchor!r} in scores")
    index = {p: i for i, p in enumerate(players)}
    design = np.zeros((len(matches), len(players)))
    points, games = np.empty(len(matches)), np.empty(len(matches))
    for i, (left, right, score, count) in enumerate(matches):
        points[i], games[i] = score, count
        if left != anchor:
            design[i, index[left]] = 1
        if right != anchor:
            design[i, index[right]] = -1
    scale = 400 / np.log(10)
    precision = (scale / 800) ** 2
    ratings = np.zeros(len(players))
    for _ in range(50):
        probability = 1 / (1 + np.exp(-np.clip(design @ ratings, -40, 40)))
        gradient = design.T @ (points - games * probability) - precision * ratings
        weight = games * probability * (1 - probability)
        hessian = design.T @ (weight[:, None] * design) + precision * np.eye(len(players))
        change = np.linalg.solve(hessian, gradient)
        ratings += change
        if np.max(np.abs(change)) < 1e-8:
            break
    return {anchor: 0.0, **dict(zip(players, ratings * scale, strict=True))}


def checkpoint_matches(rows, log_dir, games, opening_plies, seed):
    from mac_alpha_zero import arena
    from mac_alpha_zero.games.othello import Othello
    from mac_alpha_zero.network import AlphaZeroNet, network_evaluator

    if games < 2 or games % 2:
        raise ValueError("--match-games must be even and at least two")
    game = Othello()
    start = arena.random_openings(game, games // 2, opening_plies, seed)
    previous = None
    for row in rows:
        if "weights" not in row:
            raise ValueError("checkpoint matches require training with --save-eval-weights")
        net = AlphaZeroNet(game.observation_shape, game.num_actions, row["channels"], row["blocks"])
        net.load_weights(str(log_dir / row["weights"]))
        net.eval()
        agent = arena.gumbel_agent(game, network_evaluator(game, net), row["simulations"])
        name = str(row["iteration"])
        if previous:
            other_name, other_agent = previous
            result = arena.evaluate(game, agent, other_agent, games, start)
            yield name, other_name, result.wins + result.draws / 2, games
        previous = name, agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--anchor", default="Edax level 1")
    parser.add_argument(
        "--match-games", type=int, default=0, help="games per adjacent checkpoint pair"
    )
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.log.read_text().splitlines()]
    matches = [
        (
            str(row["iteration"]),
            key.removeprefix("search vs "),
            score * row["eval_games"],
            row["eval_games"],
        )
        for row in rows
        for key, score in row["scores"].items()
        if key.startswith("search vs ")
    ]
    if args.match_games:
        matches.extend(
            checkpoint_matches(
                rows, args.log.parent, args.match_games, args.opening_plies, args.seed
            )
        )
    ratings = fit_elo(matches, args.anchor)
    for row in rows:
        print(
            json.dumps(
                {
                    "iteration": row["iteration"],
                    "minutes": row["minutes"],
                    "elo": round(ratings[str(row["iteration"])], 1),
                }
            )
        )


if __name__ == "__main__":
    main()
