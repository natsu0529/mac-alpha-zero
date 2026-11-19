"""Build the fixed Connect4 position set used by solver_agreement.py.

    uv run python benchmarks/make_connect4_positions.py

Positions come from uniformly random play and have 14 to 30 stones. Each stores the exact
win/draw/loss value of every legal move. Positions where every move has the same value
are dropped, since no move there can be wrong.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from mac_alpha_zero import reference
from mac_alpha_zero.games.connect4_solver import move_values

OUTPUT = Path(__file__).parent / "data" / "connect4_positions.json"


def random_position(args: tuple[int, int]) -> dict | None:
    seed, stones = args
    rng = np.random.default_rng(seed)
    game = reference.Connect4()
    state = game.initial_state()
    moves = ""
    for _ in range(stones):
        action = int(rng.choice(np.flatnonzero(game.legal_mask(state))))
        state = game.step(state, action)
        moves += str(action + 1)
        if game.terminal_value(state) is not None:
            return None
    values = move_values(state.board, state.player)
    if len(set(values.values())) == 1:
        return None
    return {"moves": moves, "values": {str(a): v for a, v in sorted(values.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    positions: dict[str, dict] = {}
    next_seed = args.seed
    with Pool() as pool:
        while len(positions) < args.count:
            jobs = [(next_seed + i, int(rng.integers(14, 31))) for i in range(256)]
            next_seed += len(jobs)
            for position in pool.imap(random_position, jobs):
                if position is not None and len(positions) < args.count:
                    positions.setdefault(position["moves"], position)
            print(f"{len(positions)} / {args.count}", flush=True)

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(list(positions.values()), separators=(",", ":")) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
