"""Record an Othello match against Edax or render a saved move list as a GIF."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mac_alpha_zero import arena, games, reference
from mac_alpha_zero.edax import DEFAULT_PATH, edax_agent
from mac_alpha_zero.network import AlphaZeroNet, network_evaluator
from mac_alpha_zero.reference.othello import PASS


def record(args):
    game = games.Othello()
    net = AlphaZeroNet(game.observation_shape, game.num_actions, args.channels, args.blocks)
    net.load_weights(str(args.checkpoint))
    net.eval()
    agent = arena.gumbel_agent(game, network_evaluator(game, net), args.simulations)
    edax = edax_agent(args.level, args.edax)
    opening = arena.random_agent(game, args.seed)
    black, white = (agent, edax) if args.color == "black" else (edax, agent)
    moves = []
    state = game.init(1)
    while not game.terminal(state)[0].item():
        player = black if len(moves) % 2 == 0 else white
        action = (opening if len(moves) < args.opening_plies else player)(state)
        moves.append(action.item())
        state = game.step(state, action)
    board = np.array(game.boards(state))[0]
    players = ["Agent", f"Edax {args.level}"]
    if args.color == "white":
        players.reverse()
    data = {
        "black": players[0],
        "white": players[1],
        "moves": moves,
        "black_discs": int((board == 1).sum()),
        "white_discs": int((board == -1).sum()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "simulations": args.simulations,
        "seed": args.seed,
        "opening_plies": args.opening_plies,
    }
    args.output.write_text(json.dumps(data, indent=2) + "\n")


def frame(state, players, move, finished):
    cell = 30
    margin = 8
    size = 8 * cell + 2 * margin
    img = Image.new("RGB", (size, size + 40), "#f6f4ef")
    draw = ImageDraw.Draw(img)
    for r, row in enumerate(state.board):
        for c, disc in enumerate(row):
            x, y = margin + c * cell, margin + r * cell
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill="#277755", outline="#1c5b41")
            if disc:
                draw.ellipse(
                    (x + 3, y + 3, x + cell - 4, y + cell - 4),
                    fill="#17211f" if disc == 1 else "#fffdf3",
                    outline="#18362b",
                )
            if move == r * 8 + c:
                draw.ellipse((x + 12, y + 12, x + 17, y + 17), fill="#df9c38")
    counts = [int((state.board == sign).sum()) for sign in (1, -1)]
    title = f"Black: {players[0]} {counts[0]}    White: {players[1]} {counts[1]}"
    status = "Pass" if move == PASS else ""
    if finished:
        status = "Draw" if counts[0] == counts[1] else f"{players[int(counts[1] > counts[0])]} wins"
    font = ImageFont.load_default(size=14)
    draw.text((margin, size), title, fill="#17211f", font=font)
    draw.text((margin, size + 18), status, fill="#17211f", font=font)
    return img


def render(args):
    data = json.loads(args.record.read_text())
    game = reference.Othello()
    state = game.initial_state()
    players = [data["black"], data["white"]]
    frames = [frame(state, players, None, False)]
    for move in data["moves"]:
        state = game.step(state, move)
        frames.append(frame(state, players, move, game.terminal_value(state) is not None))
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=[200] * (len(frames) - 1) + [1800],
        loop=0,
        optimize=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    match = commands.add_parser("record")
    match.add_argument("checkpoint", type=Path)
    match.add_argument("output", type=Path)
    match.add_argument("--level", type=int, default=1)
    match.add_argument("--edax", type=Path, default=DEFAULT_PATH)
    match.add_argument("--simulations", type=int, default=32)
    match.add_argument("--channels", type=int, default=64)
    match.add_argument("--blocks", type=int, default=4)
    match.add_argument("--color", choices=("black", "white"), default="black")
    match.add_argument("--seed", type=int, default=0)
    match.add_argument("--opening-plies", type=int, default=6)
    match.set_defaults(run=record)
    gif = commands.add_parser("render")
    gif.add_argument("record", type=Path)
    gif.add_argument("output", type=Path)
    gif.set_defaults(run=render)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
