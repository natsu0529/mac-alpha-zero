import argparse
from pathlib import Path

import numpy as np

from mac_alpha_zero import arena, games, reference
from mac_alpha_zero.network import AlphaZeroNet, network_evaluator
from mac_alpha_zero.reference.othello import PASS


def square_name(action):
    row, column = divmod(int(action), 8)
    return f"{'abcdefgh'[column]}{row + 1}"


def play(agent_move, human, read=input):
    game = reference.Othello()
    state = game.initial_state()
    while True:
        print("  a b c d e f g h")
        for row, cells in enumerate(state.board, 1):
            print(row, " ".join({1: "X", -1: "O", 0: "."}[disc] for disc in cells))
        result = game.terminal_value(state)
        if result is not None:
            black, white = (np.count_nonzero(state.board == color) for color in (1, -1))
            outcome = (
                "Draw"
                if result == 0
                else ("You win" if result * state.player == human else "Agent wins")
            )
            print(f"{outcome}. Black {black}, white {white}.")
            return
        legal = np.flatnonzero(game.legal_mask(state))
        side = "Black" if state.player == 1 else "White"
        if legal[0] == PASS:
            action = PASS
            print(f"{side} passes.")
        elif state.player == human:
            choices = {square_name(action): action for action in legal}
            while True:
                try:
                    move = (
                        read(f"{side} to play ({', '.join(choices)}), or q to quit: ")
                        .strip()
                        .lower()
                    )
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                if move in ("q", "quit"):
                    return
                if move in choices:
                    action = choices[move]
                    break
                print("Choose one of the listed squares.")
        else:
            action = agent_move(state)
            print(f"{side} plays {square_name(action)}.")
        state = game.step(state, action)


def main():
    parser = argparse.ArgumentParser(description="Play Othello against a saved checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--color", choices=("black", "white"), default="black", help="your discs")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--improved-nonroot", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.simulations < 1:
        parser.error("--simulations must be positive")
    game = games.Othello()
    net = AlphaZeroNet(game.observation_shape, game.num_actions, args.channels, args.blocks)
    net.load_weights(str(args.checkpoint))
    net.eval()
    agent = arena.gumbel_agent(
        game, network_evaluator(game, net), args.simulations, improved_nonroot=args.improved_nonroot
    )
    play(
        lambda state: agent(game.from_reference([state])).item(), 1 if args.color == "black" else -1
    )


if __name__ == "__main__":
    main()
