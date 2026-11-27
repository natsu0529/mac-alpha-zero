import numpy as np
import pytest

from mac_alpha_zero import reference
from mac_alpha_zero.reference.chess import (
    BISHOP,
    KNIGHT,
    NUM_PLANES,
    QUEEN,
    ROOK,
    action_to_uci,
    encode_move,
    from_fen,
    perft,
    to_fen,
    uci_to_action,
)

game = reference.Chess()

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
POSITION_3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
POSITION_4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
POSITION_4_MIRRORED = "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1"
POSITION_5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
POSITION_6 = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"

# Published perft counts (chessprogramming.org/Perft_Results), depth 1 first. Together
# these positions cover castling through and out of check, en passant (including the
# discovered check along the rank in position 3), promotions and underpromotions.
PERFT = {
    reference.chess.START_FEN: [20, 400, 8902],
    KIWIPETE: [48, 2039, 97862],
    POSITION_3: [14, 191, 2812, 43238],
    POSITION_4: [6, 264, 9467],
    POSITION_4_MIRRORED: [6, 264, 9467],
    POSITION_5: [44, 1486, 62379],
    POSITION_6: [46, 2079, 89890],
}


def play(*moves: str, fen: str = reference.chess.START_FEN) -> reference.ChessState:
    state = from_fen(fen)
    for move in moves:
        state = game.step(state, uci_to_action(state, move))
    return state


def protocol_perft(state, depth: int) -> int:
    """Perft through the `Game` protocol, to check the action encoding and `step`."""
    if depth == 0:
        return 1
    return sum(
        protocol_perft(game.step(state, a), depth - 1)
        for a in np.flatnonzero(game.legal_mask(state))
    )


@pytest.mark.parametrize("fen", PERFT)
def test_perft(fen):
    state = from_fen(fen)
    for depth, expected in enumerate(PERFT[fen], start=1):
        assert perft(state, depth) == expected


@pytest.mark.parametrize("fen", [reference.chess.START_FEN, KIWIPETE, POSITION_3, POSITION_4])
def test_protocol_matches_perft(fen):
    assert protocol_perft(from_fen(fen), 2) == PERFT[fen][1]


@pytest.mark.parametrize("fen", PERFT)
def test_fen_round_trip(fen):
    assert to_fen(from_fen(fen)) == fen


@pytest.mark.parametrize("fen", PERFT)
def test_uci_round_trip(fen):
    state = from_fen(fen)
    for action in np.flatnonzero(game.legal_mask(state)):
        assert uci_to_action(state, action_to_uci(state, action)) == action


def test_initial_state():
    state = game.initial_state()
    assert game.num_actions == 64 * NUM_PLANES == 4672
    assert game.terminal_value(state) is None
    assert game.legal_mask(state).sum() == 20
    assert to_fen(state) == reference.chess.START_FEN


def test_actions_are_seen_from_the_player_to_move():
    white = game.initial_state()
    black = play("e2e4")
    assert to_fen(black) == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    # Black's ranks are mirrored, so its moves use the same actions as White's.
    assert uci_to_action(black, "e7e5") == uci_to_action(white, "e2e4")
    assert uci_to_action(black, "g8f6") == uci_to_action(white, "g1f3")
    assert np.array_equal(game.legal_mask(black), game.legal_mask(white))


def test_promotions_have_distinct_actions():
    state = from_fen("1r5k/P7/8/8/8/8/8/K7 w - - 0 1")
    promotions = ["a7a8q", "a7a8r", "a7a8b", "a7a8n", "a7b8q", "a7b8r", "a7b8b", "a7b8n"]
    actions = [uci_to_action(state, move) for move in promotions]
    assert len(set(actions)) == 8
    # A queen promotion is a one-square queen-like move; underpromotions have their own planes.
    a7 = 6 * 8
    assert actions[0] == a7 * NUM_PLANES + 0
    assert actions[4] == a7 * NUM_PLANES + 7 * 1
    for move, plane in [("a7a8n", 67), ("a7a8b", 68), ("a7a8r", 69), ("a7b8n", 70)]:
        assert uci_to_action(state, move) == a7 * NUM_PLANES + plane
    after = game.step(state, uci_to_action(state, "a7b8n"))
    assert after.board[7, 1] == KNIGHT and after.board[6, 0] == 0
    assert encode_move((a7, 56, BISHOP), 1) % NUM_PLANES == 68
    assert encode_move((a7, 56, ROOK), 1) % NUM_PLANES == 69
    assert encode_move((a7, 56, QUEEN), 1) % NUM_PLANES == 0


def test_castling_moves_the_rook_and_removes_rights():
    state = play("e1g1", fen=KIWIPETE)
    assert to_fen(state).split()[:3] == [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R4RK1",
        "b",
        "kq",
    ]
    # Capturing a rook on its corner removes the other side's right too.
    state = play("a1a8", fen="r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert to_fen(state).split()[2] == "Kk"


def test_en_passant():
    state = play("e2e4", "a7a6", "e4e5", "d7d5")
    assert to_fen(state).split()[3] == "d6"
    state = game.step(state, uci_to_action(state, "e5d6"))
    assert state.board[4, 3] == 0  # The black pawn on d5 is gone.
    with pytest.raises(ValueError):
        uci_to_action(play("e2e4", "a7a6", "e4e5", "d7d5", "a2a3", "a6a5"), "e5d6")


def test_illegal_actions_raise():
    state = game.initial_state()
    with pytest.raises(ValueError):
        game.step(state, uci_to_action(state, "e2e4") + 1)
    with pytest.raises(ValueError):
        uci_to_action(state, "e2e5")


def test_checkmate():
    state = play("f2f3", "e7e5", "g2g4", "d8h4")
    assert game.terminal_value(state) == -1.0
    assert not game.legal_mask(state).any()


def test_stalemate():
    state = from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert game.terminal_value(state) == 0.0
    assert not game.legal_mask(state).any()


def test_threefold_repetition():
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    state = play(*shuffle, *shuffle[:3])
    assert game.terminal_value(state) is None
    state = game.step(state, uci_to_action(state, shuffle[3]))
    # The starting position has now occurred three times.
    assert game.terminal_value(state) == 0.0
    assert not game.legal_mask(state).any()


def test_repetition_ignores_an_unusable_en_passant_square():
    # After 1. e4 the FEN records e3, but no black pawn can capture there, so the
    # position with Black to move repeats after 3. Ng1 and 5. Ng1.
    state = play("e2e4", "g8f6", "g1f3", "f6g8", "f3g1", "g8f6", "g1f3", "f6g8")
    assert game.terminal_value(state) is None
    state = game.step(state, uci_to_action(state, "f3g1"))
    assert game.terminal_value(state) == 0.0


def test_fifty_move_rule():
    state = from_fen("8/8/8/4k3/8/8/8/R3K3 w - - 99 60")
    assert game.terminal_value(state) is None
    assert game.terminal_value(play("a1a2", fen=to_fen(state))) == 0.0
    # A checkmate on the hundredth ply still wins.
    state = play("a1a8", fen="7k/8/6K1/8/8/8/8/R7 w - - 99 80")
    assert state.halfmove_clock == 100
    assert game.terminal_value(state) == -1.0


@pytest.mark.parametrize(
    ("fen", "drawn"),
    [
        ("8/8/8/4k3/8/8/8/4K3 w - - 0 1", True),
        ("8/8/8/4k3/8/8/8/2B1K3 w - - 0 1", True),
        ("8/8/8/4k3/8/8/8/1N2K3 w - - 0 1", True),
        ("5b2/8/8/4k3/8/8/8/2B1K3 w - - 0 1", True),  # Bishops on squares of one colour.
        ("2b5/8/8/4k3/8/8/8/2B1K3 w - - 0 1", False),
        ("1n6/8/8/4k3/8/8/8/1N2K3 w - - 0 1", False),
        ("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1", False),
    ],
)
def test_insufficient_material(fen, drawn):
    assert game.terminal_value(from_fen(fen)) == (0.0 if drawn else None)


def test_mcts_finds_mate_in_one():
    state = from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")

    def evaluator(_):
        return np.ones(game.num_actions), 0.0

    result = reference.run_mcts(game, state, evaluator, num_simulations=64)
    assert action_to_uci(state, int(np.argmax(result.visit_counts))) == "a1a8"
