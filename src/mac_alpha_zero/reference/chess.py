"""Chess rules: the oracle for a future fast move generator.

Squares are numbered `rank * 8 + file`, with a1 = 0, h1 = 7 and h8 = 63. Pieces are
signed integers: positive for White, negative for Black.
"""

from dataclasses import dataclass

import numpy as np

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
WHITE, BLACK = 1, -1

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# The action encoding follows AlphaZero: 73 move types ("planes") from each of the 64
# squares. Squares and directions are seen from the player to move, so Black's board
# is mirrored rank by rank. Planes 0-55 are queen-like moves (8 directions x distance
# 1-7, which includes promotion to a queen), 56-63 knight moves, and 64-72
# underpromotions (file change -1, 0, +1 x knight, bishop, rook).
NUM_PLANES = 73
# (rank step, file step): N, NE, E, SE, S, SW, W, NW.
_QUEEN_DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
_ROOK_DIRECTIONS = _QUEEN_DIRECTIONS[0::2]
_BISHOP_DIRECTIONS = _QUEEN_DIRECTIONS[1::2]
_KNIGHT_STEPS = ((2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1))
_UNDERPROMOTIONS = (KNIGHT, BISHOP, ROOK)

# Castling rights in FEN order: White O-O, White O-O-O, Black O-O, Black O-O-O. Each
# entry: king from, king to, rook from, rook to, squares that must be empty, and
# squares the king starts on, passes or lands on, none of which may be attacked.
_CASTLES = (
    (4, 6, 7, 5, (5, 6), (4, 5, 6)),
    (4, 2, 0, 3, (1, 2, 3), (4, 3, 2)),
    (60, 62, 63, 61, (61, 62), (60, 61, 62)),
    (60, 58, 56, 59, (57, 58, 59), (60, 59, 58)),
)

_PIECE_LETTERS = {PAWN: "p", KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q", KING: "k"}
_PIECE_CODES = {letter: piece for piece, letter in _PIECE_LETTERS.items()}

# A move is (from square, to square, promotion piece or 0).
Move = tuple[int, int, int]


@dataclass(frozen=True, eq=False)
class ChessState:
    # board[rank, file]; rank 0 is White's back rank.
    board: np.ndarray
    player: int
    castling: tuple[bool, bool, bool, bool]
    # The square a pawn skipped with a double step on the previous move, as in FEN.
    en_passant: int | None
    halfmove_clock: int
    fullmove: int
    # Repetition keys of the earlier positions since the last capture or pawn move.
    history: tuple = ()


class Chess:
    """Chess with draws by stalemate, threefold repetition, the 50-move rule, and
    insufficient material.

    Draws that a player could claim are applied automatically: the game ends when a
    position occurs for the third time or 100 plies pass without a capture or pawn move.
    A checkmate on that ply still counts as a win. A finished game has no legal actions.
    """

    num_actions = 64 * NUM_PLANES

    def initial_state(self) -> ChessState:
        return from_fen(START_FEN)

    def legal_mask(self, state: ChessState) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=bool)
        mask[list(_actions(state))] = True
        return mask

    def step(self, state: ChessState, action: int) -> ChessState:
        move = _actions(state).get(int(action))
        if move is None:
            raise ValueError(f"illegal action {action}")
        return _play(state, move)

    def terminal_value(self, state: ChessState) -> float | None:
        if not _legal_moves(state):
            board = state.board.ravel().tolist()
            in_check = _attacked(board, board.index(state.player * KING), -state.player)
            return -1.0 if in_check else 0.0
        return 0.0 if _drawn_by_rule(state) else None


def encode_move(move: Move, player: int) -> int:
    """The action index of `move` played by `player`."""
    start, end, promotion = (_relative(move[0], player), _relative(move[1], player), move[2])
    rank_step, file_step = end // 8 - start // 8, end % 8 - start % 8
    if promotion in _UNDERPROMOTIONS:
        plane = 64 + 3 * (file_step + 1) + _UNDERPROMOTIONS.index(promotion)
    elif (rank_step, file_step) in _KNIGHT_STEPS:
        plane = 56 + _KNIGHT_STEPS.index((rank_step, file_step))
    else:
        distance = max(abs(rank_step), abs(file_step))
        direction = _QUEEN_DIRECTIONS.index((rank_step // distance, file_step // distance))
        plane = 7 * direction + distance - 1
    return start * NUM_PLANES + plane


def action_to_uci(state: ChessState, action: int) -> str:
    """The legal action as a UCI move string such as `e2e4` or `e7e8q`."""
    move = _actions(state).get(int(action))
    if move is None:
        raise ValueError(f"illegal action {action}")
    start, end, promotion = move
    suffix = _PIECE_LETTERS[promotion] if promotion else ""
    return _square_name(start) + _square_name(end) + suffix


def uci_to_action(state: ChessState, uci: str) -> int:
    """The action index of a legal UCI move string."""
    promotion = _PIECE_CODES[uci[4]] if len(uci) == 5 else 0
    move = (_square_index(uci[:2]), _square_index(uci[2:4]), promotion)
    if move not in _actions(state).values():
        raise ValueError(f"illegal move {uci}")
    return encode_move(move, state.player)


def from_fen(fen: str) -> ChessState:
    """A state from Forsyth-Edwards Notation. The move counters are optional."""
    fields = fen.split()
    placement, side, castling, en_passant = fields[:4]
    halfmove, fullmove = (int(fields[4]), int(fields[5])) if len(fields) >= 6 else (0, 1)
    rows = placement.split("/")
    if len(rows) != 8:
        raise ValueError(f"expected 8 ranks in {placement!r}")
    board = np.zeros((8, 8), dtype=np.int8)
    for row, text in enumerate(rows):
        rank, file = 7 - row, 0
        for ch in text:
            if ch.isdigit():
                file += int(ch)
            else:
                board[rank, file] = _PIECE_CODES[ch.lower()] * (WHITE if ch.isupper() else BLACK)
                file += 1
        if file != 8:
            raise ValueError(f"expected 8 files in {text!r}")
    return ChessState(
        board=board,
        player=WHITE if side == "w" else BLACK,
        castling=tuple(right in castling for right in "KQkq"),
        en_passant=None if en_passant == "-" else _square_index(en_passant),
        halfmove_clock=halfmove,
        fullmove=fullmove,
    )


def to_fen(state: ChessState) -> str:
    rows = []
    for rank in range(7, -1, -1):
        text, empty = "", 0
        for piece in state.board[rank].tolist():
            if piece == 0:
                empty += 1
                continue
            letter = _PIECE_LETTERS[abs(piece)]
            text += (str(empty) if empty else "") + (letter.upper() if piece > 0 else letter)
            empty = 0
        rows.append(text + (str(empty) if empty else ""))
    castling = "".join(right for right, ok in zip("KQkq", state.castling, strict=True) if ok) or "-"
    en_passant = "-" if state.en_passant is None else _square_name(state.en_passant)
    side = "w" if state.player == WHITE else "b"
    fields = ["/".join(rows), side, castling, en_passant, state.halfmove_clock, state.fullmove]
    return " ".join(str(field) for field in fields)


def perft(state: ChessState, depth: int) -> int:
    """The number of legal move sequences of length `depth`. Like every perft, this
    counts moves past draws by rule, so it checks move generation alone."""
    moves = _legal_moves(state)
    if depth <= 1:
        return len(moves) if depth == 1 else 1
    return sum(perft(_play(state, move), depth - 1) for move in moves)


def _actions(state: ChessState) -> dict[int, Move]:
    """Legal moves keyed by action index. Empty once the game has ended."""
    if _drawn_by_rule(state):
        return {}
    return {encode_move(move, state.player): move for move in _legal_moves(state)}


def _legal_moves(state: ChessState) -> list[Move]:
    board = state.board.ravel().tolist()
    moves = []
    for move in _pseudo_legal_moves(board, state.player, state.castling, state.en_passant):
        after = _make(board, move, state.player)
        if not _attacked(after, after.index(state.player * KING), -state.player):
            moves.append(move)
    return moves


def _pseudo_legal_moves(
    board: list[int], player: int, castling: tuple[bool, ...], en_passant: int | None
) -> list[Move]:
    """Moves that obey how each piece moves, possibly leaving the own king in check."""
    moves = []
    for square in range(64):
        piece = board[square] * player
        if piece <= 0:
            continue
        rank, file = divmod(square, 8)
        if piece == PAWN:
            ahead = rank + player
            targets = []
            if board[ahead * 8 + file] == 0:
                targets.append(ahead * 8 + file)
                start_rank = 1 if player == WHITE else 6
                if rank == start_rank and board[(ahead + player) * 8 + file] == 0:
                    targets.append((ahead + player) * 8 + file)
            for side in (file - 1, file + 1):
                target = ahead * 8 + side
                if 0 <= side < 8 and (board[target] * player < 0 or target == en_passant):
                    targets.append(target)
            for target in targets:
                if target // 8 in (0, 7):
                    moves.extend((square, target, p) for p in (QUEEN, *_UNDERPROMOTIONS))
                else:
                    moves.append((square, target, 0))
        elif piece in (KNIGHT, KING):
            for rank_step, file_step in _KNIGHT_STEPS if piece == KNIGHT else _QUEEN_DIRECTIONS:
                r, f = rank + rank_step, file + file_step
                if 0 <= r < 8 and 0 <= f < 8 and board[r * 8 + f] * player <= 0:
                    moves.append((square, r * 8 + f, 0))
        else:
            directions = {BISHOP: _BISHOP_DIRECTIONS, ROOK: _ROOK_DIRECTIONS}.get(
                piece, _QUEEN_DIRECTIONS
            )
            for rank_step, file_step in directions:
                r, f = rank + rank_step, file + file_step
                while 0 <= r < 8 and 0 <= f < 8 and board[r * 8 + f] * player <= 0:
                    moves.append((square, r * 8 + f, 0))
                    if board[r * 8 + f]:
                        break
                    r, f = r + rank_step, f + file_step

    rights = castling[:2] if player == WHITE else castling[2:]
    castles = _CASTLES[:2] if player == WHITE else _CASTLES[2:]
    for right, (king_from, king_to, rook_from, _, empty, safe) in zip(rights, castles, strict=True):
        if (
            right
            and board[king_from] == player * KING
            and board[rook_from] == player * ROOK
            and all(board[s] == 0 for s in empty)
            and not any(_attacked(board, s, -player) for s in safe)
        ):
            moves.append((king_from, king_to, 0))
    return moves


def _make(board: list[int], move: Move, player: int) -> list[int]:
    """The board after `move`, without checking that it is legal."""
    start, end, promotion = move
    board = list(board)
    piece = board[start]
    if abs(piece) == PAWN and (end - start) % 8 != 0 and board[end] == 0:
        board[end - 8 * player] = 0  # En passant: remove the pawn that was passed.
    if abs(piece) == KING and abs(end - start) == 2:
        for king_from, king_to, rook_from, rook_to, _, _ in _CASTLES:
            if (king_from, king_to) == (start, end):
                board[rook_to], board[rook_from] = board[rook_from], 0
    board[start] = 0
    board[end] = promotion * player if promotion else piece
    return board


def _play(state: ChessState, move: Move) -> ChessState:
    board = state.board.ravel().tolist()
    start, end, _ = move
    piece = abs(board[start])
    irreversible = piece == PAWN or board[end] != 0
    after = _make(board, move, state.player)
    # A right is lost once anything moves from or to its king or rook square.
    castling = tuple(
        right and not {start, end} & {king_from, rook_from}
        for right, (king_from, _, rook_from, _, _, _) in zip(state.castling, _CASTLES, strict=True)
    )
    return ChessState(
        board=np.array(after, dtype=np.int8).reshape(8, 8),
        player=-state.player,
        castling=castling,
        en_passant=(start + end) // 2 if piece == PAWN and abs(end - start) == 16 else None,
        halfmove_clock=0 if irreversible else state.halfmove_clock + 1,
        fullmove=state.fullmove + (state.player == BLACK),
        # Positions before a capture or pawn move cannot occur again.
        history=() if irreversible else state.history + (_position_key(state),),
    )


def _position_key(state: ChessState) -> tuple:
    """Positions repeat when the same player is to move with the same pieces, castling
    rights and en passant captures. An en passant square counts only if a capture onto
    it is legal."""
    en_passant = None
    if state.en_passant is not None:
        board = state.board.ravel().tolist()
        rank, file = divmod(state.en_passant, 8)
        for side in (file - 1, file + 1):
            start = (rank - state.player) * 8 + side
            if 0 <= side < 8 and board[start] == state.player * PAWN:
                after = _make(board, (start, state.en_passant, 0), state.player)
                if not _attacked(after, after.index(state.player * KING), -state.player):
                    en_passant = state.en_passant
    return (state.board.tobytes(), state.player, state.castling, en_passant)


def _drawn_by_rule(state: ChessState) -> bool:
    """Draw by the 50-move rule, threefold repetition or insufficient material. The
    caller checks for checkmate and stalemate first."""
    if state.halfmove_clock >= 100:
        return True
    if state.history.count(_position_key(state)) >= 2:
        return True
    return _insufficient_material(state.board)


def _insufficient_material(board: np.ndarray) -> bool:
    """Kings alone, a king and one minor piece against a king, or kings and bishops
    that all stand on squares of one colour. No sequence of moves can then mate."""
    ranks, files = np.nonzero((board != 0) & (np.abs(board) != KING))
    pieces = np.abs(board[ranks, files])
    if len(pieces) == 0:
        return True
    if len(pieces) == 1 and pieces[0] in (KNIGHT, BISHOP):
        return True
    return bool(np.all(pieces == BISHOP) and len(set(((ranks + files) % 2).tolist())) == 1)


def _attacked(board: list[int], square: int, by: int) -> bool:
    """Whether any piece of player `by` attacks `square`."""
    rank, file = divmod(square, 8)
    for side in (file - 1, file + 1):
        r = rank - by  # Pawns attack diagonally forward from one rank behind.
        if 0 <= r < 8 and 0 <= side < 8 and board[r * 8 + side] == by * PAWN:
            return True
    for steps, piece in ((_KNIGHT_STEPS, KNIGHT), (_QUEEN_DIRECTIONS, KING)):
        for rank_step, file_step in steps:
            r, f = rank + rank_step, file + file_step
            if 0 <= r < 8 and 0 <= f < 8 and board[r * 8 + f] == by * piece:
                return True
    for directions, sliders in ((_ROOK_DIRECTIONS, ROOK), (_BISHOP_DIRECTIONS, BISHOP)):
        for rank_step, file_step in directions:
            r, f = rank + rank_step, file + file_step
            while 0 <= r < 8 and 0 <= f < 8:
                piece = board[r * 8 + f]
                if piece:
                    if piece in (by * sliders, by * QUEEN):
                        return True
                    break
                r, f = r + rank_step, f + file_step
    return False


def _relative(square: int, player: int) -> int:
    """The square as seen by `player`: Black's ranks are mirrored."""
    return square if player == WHITE else square ^ 56


def _square_name(square: int) -> str:
    return "abcdefgh"[square % 8] + str(square // 8 + 1)


def _square_index(name: str) -> int:
    return (int(name[1]) - 1) * 8 + "abcdefgh".index(name[0])
