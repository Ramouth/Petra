"""
Board representation.

Converts a python-chess Board to a (14, 8, 8) float32 tensor:

  Planes 0–5:   White pieces  (P N B R Q K)
  Planes 6–11:  Black pieces  (p n b r q k)
  Plane 12:     Side to move  (all 1s = White to move, all 0s = Black)
  Plane 13:     Castling rights (1s at four corners: WK, WQ, BK, BQ)

Value convention: +1 means the side to move wins.
When constructing training targets, flip the game outcome accordingly.
"""

import chess
import torch

PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a python-chess Board to a (14, 8, 8) float32 tensor.

    Rank 0 = rank 1 (White's back rank), rank 7 = rank 8 (Black's back rank).
    File 0 = a-file, file 7 = h-file.
    """
    t = torch.zeros(14, 8, 8, dtype=torch.float32)

    for sq, piece in board.piece_map().items():
        rank = sq >> 3   # sq // 8
        file = sq & 7    # sq % 8
        t[PIECE_TO_PLANE[(piece.piece_type, piece.color)], rank, file] = 1.0

    if board.turn == chess.WHITE:
        t[12] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        t[13, 0, 7] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        t[13, 0, 0] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        t[13, 7, 7] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        t[13, 7, 0] = 1.0

    return t


def flip_board_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Color-flip a (..., 14, 8, 8) board tensor without going through chess.Board.

    Equivalent to board.mirror(): swap white/black piece planes, flip ranks,
    flip side-to-move plane, flip castling plane ranks.

    Works on single tensors (14, 8, 8) and batches (N, 14, 8, 8).
    """
    f = t.clone()
    # Swap white pieces (planes 0-5) and black pieces (planes 6-11), flip ranks
    f[..., 0:6, :, :] = t[..., 6:12, :, :].flip(-2)
    f[..., 6:12, :, :] = t[..., 0:6, :, :].flip(-2)
    # Flip side-to-move (uniform plane: 1=white, 0=black)
    f[..., 12, :, :] = 1.0 - t[..., 12, :, :]
    # Flip castling plane ranks (WK/WQ corners ↔ BK/BQ corners)
    f[..., 13, :, :] = t[..., 13, :, :].flip(-2)
    return f


def flip_turn_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Flip only the side-to-move plane of a (..., 14, 8, 8) board tensor.

    Produces the same position with the opposite side to move.
    Used for the antipodal loss: under STM-relative labels, a position and
    its turn-flip have opposite labels (+1 / -1), so their geometry vectors
    should point in opposite directions. This is consistent — unlike
    flip_board_tensor, whose color-flipped partner has the SAME STM-relative
    label (+1 both), creating a contradiction with the antipodal constraint.

    Works on single tensors (14, 8, 8) and batches (N, 14, 8, 8).
    """
    f = t.clone()
    f[..., 12, :, :] = 1.0 - t[..., 12, :, :]
    return f


def outcome_to_value(result: str, turn: chess.Color) -> float:
    """
    Convert a PGN result string to a value from the perspective of `turn`.

    result: "1-0", "0-1", "1/2-1/2"
    Returns +1 (current side wins), -1 (loses), or DRAW_VALUE (draw).
    """
    DRAW_VALUE = -0.1   # draw contempt: draws are slightly negative

    if result == "1-0":
        return 1.0 if turn == chess.WHITE else -1.0
    if result == "0-1":
        return 1.0 if turn == chess.BLACK else -1.0
    return DRAW_VALUE


def move_to_index(move: chess.Move) -> int:
    """Encode a move as from_sq * 64 + to_sq. Range [0, 4095]."""
    return move.from_square * 64 + move.to_square


def index_to_squares(idx: int):
    """Decode a move index to (from_sq, to_sq)."""
    return idx >> 6, idx & 63
