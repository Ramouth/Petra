"""
Stage 1 curriculum: KQ vs K endgame position generator.

Generates random legal KQ vs K positions and their color-flipped mirrors,
labels them (rule-based or Stockfish), and saves a dataset compatible
with train.py.

Label convention (side-to-move relative, as in the rest of the pipeline):
  White has queen, white to move → +1.0
  Black has queen, white to move → -1.0  (color-flipped positions)

No Stockfish needed for KQ vs K — the label is unambiguous from the rules.
SF depth is available as an option for verification.

Usage
-----
    python3 generate_endgame.py --positions 10000 --out data/kqk_stage1.pt
    python3 generate_endgame.py --positions 10000 --out data/kqk_stage1.pt --stockfish /path/to/sf --depth 5
"""

import argparse
import os
import random
import sys

import chess
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board import board_to_tensor, move_to_index


# ---------------------------------------------------------------------------
# Position generation
# ---------------------------------------------------------------------------

def random_kqk_position(white_has_queen: bool = True) -> chess.Board:
    """
    Generate one random legal KQ vs K position.

    white_has_queen=True  → white king + queen, black bare king, white to move
    white_has_queen=False → black king + queen, white bare king, white to move
                            (the color-flipped antipodal partner)
    """
    while True:
        squares = random.sample(range(64), 3)
        wk_sq, piece_sq, bk_sq = squares

        board = chess.Board(fen=None)
        board.clear()

        if white_has_queen:
            board.set_piece_at(wk_sq, chess.Piece(chess.KING,  chess.WHITE))
            board.set_piece_at(piece_sq, chess.Piece(chess.QUEEN, chess.WHITE))
            board.set_piece_at(bk_sq, chess.Piece(chess.KING,  chess.BLACK))
        else:
            board.set_piece_at(wk_sq, chess.Piece(chess.KING,  chess.WHITE))
            board.set_piece_at(piece_sq, chess.Piece(chess.QUEEN, chess.BLACK))
            board.set_piece_at(bk_sq, chess.Piece(chess.KING,  chess.BLACK))

        board.turn = chess.WHITE

        # Validity: not illegal (kings adjacent via queen occupancy, etc.)
        if not board.is_valid():
            continue
        # Skip terminal positions — they are not useful training positions
        if board.is_game_over():
            continue

        return board


def random_krk_position(white_has_rook: bool = True) -> chess.Board:
    """
    Generate one random legal KR vs K position.

    white_has_rook=True  → white king + rook, black bare king, white to move
    white_has_rook=False → black king + rook, white bare king, white to move
                           (the color-flipped antipodal partner)
    """
    while True:
        squares = random.sample(range(64), 3)
        wk_sq, piece_sq, bk_sq = squares

        board = chess.Board(fen=None)
        board.clear()

        if white_has_rook:
            board.set_piece_at(wk_sq,   chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(piece_sq, chess.Piece(chess.ROOK, chess.WHITE))
            board.set_piece_at(bk_sq,   chess.Piece(chess.KING, chess.BLACK))
        else:
            board.set_piece_at(wk_sq,   chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(piece_sq, chess.Piece(chess.ROOK, chess.BLACK))
            board.set_piece_at(bk_sq,   chess.Piece(chess.KING, chess.BLACK))

        board.turn = chess.WHITE

        if not board.is_valid():
            continue
        if board.is_game_over():
            continue

        return board


def random_kpk_position(white_has_pawn: bool = True) -> chess.Board:
    """
    Generate one random legal K+P vs K position.

    white_has_pawn=True  → white king + pawn, black bare king, white to move
    white_has_pawn=False → black king + pawn, white bare king, white to move

    Pawn is placed on ranks 2–7 (never rank 1 or 8).
    Labels are noisy — some K+P vs K positions are theoretical draws —
    but the pawn's geometry will cluster distinctly from major pieces.
    """
    pawn_squares = [sq for sq in range(64) if 1 <= chess.square_rank(sq) <= 6]

    while True:
        pawn_sq = random.choice(pawn_squares)
        remaining = [sq for sq in range(64) if sq != pawn_sq]
        wk_sq, bk_sq = random.sample(remaining, 2)

        board = chess.Board(fen=None)
        board.clear()

        if white_has_pawn:
            board.set_piece_at(wk_sq,   chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.WHITE))
            board.set_piece_at(bk_sq,   chess.Piece(chess.KING, chess.BLACK))
        else:
            board.set_piece_at(wk_sq,   chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.BLACK))
            board.set_piece_at(bk_sq,   chess.Piece(chess.KING, chess.BLACK))

        board.turn = chess.WHITE

        if not board.is_valid():
            continue
        if board.is_game_over():
            continue

        return board


def random_4piece_position(white_piece: int, black_piece: int,
                           white_piece_on_ranks=None, black_piece_on_ranks=None) -> chess.Board:
    """
    Generate one random legal 4-piece position: white king + piece vs black king + piece.
    Both sides have material — forces the model to learn relative piece strength.

    white_piece / black_piece: chess.QUEEN, chess.ROOK, chess.PAWN, etc.
    white_piece_on_ranks / black_piece_on_ranks: restrict piece to these ranks (for pawns).
    Label convention (caller's responsibility): stronger side = +1, weaker side = -1.
    """
    def valid_squares(piece_type, ranks):
        if ranks is not None:
            return [sq for sq in range(64) if chess.square_rank(sq) in ranks]
        return list(range(64))

    wp_squares = valid_squares(white_piece, white_piece_on_ranks)
    bp_squares = valid_squares(black_piece, black_piece_on_ranks)

    while True:
        wp_sq = random.choice(wp_squares)
        bp_sq = random.choice(bp_squares)
        if wp_sq == bp_sq:
            continue
        remaining = [sq for sq in range(64) if sq not in (wp_sq, bp_sq)]
        wk_sq, bk_sq = random.sample(remaining, 2)

        board = chess.Board(fen=None)
        board.clear()
        board.set_piece_at(wk_sq, chess.Piece(chess.KING,  chess.WHITE))
        board.set_piece_at(wp_sq, chess.Piece(white_piece, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING,  chess.BLACK))
        board.set_piece_at(bp_sq, chess.Piece(black_piece, chess.BLACK))
        board.turn = chess.WHITE

        if not board.is_valid():
            continue
        if board.is_game_over():
            continue

        return board


def random_kp_kp_position(white_more_advanced: bool = True):
    """
    KP vs KP: balanced pawns, both sides have a pawn.
    Label by pawn advancement — whichever pawn is closer to promotion wins.
    Skips positions where both pawns are equally advanced.

    white_more_advanced=True  → white pawn further along → label +1
    white_more_advanced=False → black pawn further along → label -1
    """
    pawn_ranks = list(range(1, 7))

    while True:
        wp_sq = random.choice([sq for sq in range(64) if chess.square_rank(sq) in pawn_ranks])
        bp_sq = random.choice([sq for sq in range(64) if chess.square_rank(sq) in pawn_ranks])
        if wp_sq == bp_sq:
            continue

        # white advancement: higher rank = closer to promotion (rank 6 = 7th rank)
        # black advancement: lower rank = closer to promotion (rank 1 = 2nd rank from black's view)
        white_adv = chess.square_rank(wp_sq)
        black_adv = 7 - chess.square_rank(bp_sq)

        if white_adv == black_adv:
            continue  # skip equal — label would be ambiguous

        if white_more_advanced and white_adv <= black_adv:
            continue
        if not white_more_advanced and black_adv <= white_adv:
            continue

        remaining = [sq for sq in range(64) if sq not in (wp_sq, bp_sq)]
        wk_sq, bk_sq = random.sample(remaining, 2)

        board = chess.Board(fen=None)
        board.clear()
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(wp_sq, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(bp_sq, chess.Piece(chess.PAWN, chess.BLACK))
        board.turn = chess.WHITE

        if not board.is_valid():
            continue
        if board.is_game_over():
            continue

        return board


_PAWN_RANKS = list(range(1, 7))  # ranks 2-7 (0-indexed 1-6)

_STAGE_GENERATORS = {
    # Single-piece endgames — who has the piece wins
    1: (lambda: random_kqk_position(white_has_queen=True),
        lambda: random_kqk_position(white_has_queen=False)),
    2: (lambda: random_krk_position(white_has_rook=True),
        lambda: random_krk_position(white_has_rook=False)),
    3: (lambda: random_kpk_position(white_has_pawn=True),
        lambda: random_kpk_position(white_has_pawn=False)),
    # Mixed-material — stronger piece wins, forces piece value ordering
    4: (lambda: random_4piece_position(chess.QUEEN, chess.ROOK),
        lambda: random_4piece_position(chess.ROOK,  chess.QUEEN)),
    5: (lambda: random_4piece_position(chess.ROOK, chess.PAWN,
                                       black_piece_on_ranks=_PAWN_RANKS),
        lambda: random_4piece_position(chess.PAWN, chess.ROOK,
                                       white_piece_on_ranks=_PAWN_RANKS)),
    # Minor pieces vs pawn — teaches bishop/knight > pawn
    6: (lambda: random_4piece_position(chess.BISHOP, chess.PAWN,
                                       black_piece_on_ranks=_PAWN_RANKS),
        lambda: random_4piece_position(chess.PAWN, chess.BISHOP,
                                       white_piece_on_ranks=_PAWN_RANKS)),
    7: (lambda: random_4piece_position(chess.KNIGHT, chess.PAWN,
                                       black_piece_on_ranks=_PAWN_RANKS),
        lambda: random_4piece_position(chess.PAWN, chess.KNIGHT,
                                       white_piece_on_ranks=_PAWN_RANKS)),
    # Balanced pawns — advancement heuristic label, forces pawn structure geometry
    8: (lambda: random_kp_kp_position(white_more_advanced=True),
        lambda: random_kp_kp_position(white_more_advanced=False)),
}


def generate_positions(n: int, include_mirrors: bool = True, stages=None):
    """
    Generate n endgame positions, mixed across one or more curriculum stages.

    stages: int or list of ints — 1=KQK, 2=KRK. Default [1].
    Positions are split evenly across stages so the model sees all piece
    types simultaneously and learns the abstraction "stronger piece = win"
    rather than memorising a single piece type.

    If include_mirrors=True, each position is paired with its color-flipped
    mirror (2n total). Mirrors are the antipodal partners for the antipodal loss.

    Returns list of (board, value) tuples, shuffled.
    """
    if stages is None:
        stages = [1]
    if isinstance(stages, int):
        stages = [stages]

    for s in stages:
        if s not in _STAGE_GENERATORS:
            raise ValueError(f"Unknown endgame stage: {s}. Supported: {list(_STAGE_GENERATORS)}")

    # Split n evenly; last stage gets any remainder
    n_per_stage = [n // len(stages)] * len(stages)
    n_per_stage[-1] += n - sum(n_per_stage)

    all_positions = []
    for stage, n_stage in zip(stages, n_per_stage):
        pos_fn, mirror_fn = _STAGE_GENERATORS[stage]
        generated = 0
        seen_fens = set()

        while generated < n_stage:
            board = pos_fn()
            fen = board.board_fen()
            if fen in seen_fens:
                continue
            seen_fens.add(fen)

            all_positions.append((board, +1.0))
            generated += 1

            if include_mirrors:
                all_positions.append((mirror_fn(), -1.0))

    random.shuffle(all_positions)
    return all_positions


# ---------------------------------------------------------------------------
# Optional Stockfish verification
# ---------------------------------------------------------------------------

def label_with_stockfish(positions, stockfish_path: str, depth: int = 5):
    """
    Replace rule-based labels with Stockfish evaluations.
    Used for verification — KQ vs K labels are theoretically certain without SF.
    Returns list of (board, value) with SF-derived values.
    """
    import chess.engine
    print(f"  Verifying labels with Stockfish depth {depth} ...")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    result = []
    for i, (board, _) in enumerate(positions):
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white()
        if score.is_mate():
            m = score.mate()
            # Positive mate = white wins, negative = black wins
            val = 1.0 if m > 0 else -1.0
        else:
            cp = score.score(mate_score=10000)
            import math
            val = math.tanh(cp / 400.0)
        # Convert to side-to-move relative
        if board.turn == chess.BLACK:
            val = -val
        result.append((board, val))
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(positions)}")
    engine.quit()
    return result


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(positions, val_frac: float = 0.1):
    """
    Convert list of (board, value) to train.py-compatible dataset dict.

    Policy: uniform over legal moves (policy loss not meaningful for endgame
    positions without MCTS visit distributions — use --policy-weight 0 in train.py).
    """
    random.shuffle(positions)
    n_val = max(1, int(len(positions) * val_frac))
    splits = {
        "train": positions[n_val:],
        "val":   positions[:n_val],
    }

    data = {}
    for split_name, split in splits.items():
        tensors    = []
        values     = []
        move_idxs  = []
        visit_dists = []

        for board, value in split:
            tensors.append(board_to_tensor(board))
            values.append(value)

            # Uniform policy over legal moves
            legal = list(board.legal_moves)
            vd = torch.zeros(4096, dtype=torch.float32)
            if legal:
                w = 1.0 / len(legal)
                for m in legal:
                    vd[move_to_index(m)] = w
                move_idxs.append(move_to_index(legal[0]))
            else:
                move_idxs.append(0)

            visit_dists.append(vd)

        data[split_name] = {
            "tensors":     torch.stack(tensors).to(torch.uint8),
            "values":      torch.tensor(values, dtype=torch.float32),
            "move_idxs":   torch.tensor(move_idxs, dtype=torch.int64),
            "visit_dists": torch.stack(visit_dists),
        }

    n_train = len(data["train"]["tensors"])
    n_val   = len(data["val"]["tensors"])
    pos_values = data["train"]["values"]
    wins  = (pos_values > 0.5).sum().item()
    losses = (pos_values < -0.5).sum().item()

    data["meta"] = {
        "source":       "endgame",
        "n_train":      n_train,
        "n_val":        n_val,
        "stage":        1,
        "label_mean":   float(pos_values.mean()),
        "label_std":    float(pos_values.std()),
        "pct_win":      wins / n_train,
        "pct_loss":     losses / n_train,
    }

    print(f"  train: {n_train:,}  val: {n_val:,}")
    print(f"  labels — win: {wins/n_train*100:.1f}%  loss: {losses/n_train*100:.1f}%")
    print(f"  mean: {float(pos_values.mean()):+.3f}  std: {float(pos_values.std()):.3f}")

    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions",  type=int, default=10000,
                    help="Number of stronger-side positions to generate (mirrors included automatically)")
    ap.add_argument("--stages",     type=int, nargs="+", default=[1],
                    help="Endgame stages to mix, space-separated: 1=KQK, 2=KRK (default: 1)")
    ap.add_argument("--out",        required=True,
                    help="Output .pt file path")
    ap.add_argument("--no-mirrors", action="store_true",
                    help="Do not include color-flipped antipodal partners")
    ap.add_argument("--stockfish",  default=None,
                    help="Path to Stockfish binary for label verification (optional)")
    ap.add_argument("--depth",      type=int, default=5,
                    help="Stockfish depth for verification (default: 5)")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    include_mirrors = not args.no_mirrors
    stage_names = {1: "KQ vs K", 2: "KR vs K"}
    label = "+".join(stage_names.get(s, f"stage{s}") for s in args.stages)

    print(f"Generating {args.positions:,} {label} positions"
          + (" + mirrors" if include_mirrors else "") + " ...")

    positions = generate_positions(args.positions, include_mirrors=include_mirrors,
                                   stages=args.stages)
    print(f"  Generated {len(positions):,} positions total")

    if args.stockfish:
        positions = label_with_stockfish(positions, args.stockfish, args.depth)
        print(f"  SF labels applied (depth {args.depth})")
    else:
        print("  Using rule-based labels (KQ vs K = white wins / black wins by construction)")

    print("\nBuilding dataset ...")
    data = build_dataset(positions)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(data, args.out)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
