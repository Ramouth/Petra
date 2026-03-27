"""
gen_openings.py — Generate a diverse set of opening FENs for selfplay seeding.

Strategy: random walks from the starting position for 10–16 half-moves,
with light filtering to reject trivially lopsided or terminal positions.
No external dependencies beyond python-chess.

The resulting FEN file is passed to selfplay.py via --opening-book so that
each self-play game starts from a different contested middlegame position,
reducing the 79%+ decisive-game rate caused by all games following the same
opening lines from the initial position.

Usage
-----
    python3 gen_openings.py --n 500 --out data/openings.txt
    python3 gen_openings.py --n 500 --min-moves 10 --max-moves 16 --out data/openings.txt
"""

import argparse
import random
import sys
import os

import chess


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_N         = 500
DEFAULT_MIN_MOVES = 10   # half-moves from start (inclusive)
DEFAULT_MAX_MOVES = 16   # half-moves from start (inclusive)

# Reject positions where one side is up by more than this many pawns worth
# (very rough: queen=9, rook=5, bishop/knight=3, pawn=1)
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}
MAX_MATERIAL_IMBALANCE = 3   # reject positions where one side is up more than a minor piece


def _material_imbalance(board: chess.Board) -> int:
    """Return |white_material - black_material| in pawn units."""
    white = sum(
        PIECE_VALUES[pt] * len(board.pieces(pt, chess.WHITE))
        for pt in PIECE_VALUES
    )
    black = sum(
        PIECE_VALUES[pt] * len(board.pieces(pt, chess.BLACK))
        for pt in PIECE_VALUES
    )
    return abs(white - black)


def _is_acceptable(board: chess.Board) -> bool:
    """Return True if this position is a suitable self-play starting point."""
    if board.is_game_over():
        return False
    if _material_imbalance(board) > MAX_MATERIAL_IMBALANCE:
        return False
    # Require at least 3 legal moves (not a dead end)
    if sum(1 for _ in board.legal_moves) < 3:
        return False
    return True


def generate_opening_fens(n: int, min_moves: int, max_moves: int,
                           seed: int = 42) -> list:
    """
    Generate n unique FENs by playing random legal moves for a random
    depth in [min_moves, max_moves] from the start position.
    """
    rng = random.Random(seed)
    seen = set()
    fens = []

    attempts = 0
    max_attempts = n * 50   # bail out if we can't find enough valid positions

    while len(fens) < n and attempts < max_attempts:
        attempts += 1
        board = chess.Board()
        depth = rng.randint(min_moves, max_moves)

        valid = True
        for _ in range(depth):
            if board.is_game_over():
                valid = False
                break
            moves = list(board.legal_moves)
            board.push(rng.choice(moves))

        if not valid:
            continue
        if not _is_acceptable(board):
            continue

        # Use FEN without move counters for deduplication
        fen = board.fen()
        key = " ".join(fen.split()[:4])   # piece placement + turn + castling + ep
        if key in seen:
            continue

        seen.add(key)
        fens.append(fen)

    if len(fens) < n:
        print(f"Warning: only generated {len(fens)}/{n} positions after "
              f"{attempts} attempts. Try relaxing --min-moves/--max-moves.",
              file=sys.stderr)
    return fens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate diverse opening FENs for selfplay.py --opening-book"
    )
    ap.add_argument("--n",          type=int, default=DEFAULT_N,
                    help=f"Number of FENs to generate (default: {DEFAULT_N})")
    ap.add_argument("--min-moves",  type=int, default=DEFAULT_MIN_MOVES,
                    help=f"Min half-moves from start (default: {DEFAULT_MIN_MOVES})")
    ap.add_argument("--max-moves",  type=int, default=DEFAULT_MAX_MOVES,
                    help=f"Max half-moves from start (default: {DEFAULT_MAX_MOVES})")
    ap.add_argument("--out",        required=True,
                    help="Output text file — one FEN per line")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    print(f"gen_openings.py")
    print(f"  n         : {args.n}")
    print(f"  moves     : {args.min_moves}–{args.max_moves} half-moves")
    print(f"  out       : {args.out}")
    print()

    fens = generate_opening_fens(
        n=args.n,
        min_moves=args.min_moves,
        max_moves=args.max_moves,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for fen in fens:
            f.write(fen + "\n")

    print(f"Generated {len(fens)} unique opening positions → {args.out}")


if __name__ == "__main__":
    main()
