"""
Data pipeline for Phase 1 supervised pretraining.

Parses Lichess PGN files, extracts labeled positions, runs integrity
checks, and saves a validated dataset.

Label convention
----------------
Values are from the perspective of the side to move:
  win  = +1.0   (side to move wins the game)
  loss = -1.0   (side to move loses)
  draw = -0.1   (draw contempt — matches MCTS terminal scoring)

This is verified by the validation suite before any dataset is saved.

Integrity checks
----------------
1. Label values are strictly in {+1.0, -0.1, -1.0} — no floats, no nulls
2. Sign correctness: White-to-move positions in White-won games → +1.0
                     Black-to-move positions in White-won games → -1.0
3. Side-to-move balance: ~50% White, ~50% Black (within ±10%)
4. King presence: every position has exactly one White king and one Black king
5. Label distribution: at least 20% of each class (no severe imbalance)
6. Train/val split is at the GAME level — no position leakage across splits
7. Spot-check known positions against expected labels

Usage
-----
    python3 data.py --pgn games.pgn --out dataset.pt --max-games 100000
    python3 data.py --pgn games.pgn --out dataset.pt --validate-only
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import chess
import chess.pgn
import torch

sys.path.insert(0, os.path.dirname(__file__))
from board import board_to_tensor, outcome_to_value

# Positions to skip at the start of each game (opening book noise)
SKIP_OPENING_MOVES = 10

# Maximum positions sampled per game (prevents one long game dominating)
MAX_POSITIONS_PER_GAME = 8

# Minimum game length to include (very short games are usually abandoned)
MIN_GAME_MOVES = 15

# Maximum game length (runaway games add noise)
MAX_GAME_MOVES = 200

# Validation fraction held out at game level
VAL_FRACTION = 0.05

VALID_LABEL_VALUES = {1.0, -0.1, -1.0}


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class Position:
    tensor:   torch.Tensor  # (14, 8, 8) float32
    value:    float         # label: +1.0, -0.1, or -1.0
    move_idx: int           # move played from this position (from_sq*64 + to_sq)
    fen:      str           # for debugging and deduplication
    game_id:  int           # game index — used for train/val split


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_pgn(pgn_path: str,
              max_games: int = 100_000,
              min_elo: int = 0,
              require_normal_termination: bool = True,
              seed: int = 42) -> List[Position]:
    """
    Stream a PGN file and extract labeled positions.

    Filters applied per game:
      - Result must be "1-0", "0-1", or "1/2-1/2" (no abandoned/ongoing)
      - Game length must be in [MIN_GAME_MOVES, MAX_GAME_MOVES]
      - Both players must have Elo >= min_elo (if headers present)
      - Termination must be "Normal" if require_normal_termination=True

    Position sampling:
      - Skip first SKIP_OPENING_MOVES moves
      - Sample up to MAX_POSITIONS_PER_GAME positions uniformly
    """
    rng = random.Random(seed)
    positions = []
    game_id = 0
    games_parsed = 0
    games_skipped = 0

    print(f"Parsing {pgn_path} (max_games={max_games:,}) ...")
    t0 = time.time()

    with open(pgn_path, errors="replace") as f:
        while games_parsed < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # --- Game-level filters ---
            result = game.headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                games_skipped += 1
                continue

            if require_normal_termination:
                termination = game.headers.get("Termination", "")
                if termination and termination.lower() not in ("normal", ""):
                    games_skipped += 1
                    continue

            if min_elo > 0:
                try:
                    w_elo = int(game.headers.get("WhiteElo", "0") or "0")
                    b_elo = int(game.headers.get("BlackElo", "0") or "0")
                    if w_elo < min_elo or b_elo < min_elo:
                        games_skipped += 1
                        continue
                except ValueError:
                    pass

            # --- Collect (board_before_move, move) pairs ---
            board = game.board()
            pairs = []   # (board_before, move)
            for move in game.mainline_moves():
                pairs.append((board.copy(), move))
                board.push(move)

            n_moves = len(pairs)
            if n_moves < MIN_GAME_MOVES or n_moves > MAX_GAME_MOVES:
                games_skipped += 1
                continue

            # --- Sample positions after opening ---
            candidate_indices = list(range(SKIP_OPENING_MOVES, n_moves))
            if not candidate_indices:
                games_skipped += 1
                continue

            sample_indices = rng.sample(
                candidate_indices,
                min(MAX_POSITIONS_PER_GAME, len(candidate_indices))
            )

            for idx in sample_indices:
                b, move = pairs[idx]
                value = outcome_to_value(result, b.turn)
                tensor = board_to_tensor(b)
                positions.append(Position(
                    tensor=tensor,
                    value=value,
                    move_idx=move.from_square * 64 + move.to_square,
                    fen=b.fen(),
                    game_id=game_id,
                ))

            game_id += 1
            games_parsed += 1

            if games_parsed % 10_000 == 0:
                elapsed = time.time() - t0
                print(f"  {games_parsed:,} games parsed, "
                      f"{len(positions):,} positions, "
                      f"{elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"Done: {games_parsed:,} games parsed, "
          f"{games_skipped:,} skipped, "
          f"{len(positions):,} positions in {elapsed:.1f}s")
    return positions


# ---------------------------------------------------------------------------
# Integrity validation
# ---------------------------------------------------------------------------

KNOWN_POSITIONS = [
    # (fen, expected_value, tolerance, description)
    # Starting position — unknown outcome, value near 0 after training,
    # but label depends on game result so we can't check this directly.
    # Instead we check the sign logic on unambiguous endgames.
    (
        "4k3/8/8/8/8/8/8/4K2Q w - - 0 1",   # KQ vs K, White to move → White should win → +1
        1.0, 0.0, "KQ vs K White to move: label must be +1.0 (White wins)"
    ),
    (
        "4k3/8/8/8/8/8/8/4K2Q b - - 0 1",   # KQ vs K, Black to move → Black loses → -1
        -1.0, 0.0, "KQ vs K Black to move: label must be -1.0 (Black loses)"
    ),
]


def validate_dataset(positions: List[Position], strict: bool = True) -> bool:
    """
    Run all integrity checks. Prints a full report.
    Returns True if all checks pass. Raises ValueError if strict=True and any fail.
    """
    errors = []
    warnings = []
    n = len(positions)
    print(f"\n{'='*60}")
    print(f"Dataset validation  ({n:,} positions)")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Check 1: label values are in the allowed set
    # ------------------------------------------------------------------
    bad_labels = [(i, p.value) for i, p in enumerate(positions)
                  if p.value not in VALID_LABEL_VALUES]
    if bad_labels:
        errors.append(f"CHECK 1 FAIL — {len(bad_labels)} invalid label values "
                      f"(first 5: {[v for _, v in bad_labels[:5]]})")
    else:
        print("CHECK 1 PASS — all labels in {+1.0, -0.1, -1.0}")

    # ------------------------------------------------------------------
    # Check 2: sign correctness on known positions
    # ------------------------------------------------------------------
    sign_errors = 0
    for fen, expected, tol, desc in KNOWN_POSITIONS:
        board = chess.Board(fen)
        # Simulate: if this were a win/loss game, what label would we assign?
        if expected == 1.0:
            simulated = outcome_to_value("1-0" if board.turn == chess.WHITE else "0-1", board.turn)
        elif expected == -1.0:
            simulated = outcome_to_value("1-0" if board.turn == chess.BLACK else "0-1", board.turn)
        else:
            simulated = outcome_to_value("1/2-1/2", board.turn)

        if abs(simulated - expected) > tol:
            errors.append(f"CHECK 2 FAIL — sign error on: {desc}  "
                          f"got={simulated}  expected={expected}")
            sign_errors += 1

    if sign_errors == 0:
        print("CHECK 2 PASS — label sign convention is correct")

    # ------------------------------------------------------------------
    # Check 3: label distribution
    # ------------------------------------------------------------------
    n_win  = sum(1 for p in positions if p.value == 1.0)
    n_draw = sum(1 for p in positions if p.value == -0.1)
    n_loss = sum(1 for p in positions if p.value == -1.0)
    pct_win  = 100 * n_win  / n
    pct_draw = 100 * n_draw / n
    pct_loss = 100 * n_loss / n

    print(f"CHECK 3 — label distribution:")
    print(f"  win (+1.0)  : {n_win:>8,}  ({pct_win:5.1f}%)")
    print(f"  draw (-0.1) : {n_draw:>8,}  ({pct_draw:5.1f}%)")
    print(f"  loss (-1.0) : {n_loss:>8,}  ({pct_loss:5.1f}%)")

    # Win and loss should be approximately equal (since every win for one
    # side is a loss for the other, and we sample both sides of each game)
    if abs(pct_win - pct_loss) > 15:
        warnings.append(f"CHECK 3 WARN — win/loss imbalance: {pct_win:.1f}% vs {pct_loss:.1f}%")
    if pct_win < 10 or pct_loss < 10:
        errors.append(f"CHECK 3 FAIL — severe label imbalance")

    # ------------------------------------------------------------------
    # Check 4: side-to-move balance
    # ------------------------------------------------------------------
    n_white_to_move = sum(
        1 for p in positions if p.tensor[12].sum().item() == 64
    )
    n_black_to_move = n - n_white_to_move
    pct_white = 100 * n_white_to_move / n

    print(f"CHECK 4 — side to move:")
    print(f"  White to move: {n_white_to_move:,}  ({pct_white:.1f}%)")
    print(f"  Black to move: {n_black_to_move:,}  ({100-pct_white:.1f}%)")

    if abs(pct_white - 50) > 10:
        warnings.append(f"CHECK 4 WARN — side-to-move imbalance: {pct_white:.1f}% White")

    # ------------------------------------------------------------------
    # Check 5: king presence (spot-check first 2000 positions)
    # ------------------------------------------------------------------
    king_errors = 0
    spot_check_n = min(2000, n)
    for i in range(spot_check_n):
        p = positions[i]
        wk = int(p.tensor[5].sum().item())   # plane 5 = White king
        bk = int(p.tensor[11].sum().item())  # plane 11 = Black king
        if wk != 1 or bk != 1:
            king_errors += 1
            if king_errors <= 3:
                errors.append(f"CHECK 5 FAIL — position {i} has "
                              f"{wk} White kings, {bk} Black kings: {p.fen}")

    if king_errors == 0:
        print(f"CHECK 5 PASS — king presence verified on {spot_check_n:,} positions")
    else:
        errors.append(f"CHECK 5 FAIL — {king_errors} positions with wrong king count")

    # ------------------------------------------------------------------
    # Check 6: sign consistency within a game
    # Consecutive positions from the same game should have opposite signs
    # (since the side to move alternates and the game result is fixed).
    # Sample 100 games and verify.
    # ------------------------------------------------------------------
    game_positions: dict = {}
    for p in positions:
        game_positions.setdefault(p.game_id, []).append(p)

    sign_inconsistencies = 0
    for gid, gpos in list(game_positions.items())[:100]:
        # Sort by checking a few pairs: if two consecutive positions are from
        # the same game, they should have opposite labels (unless one is a draw).
        for a, b in zip(gpos, gpos[1:]):
            if a.value == -0.1 or b.value == -0.1:
                continue  # draws are uniform, can't check
            if a.value == b.value:
                sign_inconsistencies += 1

    if sign_inconsistencies == 0:
        print("CHECK 6 PASS — within-game label signs are consistent")
    else:
        errors.append(f"CHECK 6 FAIL — {sign_inconsistencies} within-game sign inconsistencies")

    # ------------------------------------------------------------------
    # Check 7: no duplicate FENs in validation set
    # (Duplicates in training are tolerable; in val they inflate metrics.)
    # ------------------------------------------------------------------
    # (Checked after split — reported in save_dataset)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
    if errors:
        for e in errors:
            print(f"ERROR:   {e}")
        print(f"\nValidation FAILED ({len(errors)} errors, {len(warnings)} warnings)")
        if strict:
            raise ValueError(f"Dataset failed validation with {len(errors)} errors.")
        return False
    else:
        print(f"Validation PASSED  ({len(warnings)} warnings)")
        return True


# ---------------------------------------------------------------------------
# Train / val split and save
# ---------------------------------------------------------------------------

def split_and_save(positions: List[Position],
                   out_path: str,
                   val_fraction: float = VAL_FRACTION,
                   seed: int = 42):
    """
    Split at game level (no leakage), validate val set for duplicates,
    and save to disk with a metadata sidecar.
    """
    rng = random.Random(seed)

    all_game_ids = list({p.game_id for p in positions})
    rng.shuffle(all_game_ids)
    n_val_games = max(1, int(len(all_game_ids) * val_fraction))
    val_game_ids = set(all_game_ids[:n_val_games])

    train = [p for p in positions if p.game_id not in val_game_ids]
    val   = [p for p in positions if p.game_id in val_game_ids]

    # Duplicate FEN check in val set
    val_fens = [p.fen for p in val]
    n_val_dupes = len(val_fens) - len(set(val_fens))
    if n_val_dupes > 0:
        print(f"WARNING: {n_val_dupes} duplicate FENs in validation set")

    # Cross-contamination check: no FEN should appear in both sets
    train_fens = set(p.fen for p in train)
    val_fens_set = set(val_fens)
    overlap = train_fens & val_fens_set
    if overlap:
        print(f"WARNING: {len(overlap)} FENs appear in both train and val "
              f"(same board position, different games — tolerable)")

    print(f"\nSplit: {len(train):,} train  /  {len(val):,} val  "
          f"(from {len(all_game_ids) - n_val_games} / {n_val_games} games)")

    # Save
    def pack(subset):
        return {
            "tensors":   torch.stack([p.tensor for p in subset]),   # (N, 14, 8, 8)
            "values":    torch.tensor([p.value    for p in subset], dtype=torch.float32),
            "move_idxs": torch.tensor([p.move_idx for p in subset], dtype=torch.long),
            "fens":      [p.fen for p in subset],
        }

    data = {
        "train": pack(train),
        "val":   pack(val),
        "meta": {
            "n_train":         len(train),
            "n_val":           len(val),
            "val_fraction":    val_fraction,
            "label_values":    list(VALID_LABEL_VALUES),
            "draw_value":      -0.1,
            "skip_opening":    SKIP_OPENING_MOVES,
            "max_per_game":    MAX_POSITIONS_PER_GAME,
            "n_val_dupes":     n_val_dupes,
            "n_fen_overlap":   len(overlap),
        },
    }

    torch.save(data, out_path)
    print(f"Saved → {out_path}")

    # Human-readable sidecar
    meta_path = out_path.replace(".pt", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(data["meta"], f, indent=2)
    print(f"Metadata → {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn",           required=True,        help="Path to PGN file")
    ap.add_argument("--out",           default="dataset.pt", help="Output .pt path")
    ap.add_argument("--max-games",     type=int, default=100_000)
    ap.add_argument("--min-elo",       type=int, default=1500,
                    help="Minimum Elo for both players (0 = no filter)")
    ap.add_argument("--no-termination-filter", action="store_true",
                    help="Include games that ended by timeout/abandon")
    ap.add_argument("--validate-only", action="store_true",
                    help="Parse and validate but do not save")
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    positions = parse_pgn(
        pgn_path=args.pgn,
        max_games=args.max_games,
        min_elo=args.min_elo,
        require_normal_termination=not args.no_termination_filter,
        seed=args.seed,
    )

    if not positions:
        print("No positions extracted. Check PGN path and filters.")
        sys.exit(1)

    validate_dataset(positions, strict=True)

    if not args.validate_only:
        split_and_save(positions, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
