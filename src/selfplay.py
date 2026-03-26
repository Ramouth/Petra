"""
selfplay.py — Self-play game generation for the zigzag training loop.

Each game uses MCTS to generate positions. For each sampled position:
  - Board tensor  (14, 8, 8) uint8
  - Visit distribution  (4096-dim sparse float32) — dense policy target
  - Value  (game outcome from side-to-move perspective) — replaced by SF later
  - FEN  (for Stockfish re-labeling)

Output format is a superset of the dataset.pt format from data.py.
reeval_stockfish.py can re-label the 'values' field without changes.
train.py picks up 'visit_dists' for the dense policy loss.

Usage
-----
    # single-process prototype (50 games)
    python3 selfplay.py --model models/sf/best.pt --games 50 --n-sim 40 \\
                        --out data/selfplay_r1.pt

    # parallelised (HPC: 32 workers)
    python3 selfplay.py --model models/sf/best.pt --games 500 --n-sim 40 \\
                        --out data/selfplay_r1.pt --workers 32
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from board import board_to_tensor, move_to_index
from model import PetraNet
from mcts import MCTS, DRAW_VALUE

# ---------------------------------------------------------------------------
# Constants  (match ZIGZAG.md spec)
# ---------------------------------------------------------------------------

SKIP_OPENING_MOVES   = 8    # skip first N half-moves (opening theory)
MAX_POSITIONS_PER_GAME = 12 # sample up to this many positions per game
MAX_HALF_MOVES       = 200  # draw if game reaches this length
RESIGN_THRESHOLD     = 0.95 # |value| must exceed this to increment counter
RESIGN_CONSECUTIVE   = 5    # resign after this many consecutive over-threshold evals
TEMP_SWITCH_MOVE     = 20   # half-moves before switching temperature 1→0


# ---------------------------------------------------------------------------
# Single-game worker (module-level for multiprocessing picklability)
# ---------------------------------------------------------------------------

def _play_game(model_path: str, n_sim: int, game_idx: int) -> dict:
    """
    Play one complete self-play game.

    Each worker process calls this function independently with its own
    loaded copy of the model. No shared state.

    Returns
    -------
    dict with keys:
        positions : list of (fen, tensor, visit_dist, half_move_number)
        outcome   : float — +1=white wins, -1=black wins, DRAW_VALUE=draw
        game_idx  : int
        n_moves   : int — total half-moves played
    """
    device = torch.device("cpu")
    model = PetraNet()
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    mcts = MCTS(model, device, dir_alpha=0.3, dir_frac=0.25)
    board = chess.Board()

    positions = []   # (fen, tensor, visit_dist, half_move)
    resign_counter = 0
    outcome = None

    while True:
        half_move = len(board.move_stack)

        # --- Natural game termination ---
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                outcome = 1.0
            elif result == "0-1":
                outcome = -1.0
            else:
                outcome = DRAW_VALUE
            break

        # --- Move-limit draw ---
        if half_move >= MAX_HALF_MOVES:
            outcome = DRAW_VALUE
            break

        # --- Temperature schedule ---
        temperature = 1.0 if half_move < TEMP_SWITCH_MOVE else 0.0

        # --- Search ---
        move, visit_dist = mcts.search(
            board, n_simulations=n_sim,
            temperature=temperature, add_noise=True,
        )

        # --- Resign check (use raw model value, not MCTS Q) ---
        val = model.value(board, device)
        if abs(val) > RESIGN_THRESHOLD:
            resign_counter += 1
            if resign_counter >= RESIGN_CONSECUTIVE:
                # val > 0 → side to move wins; determine white/black winner
                side_wins = board.turn if val > 0 else (not board.turn)
                outcome = 1.0 if side_wins == chess.WHITE else -1.0
                break
        else:
            resign_counter = 0

        # --- Record position (before pushing move) ---
        positions.append((board.fen(), board_to_tensor(board), visit_dist, half_move))

        board.push(move)

    if outcome is None:
        outcome = DRAW_VALUE

    # --- Sample positions: skip opening, cap per game ---
    eligible = [p for p in positions if p[3] >= SKIP_OPENING_MOVES]
    if len(eligible) > MAX_POSITIONS_PER_GAME:
        idxs = sorted(
            np.random.choice(len(eligible), MAX_POSITIONS_PER_GAME, replace=False)
        )
        eligible = [eligible[i] for i in idxs]

    return {
        "positions": eligible,
        "outcome":   outcome,
        "game_idx":  game_idx,
        "n_moves":   len(board.move_stack),
    }


def _worker_fn(args):
    """Pool-compatible wrapper around _play_game."""
    return _play_game(*args)


# ---------------------------------------------------------------------------
# Outcome → per-position value
# ---------------------------------------------------------------------------

def _outcome_to_value(outcome: float, fen: str) -> float:
    """
    Convert game outcome (white's perspective) to side-to-move value.

    outcome : +1.0 = white wins, -1.0 = black wins, DRAW_VALUE = draw
    fen     : position whose turn determines the sign flip
    """
    if outcome == DRAW_VALUE:
        return DRAW_VALUE
    turn = chess.Board(fen).turn
    return outcome if turn == chess.WHITE else -outcome


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def play_games(model_path: str, n_games: int, n_sim: int, workers: int) -> dict:
    """
    Play n_games self-play games and return a dataset dict.

    Format
    ------
    {
      "train": { tensors, values, move_idxs, visit_dists, fens },
      "val":   { tensors, values, move_idxs, visit_dists, fens },
      "meta":  { ... },
    }

    'values' holds game-outcome values (side-to-move perspective).
    reeval_stockfish.py will overwrite these with SF evals.
    'visit_dists' is (N, 4096) float32 — the dense policy training target.
    """
    t0 = time.time()

    all_fens        = []
    all_tensors     = []
    all_visit_dists = []
    all_values      = []
    all_move_idxs   = []

    args_list = [(model_path, n_sim, i) for i in range(n_games)]

    def _collect(result):
        outcome = result["outcome"]
        for fen, tensor, visit_dist, _ in result["positions"]:
            all_fens.append(fen)
            all_tensors.append(tensor)
            all_visit_dists.append(visit_dist)
            all_values.append(_outcome_to_value(outcome, fen))
            best_move = max(visit_dist, key=visit_dist.get) if visit_dist else None
            all_move_idxs.append(move_to_index(best_move) if best_move else 0)

    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker_fn, args_list)):
                _collect(result)
                elapsed = time.time() - t0
                print(f"  [{i+1:4d}/{n_games}]  positions: {len(all_fens):,}  "
                      f"({elapsed:.0f}s)", flush=True)
    else:
        for i, args in enumerate(args_list):
            result = _play_game(*args)
            _collect(result)
            elapsed = time.time() - t0
            print(f"  [{i+1:4d}/{n_games}]  positions: {len(all_fens):,}  "
                  f"outcome={result['outcome']:+.1f}  moves={result['n_moves']}  "
                  f"({elapsed:.0f}s)", flush=True)

    n = len(all_fens)
    print(f"\nTotal positions: {n:,}  ({time.time()-t0:.0f}s)")

    # --- Build tensors ---
    tensors_uint8 = torch.stack(all_tensors).to(torch.uint8)  # (N,14,8,8)
    values        = torch.tensor(all_values, dtype=torch.float32)
    move_idxs     = torch.tensor(all_move_idxs, dtype=torch.int64)

    visit_dist_t  = torch.zeros(n, 4096, dtype=torch.float32)
    for i, vd in enumerate(all_visit_dists):
        for move, prob in vd.items():
            visit_dist_t[i, move_to_index(move)] = prob

    # --- Train / val split (position level, shuffle) ---
    n_val  = max(1, n // 10)
    perm   = torch.randperm(n)
    v_idx  = perm[:n_val]
    t_idx  = perm[n_val:]

    def _split(tensor, idx):
        return tensor[idx]

    def _split_fens(fens, idx):
        return [fens[i] for i in idx.tolist()]

    dataset = {
        "train": {
            "tensors":     _split(tensors_uint8, t_idx),
            "values":      _split(values,        t_idx),
            "move_idxs":   _split(move_idxs,     t_idx),
            "visit_dists": _split(visit_dist_t,  t_idx),
            "fens":        _split_fens(all_fens, t_idx),
        },
        "val": {
            "tensors":     _split(tensors_uint8, v_idx),
            "values":      _split(values,        v_idx),
            "move_idxs":   _split(move_idxs,     v_idx),
            "visit_dists": _split(visit_dist_t,  v_idx),
            "fens":        _split_fens(all_fens, v_idx),
        },
        "meta": {
            "source":     "selfplay",
            "n_games":    n_games,
            "n_train":    len(t_idx),
            "n_val":      len(v_idx),
            "n_positions": n,
            "n_sim":      n_sim,
            "model_path": model_path,
        },
    }
    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate self-play games for zigzag training")
    ap.add_argument("--model",   required=True,       help="Path to PetraNet .pt weights")
    ap.add_argument("--games",   type=int, default=50, help="Number of games to play")
    ap.add_argument("--n-sim",   type=int, default=40, help="MCTS simulations per move")
    ap.add_argument("--out",     required=True,        help="Output .pt path")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel workers (each loads its own model copy)")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"selfplay.py")
    print(f"  model  : {args.model}")
    print(f"  games  : {args.games}")
    print(f"  n_sim  : {args.n_sim}")
    print(f"  workers: {args.workers}")
    print(f"  out    : {args.out}")
    print()

    dataset = play_games(
        model_path=args.model,
        n_games=args.games,
        n_sim=args.n_sim,
        workers=args.workers,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(dataset, args.out)

    meta = dataset["meta"]
    print(f"\nSaved → {args.out}")
    print(f"  train: {meta['n_train']:,}  val: {meta['n_val']:,}  "
          f"total: {meta['n_positions']:,} positions")


if __name__ == "__main__":
    main()
