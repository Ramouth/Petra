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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board import board_to_tensor, move_to_index
from model import PetraNet
from mcts import MCTS, DRAW_VALUE

# ---------------------------------------------------------------------------
# Constants  (match ZIGZAG.md spec)
# ---------------------------------------------------------------------------

SKIP_OPENING_MOVES   = 8    # skip first N half-moves (opening theory)
MAX_POSITIONS_PER_GAME = 12 # sample up to this many positions per game
MAX_HALF_MOVES       = 200  # draw if game reaches this length
RESIGN_THRESHOLD     = 0.95 # default — overridden by --resign-threshold CLI arg
RESIGN_CONSECUTIVE   = 10   # default — overridden by --resign-consecutive CLI arg
TEMP_SWITCH_MOVE     = 20   # half-moves before switching temperature 1→0


# ---------------------------------------------------------------------------
# Single-game worker (module-level for multiprocessing picklability)
# ---------------------------------------------------------------------------

def _play_game(model_path: str, n_sim: int, game_idx: int,
               start_fen: str = None,
               resign_threshold: float = RESIGN_THRESHOLD,
               resign_consecutive: int = RESIGN_CONSECUTIVE) -> dict:
    """
    Play one complete self-play game.

    Each worker process calls this function independently with its own
    loaded copy of the model. No shared state.

    Parameters
    ----------
    start_fen : optional FEN string to start from instead of the initial position.
                Use gen_openings.py to generate a diverse opening book.

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
    board = chess.Board(start_fen) if start_fen else chess.Board()

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
        if abs(val) > resign_threshold:
            resign_counter += 1
            if resign_counter >= resign_consecutive:
                # val > 0 → side to move wins; determine white/black winner
                side_wins = board.turn if val > 0 else (not board.turn)
                outcome = 1.0 if side_wins == chess.WHITE else -1.0
                break
        else:
            resign_counter = 0

        # --- Record position (before pushing move) ---
        # Use .numpy() to avoid PyTorch shared-memory fd passing across pool workers
        positions.append((board.fen(), board_to_tensor(board).numpy(), visit_dist, half_move))

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
    """Pool-compatible wrapper around _play_game (args may include start_fen)."""
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

def play_games(model_path: str, n_games: int, n_sim: int, workers: int,
               opening_fens: list = None,
               resign_threshold: float = RESIGN_THRESHOLD,
               resign_consecutive: int = RESIGN_CONSECUTIVE) -> dict:
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
    outcome_counts  = {"decisive": 0, "draw": 0}   # for reporting

    rng = np.random.default_rng(seed=None)
    if opening_fens:
        sampled_fens = [opening_fens[i % len(opening_fens)]
                        for i in rng.permutation(n_games)]
    else:
        sampled_fens = [None] * n_games
    args_list = [(model_path, n_sim, i, sampled_fens[i],
                  resign_threshold, resign_consecutive)
                 for i in range(n_games)]

    def _collect(result):
        outcome = result["outcome"]
        if outcome == DRAW_VALUE:
            outcome_counts["draw"] += 1
        else:
            outcome_counts["decisive"] += 1
        for fen, tensor, visit_dist, _ in result["positions"]:
            all_fens.append(fen)
            all_tensors.append(torch.from_numpy(tensor))
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
                n_done = i + 1
                dec_pct = 100 * outcome_counts["decisive"] / n_done
                print(f"  [{n_done:4d}/{n_games}]  positions: {len(all_fens):,}  "
                      f"decisive: {dec_pct:.0f}%  ({elapsed:.0f}s)", flush=True)
    else:
        for i, args in enumerate(args_list):
            result = _play_game(*args)
            _collect(result)
            elapsed = time.time() - t0
            n_done = i + 1
            dec_pct = 100 * outcome_counts["decisive"] / n_done
            print(f"  [{n_done:4d}/{n_games}]  positions: {len(all_fens):,}  "
                  f"outcome={result['outcome']:+.1f}  moves={result['n_moves']}  "
                  f"decisive: {dec_pct:.0f}%  ({elapsed:.0f}s)", flush=True)

    n = len(all_fens)
    n_games_done = outcome_counts["decisive"] + outcome_counts["draw"]
    dec_pct  = 100 * outcome_counts["decisive"] / max(1, n_games_done)
    draw_pct = 100 * outcome_counts["draw"]     / max(1, n_games_done)
    print(f"\nTotal positions : {n:,}  ({time.time()-t0:.0f}s)")
    print(f"Outcome summary : decisive={dec_pct:.1f}%  draw={draw_pct:.1f}%")
    if dec_pct < 40:
        print("  WARNING: decisive rate < 40% — value targets may lack diversity. "
              "Consider using --max-moves with a smaller value in gen_openings.py.")
    elif dec_pct > 85:
        print("  WARNING: decisive rate > 85% — draw signal is sparse. "
              "Consider using --min-moves with a larger value in gen_openings.py.")

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
            "source":          "selfplay",
            "n_games":         n_games,
            "n_train":         len(t_idx),
            "n_val":           len(v_idx),
            "n_positions":     n,
            "n_sim":           n_sim,
            "model_path":      model_path,
            "opening_book_sz":   len(opening_fens) if opening_fens else 0,
            "decisive_pct":      round(dec_pct, 1),
            "draw_pct":          round(draw_pct, 1),
            "resign_threshold":  resign_threshold,
            "resign_consecutive": resign_consecutive,
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
    ap.add_argument("--opening-book", type=str, default=None,
                    help="Path to newline-separated FEN file (gen_openings.py output). "
                         "Each game starts from a randomly sampled FEN.")
    ap.add_argument("--pilot", type=int, default=None, metavar="N",
                    help="Run N games, print decisive-rate assessment, then exit "
                         "without saving. Use before a full HPC run to verify the "
                         "opening book produces a healthy outcome mix.")
    ap.add_argument("--resign-threshold",  type=float, default=RESIGN_THRESHOLD,
                    help=f"Model value magnitude to start resign counter (default: {RESIGN_THRESHOLD})")
    ap.add_argument("--resign-consecutive", type=int,  default=RESIGN_CONSECUTIVE,
                    help=f"Consecutive moves above threshold before resigning (default: {RESIGN_CONSECUTIVE})")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    opening_fens = None
    if args.opening_book:
        with open(args.opening_book) as f:
            opening_fens = [line.strip() for line in f if line.strip()]

    print(f"selfplay.py")
    print(f"  model  : {args.model}")
    if args.pilot:
        print(f"  mode   : PILOT ({args.pilot} games — no output saved)")
    else:
        print(f"  games  : {args.games}")
    print(f"  n_sim  : {args.n_sim}")
    print(f"  workers: {args.workers}")
    print(f"  resign : threshold={args.resign_threshold}  consecutive={args.resign_consecutive}")
    if opening_fens:
        print(f"  opening book: {args.opening_book}  ({len(opening_fens)} positions)")
    if not args.pilot:
        print(f"  out    : {args.out}")
    print()

    if args.pilot:
        dataset = play_games(
            model_path=args.model,
            n_games=args.pilot,
            n_sim=args.n_sim,
            workers=args.workers,
            opening_fens=opening_fens,
            resign_threshold=args.resign_threshold,
            resign_consecutive=args.resign_consecutive,
        )
        dec  = dataset["meta"]["decisive_pct"]
        draw = dataset["meta"]["draw_pct"]
        print(f"\n{'='*50}")
        print(f"  PILOT RESULT  ({args.pilot} games)")
        print(f"{'='*50}")
        print(f"  decisive : {dec}%")
        print(f"  draw     : {draw}%")
        if dec < 40:
            verdict = "WARN  — too many draws. Try gen_openings.py --max-moves 12"
        elif dec > 85:
            verdict = "WARN  — too few draws. Try --resign-consecutive 15 or gen_openings.py --min-moves 14"
        else:
            verdict = "OK    — outcome mix looks healthy"
        print(f"  verdict  : {verdict}")
        print(f"{'='*50}")
        import sys; sys.exit(0)

    dataset = play_games(
        model_path=args.model,
        n_games=args.games,
        n_sim=args.n_sim,
        workers=args.workers,
        opening_fens=opening_fens,
        resign_threshold=args.resign_threshold,
        resign_consecutive=args.resign_consecutive,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(dataset, args.out)

    meta = dataset["meta"]
    print(f"\nSaved → {args.out}")
    print(f"  train: {meta['n_train']:,}  val: {meta['n_val']:,}  "
          f"total: {meta['n_positions']:,} positions")
    print(f"  decisive: {meta['decisive_pct']}%  draw: {meta['draw_pct']}%")
    if meta["opening_book_sz"]:
        print(f"  opening book: {meta['opening_book_sz']} positions used")


if __name__ == "__main__":
    main()
