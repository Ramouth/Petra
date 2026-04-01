"""
Endgame conversion test for geometry-trained models.

Measures how reliably and efficiently the model converts known-winning endgame
positions — not general play strength. This is the right gate for a model trained
exclusively on endgame curricula.

Metric: mate rate (% of positions converted within max_moves) and average moves
to mate. Compared against material-MCTS baseline on the same positions.

Stages tested (mirrors generate_endgame.py):
  1: KQ vs K  — queen endgame conversion
  2: KR vs K  — rook endgame conversion
  3: KP vs K  — pawn promotion (noisy — many theoretical draws)

Usage
-----
    python3 src/eval_endgame.py --model models/geometry/stage3/best.pt
    python3 src/eval_endgame.py --model models/geometry/stage3/best.pt \\
        --positions 200 --n-sim 400 --workers 8 --stages 1 2
"""

import argparse
import math
import multiprocessing as mp
import os
import random
import sys
import time

import chess
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PetraNet
from mcts import MCTS
from board import board_to_tensor
from config import device
from evaluate import material_value, RandomAgent
from generate_endgame import random_kqk_position, random_krk_position, random_kpk_position


# ---------------------------------------------------------------------------
# Position generators (winning side always White for simplicity)
# ---------------------------------------------------------------------------

_STAGE_GEN = {
    1: (lambda: random_kqk_position(white_has_queen=True), "KQ vs K"),
    2: (lambda: random_krk_position(white_has_rook=True),  "KR vs K"),
    3: (lambda: random_kpk_position(white_has_pawn=True),  "KP vs K"),
}


def _generate_positions_simple(n: int, stages: list, seed: int = 42) -> list:
    """
    Generate n positions (FEN strings) distributed evenly across stages.
    White is always the winning side (White to move).
    Returns list of (fen, stage) tuples.
    """
    rng = random.Random(seed)
    n_per_stage = [n // len(stages)] * len(stages)
    n_per_stage[-1] += n - sum(n_per_stage)

    positions = []
    for stage, n_stage in zip(stages, n_per_stage):
        gen_fn, _ = _STAGE_GEN[stage]
        seen = set()
        count = 0
        while count < n_stage:
            board = gen_fn()
            fen = board.fen()
            if fen not in seen:
                seen.add(fen)
                positions.append((fen, stage))
                count += 1

    rng.shuffle(positions)
    return positions


# ---------------------------------------------------------------------------
# Single game worker (module-level for picklability)
# ---------------------------------------------------------------------------

def _conversion_worker(args):
    """
    Play one endgame conversion game.
    White = agent under test (learned value or material MCTS).
    Black = RandomAgent (bare king, just shuffles).

    Returns dict with outcome info.
    """
    pos_idx, fen, stage, model_path, agent_type, n_sim, max_moves, seed = args

    model = PetraNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    # learned: use model.value() — the full CNN → geometry → value head pipeline
    # material: hardcoded piece counts, no neural network
    value_fn = None if agent_type == "learned" else material_value

    mcts = MCTS(model, torch.device("cpu"), value_fn=value_fn)
    black = RandomAgent(seed=seed)

    board = chess.Board(fen)
    n_moves = 0

    for _ in range(max_moves):
        if board.is_game_over():
            break
        if board.turn == chess.WHITE:
            move, _ = mcts.search(board, n_simulations=n_sim,
                                  temperature=0.0, add_noise=False)
        else:
            move = black.select_move(board)
        board.push(move)
        n_moves += 1

    outcome = board.outcome()
    if outcome is not None and outcome.winner == chess.WHITE:
        result = "mate"
    elif outcome is not None and outcome.winner is None:
        result = "stalemate"
    else:
        result = "move_limit"

    return {
        "pos_idx":  pos_idx,
        "stage":    stage,
        "result":   result,
        "n_moves":  n_moves,
        "agent":    agent_type,
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_conversion_eval(model_path: str, positions: list,
                        n_sim: int = 200, max_moves: int = 100,
                        workers: int = 1) -> dict:
    """
    Run geometry-MCTS and material-MCTS on the same positions.
    Returns comparison stats.
    """
    results = {"learned": [], "material": []}

    for agent_type in ("learned", "material"):
        args_list = [
            (i, fen, stage, model_path, agent_type, n_sim, max_moves, i + 1000)
            for i, (fen, stage) in enumerate(positions)
        ]

        t0 = time.time()
        print(f"\n  Running {agent_type} agent ({len(positions)} positions) ...")

        if workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                for r in pool.imap(_conversion_worker, args_list):
                    results[agent_type].append(r)
                    done = len(results[agent_type])
                    if done % max(1, len(positions) // 5) == 0:
                        mates = sum(1 for x in results[agent_type] if x["result"] == "mate")
                        print(f"    [{done:>4}/{len(positions)}]  "
                              f"mates={mates}  ({time.time()-t0:.0f}s)")
        else:
            for i, a in enumerate(args_list):
                r = _conversion_worker(a)
                results[agent_type].append(r)
                if (i + 1) % max(1, len(positions) // 5) == 0:
                    mates = sum(1 for x in results[agent_type] if x["result"] == "mate")
                    print(f"    [{i+1:>4}/{len(positions)}]  mates={mates}  "
                          f"({time.time()-t0:.0f}s)")

    return results


def _stage_stats(records: list, stage: int) -> dict:
    subset = [r for r in records if r["stage"] == stage]
    if not subset:
        return None
    n = len(subset)
    mates = [r for r in subset if r["result"] == "mate"]
    stalemates = [r for r in subset if r["result"] == "stalemate"]
    move_limits = [r for r in subset if r["result"] == "move_limit"]
    mate_moves = [r["n_moves"] for r in mates]
    return {
        "n":           n,
        "mate_rate":   len(mates) / n,
        "avg_moves":   sum(mate_moves) / len(mate_moves) if mate_moves else None,
        "stalemate_rate": len(stalemates) / n,
        "move_limit_rate": len(move_limits) / n,
    }


def print_results(results: dict, stages: list):
    print("\n" + "=" * 62)
    print("ENDGAME CONVERSION RESULTS")
    print("=" * 62)
    print(f"  {'Stage':<12}  {'Agent':<10}  {'Mate%':>6}  {'AvgMoves':>9}  "
          f"{'Stale%':>7}  {'Limit%':>7}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*6}  {'-'*9}  {'-'*7}  {'-'*7}")

    for stage in stages:
        _, label = _STAGE_GEN[stage]
        for agent in ("learned", "material"):
            s = _stage_stats(results[agent], stage)
            if s is None:
                continue
            avg = f"{s['avg_moves']:.1f}" if s["avg_moves"] is not None else "  —"
            print(f"  {label:<12}  {agent:<10}  "
                  f"{s['mate_rate']*100:>5.1f}%  "
                  f"{avg:>9}  "
                  f"{s['stalemate_rate']*100:>6.1f}%  "
                  f"{s['move_limit_rate']*100:>6.1f}%")
        print()

    # Overall verdict
    print("=" * 62)
    learned_all = results["learned"]
    mat_all     = results["material"]
    learned_rate = sum(1 for r in learned_all if r["result"] == "mate") / len(learned_all)
    mat_rate     = sum(1 for r in mat_all     if r["result"] == "mate") / len(mat_all)
    delta = learned_rate - mat_rate

    learned_moves = [r["n_moves"] for r in learned_all if r["result"] == "mate"]
    mat_moves     = [r["n_moves"] for r in mat_all     if r["result"] == "mate"]
    learned_avg = sum(learned_moves) / len(learned_moves) if learned_moves else None
    mat_avg     = sum(mat_moves)     / len(mat_moves)     if mat_moves     else None

    print(f"  Overall mate rate:  learned={learned_rate*100:.1f}%  "
          f"material={mat_rate*100:.1f}%  Δ={delta*100:+.1f}%")
    if learned_avg and mat_avg:
        print(f"  Average moves:     learned={learned_avg:.1f}  "
              f"material={mat_avg:.1f}  Δ={learned_avg-mat_avg:+.1f}")
    print("=" * 62)

    if delta > 0.05:
        print("\n  MODEL CONVERTS BETTER — learned value drives endgame play.")
    elif delta > -0.05:
        print("\n  MODEL COMPARABLE — similar conversion rate to material.")
    else:
        print("\n  MODEL WORSE — learned value does not drive endgame conversion.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     required=True, help="Path to geometry model .pt")
    ap.add_argument("--positions", type=int, default=100,
                    help="Number of test positions per stage (default: 100)")
    ap.add_argument("--n-sim",     type=int, default=200,
                    help="MCTS simulations per move (default: 200)")
    ap.add_argument("--max-moves", type=int, default=100,
                    help="Max half-moves before declaring move_limit (default: 100)")
    ap.add_argument("--workers",   type=int, default=1,
                    help="Parallel workers (default: 1)")
    ap.add_argument("--stages",    type=int, nargs="+", default=[1, 2],
                    help="Endgame stages to test: 1=KQK 2=KRK 3=KPK (default: 1 2)")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    for s in args.stages:
        if s not in _STAGE_GEN:
            print(f"Unknown stage {s}. Supported: {list(_STAGE_GEN)}")
            sys.exit(1)

    print(f"Endgame conversion eval")
    print(f"  Model:     {args.model}")
    print(f"  Stages:    {[_STAGE_GEN[s][1] for s in args.stages]}")
    print(f"  Positions: {args.positions} per stage")
    print(f"  n_sim:     {args.n_sim}  max_moves: {args.max_moves}")

    print("\nGenerating test positions ...")
    random.seed(args.seed)
    positions = _generate_positions_simple(
        args.positions * len(args.stages), args.stages, seed=args.seed
    )
    print(f"  {len(positions)} positions generated")

    results = run_conversion_eval(
        model_path=args.model,
        positions=positions,
        n_sim=args.n_sim,
        max_moves=args.max_moves,
        workers=args.workers,
    )

    print_results(results, args.stages)


if __name__ == "__main__":
    main()
