"""
Cross-round geometry comparison for PetraNet.

Runs the geometry probe across multiple model checkpoints on a fixed reference
dataset and outputs a summary table. Use this after each round to track whether
the encoder geometry is actually improving.

Usage
-----
    python3 src/compare_geometry.py --dataset data/selfplay_r1_full_sf.pt
    python3 src/compare_geometry.py --dataset data/selfplay_r1_full_sf.pt --rounds r1 r2 r4 r5
    python3 src/compare_geometry.py --dataset data/selfplay_r5.pt --rounds r4 r5
"""

import argparse
import os
import sys

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PetraNet
from board import board_to_tensor
from config import device
from probe_geometry import (
    load_geometry_vectors, check_eigenvalue_distribution,
    check_label_separation, board_to_geo, cosine_sim,
    check_nearest_neighbours
)


# ---------------------------------------------------------------------------
# Known position probes — extended set
# ---------------------------------------------------------------------------

KNOWN_POSITIONS = [
    # Basic material probes (should always label correctly)
    ("KQ vs K (W to move)",  chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1"),  "win"),
    ("KQ vs K (B to move)",  chess.Board("4k3/8/8/8/8/8/8/4K2Q b - - 0 1"),  "win"),
    ("K vs KQ (W to move)",  chess.Board("4K3/8/8/8/8/8/8/4k2q w - - 0 1"),  "loss"),
    ("KR vs K (W to move)",  chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 0 1"),  "win"),
    ("White queen up",       chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1"),  "win"),
    ("Black queen up",       chess.Board("4K3/8/8/8/8/8/8/q3k3 w - - 0 1"),  "loss"),
    # Structural probes
    ("Starting position",    chess.Board(),                                    None),
    ("Equal KR vs KR",       chess.Board("4k3/8/8/8/8/8/8/R3K2R w - - 0 1"), None),
    ("Complex middlegame",   chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"), None),
]


def probe_round(model_path: str, dataset_path: str, n: int = 2000) -> dict:
    """Run all geometry checks for one model. Returns a results dict."""
    model = PetraNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    vecs, values = load_geometry_vectors(model, dataset_path, n=n)

    # Check 1: eigenvalue distribution
    eigvals = check_eigenvalue_distribution.__wrapped__(vecs) if hasattr(check_eigenvalue_distribution, '__wrapped__') else _eigen(vecs)
    total = eigvals.sum() + 1e-8
    top1_pct = 100 * eigvals[0] / total

    # Check 2: separation
    labels = np.array([_label_class(v) for v in values])
    win_vecs  = vecs[labels == "win"]
    loss_vecs = vecs[labels == "loss"]

    if len(win_vecs) < 2 or len(loss_vecs) < 2:
        sep_gap = float('nan')
        centroid_sim = float('nan')
        c_win = c_loss = None
    else:
        c_win  = win_vecs.mean(axis=0)
        c_loss = loss_vecs.mean(axis=0)
        centroid_sim = cosine_sim(c_win, c_loss)

        rng = np.random.default_rng(42)
        n_s = min(500, len(win_vecs), len(loss_vecs))
        w = win_vecs[rng.choice(len(win_vecs),   n_s, replace=False)]
        l = loss_vecs[rng.choice(len(loss_vecs), n_s, replace=False)]
        within_win  = np.mean([cosine_sim(w[i], w[j]) for i, j in zip(range(0, n_s, 2), range(1, n_s, 2))])
        within_loss = np.mean([cosine_sim(l[i], l[j]) for i, j in zip(range(0, n_s, 2), range(1, n_s, 2))])
        between     = np.mean([cosine_sim(w[i], l[i]) for i in range(n_s)])
        sep_gap = (within_win + within_loss) / 2 - between

    # Check 4: NN consistency
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    normed = vecs / norms
    rng2   = np.random.default_rng(42)
    n_probe = min(200, len(vecs))
    idxs   = rng2.choice(len(vecs), n_probe, replace=False)
    match_rates = []
    for idx in idxs:
        sims = normed @ normed[idx]
        sims[idx] = -1.0
        top_k = np.argsort(sims)[-5:]
        match_rates.append(sum(labels[j] == labels[idx] for j in top_k) / 5)
    nn_consistency = np.mean(match_rates)

    # Check 3: known positions
    known_results = {}
    if c_win is not None and c_loss is not None:
        for name, board, expected in KNOWN_POSITIONS:
            g  = board_to_geo(model, board)
            sw = cosine_sim(g, c_win)
            sl = cosine_sim(g, c_loss)
            val = model.value(board, device)
            predicted = "win" if sw > sl else "loss"
            correct = (expected is None) or (predicted == expected)
            known_results[name] = {"predicted": predicted, "expected": expected,
                                   "correct": correct, "value": val,
                                   "win_sim": sw, "loss_sim": sl}

    return {
        "top1_pct":      top1_pct,
        "centroid_sim":  centroid_sim,
        "sep_gap":       sep_gap,
        "nn_consistency": nn_consistency,
        "vec_norm":      np.linalg.norm(vecs, axis=1).mean(),
        "n_win":         int((labels == "win").sum()),
        "n_loss":        int((labels == "loss").sum()),
        "n_draw":        int((labels == "draw").sum()),
        "known":         known_results,
    }


def _label_class(v: float) -> str:
    if abs(v - 1.0)  < 1e-4: return "win"
    if abs(v + 1.0)  < 1e-4: return "loss"
    if abs(abs(v) - 0.1) < 1e-4: return "draw"   # catches ±0.1 after perspective flip
    return "invalid"


def _eigen(vecs):
    centred = vecs - vecs.mean(axis=0)
    cov = np.cov(centred.T)
    eigvals = np.linalg.eigvalsh(cov)
    return np.sort(eigvals)[::-1]


def print_comparison(results: dict):
    rounds = list(results.keys())

    print("\n" + "="*70)
    print("GEOMETRY COMPARISON ACROSS ROUNDS")
    print("="*70)

    # Summary table
    header = f"{'Metric':<28}" + "".join(f"  {r:>8}" for r in rounds)
    print(f"\n{header}")
    print("-" * len(header))

    def row(name, key, fmt=".4f", lower_is_better=False):
        vals = []
        for r in rounds:
            v = results[r].get(key, float('nan'))
            vals.append(v)
        # Find best
        valid = [(i, v) for i, v in enumerate(vals) if not (isinstance(v, float) and np.isnan(v))]
        if valid:
            best_i = min(valid, key=lambda x: x[1])[0] if lower_is_better else max(valid, key=lambda x: x[1])[0]
        else:
            best_i = -1
        line = f"  {name:<26}"
        for i, v in enumerate(vals):
            if np.isnan(v):
                s = "     N/A"
            else:
                s = f"  {v:{fmt}}"
                if i == best_i and len(rounds) > 1:
                    s = f"  {v:{fmt}}*"
        # redo cleanly
        line = f"  {name:<26}"
        for i, v in enumerate(vals):
            if isinstance(v, float) and np.isnan(v):
                cell = "     N/A"
            else:
                cell = f"  {v:{fmt}}"
            line += cell
        print(line)

    row("Top-1 eigenvalue (%)",    "top1_pct",      ".1f",  lower_is_better=True)
    row("Centroid sim (↓better)",  "centroid_sim",  ".4f",  lower_is_better=True)
    row("Separation gap (↑better)","sep_gap",        ".4f",  lower_is_better=False)
    row("NN consistency (↑better)","nn_consistency", ".4f",  lower_is_better=False)
    row("Mean vec norm",            "vec_norm",       ".2f",  lower_is_better=False)

    # Known position pass/fail
    print(f"\n{'Known position probes':}")
    print(f"  {'Position':<30}" + "".join(f"  {r:>8}" for r in rounds))
    print("  " + "-" * (28 + 10 * len(rounds)))

    all_positions = set()
    for r in rounds:
        all_positions.update(results[r].get("known", {}).keys())

    for pos in all_positions:
        expected = None
        for r in rounds:
            kp = results[r].get("known", {}).get(pos)
            if kp and kp["expected"]:
                expected = kp["expected"]
                break
        line = f"  {pos:<30}"
        for r in rounds:
            kp = results[r].get("known", {}).get(pos)
            if kp is None:
                cell = "     N/A"
            elif kp["expected"] is None:
                cell = f"  {kp['predicted']:>8}"
            else:
                cell = f"  {'✓' if kp['correct'] else '✗':>3} {kp['predicted']:>4}"
        # redo
        line = f"  {pos:<30}"
        for r in rounds:
            kp = results[r].get("known", {}).get(pos)
            if kp is None:
                cell = "       -"
            elif kp["expected"] is None:
                cell = f"  {kp['predicted']:>6}"
            else:
                mark = "✓" if kp["correct"] else "✗"
                cell = f"  {mark} {kp['predicted']:>4}"
            line += cell
        print(line)

    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Fixed reference dataset .pt (same for all rounds)")
    ap.add_argument("--rounds", nargs="+", default=None,
                    help="Round names (e.g. r1 r2 r4 r5). Auto-detects if omitted.")
    ap.add_argument("--model-dir", default="../models/zigzag",
                    help="Base directory containing round subdirs (default: ../models/zigzag)")
    ap.add_argument("--n", type=int, default=2000,
                    help="Number of val positions to use (default 2000)")
    args = ap.parse_args()

    # Auto-detect rounds
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model_dir)
    if args.rounds is None:
        args.rounds = sorted([
            d for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d)) and
               os.path.exists(os.path.join(model_dir, d, "best.pt"))
        ])

    print(f"Reference dataset: {args.dataset}")
    print(f"Rounds: {args.rounds}")

    results = {}
    for r in args.rounds:
        model_path = os.path.join(model_dir, r, "best.pt")
        if not os.path.exists(model_path):
            print(f"  Skipping {r} — no best.pt found")
            continue
        print(f"\nProbing {r}...")
        results[r] = probe_round(model_path, args.dataset, n=args.n)

    print_comparison(results)


if __name__ == "__main__":
    main()
