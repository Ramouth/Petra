"""
Geometry space probe for PetraNet.

Checks whether the 128-dim bottleneck has learned meaningful structure:

  1. Eigenvalue distribution  — is variance spread or collapsed?
  2. Win/draw/loss clustering  — do positions with different labels separate?
  3. Known position probes     — do structurally extreme positions (KQ vs K,
                                 equal endgame, opening) land in different regions?
  4. Nearest-neighbour sanity  — do the 5 nearest neighbours of a probe position
                                 share its label more than random?

Usage
-----
    python3 src/probe_geometry.py --model models/best.pt --dataset dataset.pt
    python3 src/probe_geometry.py --model models/best.pt --dataset dataset.pt --n 5000
"""

import argparse
import os
import sys

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import PetraNet
from board import board_to_tensor
from config import device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_geometry_vectors(model: PetraNet, dataset_path: str, n: int = 5000):
    """
    Load n positions from the val set, run through geometry(), return
    (vectors, labels) as numpy arrays.
    """
    data   = torch.load(dataset_path, map_location="cpu", weights_only=False)
    split  = data["val"]
    tensors = split["tensors"][:n].float()   # (N, 14, 8, 8) uint8 → float32
    values  = split["values"][:n].numpy()    # (N,)

    model.eval()
    batch_size = 256
    vecs = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i+batch_size].to(device)
            g = model.geometry(batch)
            vecs.append(g.cpu().numpy())

    vecs = np.concatenate(vecs, axis=0)   # (N, 128)
    return vecs, values


def _label_class(v: float) -> str:
    if abs(v - 1.0)  < 1e-4: return "win"
    if abs(v + 1.0)  < 1e-4: return "loss"
    if abs(v + 0.1)  < 1e-4: return "draw"
    return "invalid"


def board_to_geo(model: PetraNet, board: chess.Board) -> np.ndarray:
    t = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.geometry(t).cpu().numpy()[0]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# Check 1: Eigenvalue distribution
# ---------------------------------------------------------------------------

def check_eigenvalue_distribution(vecs: np.ndarray):
    print("\n" + "="*60)
    print("CHECK 1 — Eigenvalue distribution")
    print("="*60)

    # Mean-centre
    centred = vecs - vecs.mean(axis=0)
    cov = np.cov(centred.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    total   = eigvals.sum() + 1e-8

    top1_pct  = 100 * eigvals[0]  / total
    top5_pct  = 100 * eigvals[:5].sum()  / total
    top10_pct = 100 * eigvals[:10].sum() / total

    print(f"  Top-1  eigenvalue: {top1_pct:.1f}% of variance")
    print(f"  Top-5  eigenvalues: {top5_pct:.1f}% of variance")
    print(f"  Top-10 eigenvalues: {top10_pct:.1f}% of variance")
    print(f"  Mean vector norm: {np.linalg.norm(vecs, axis=1).mean():.4f}")

    # Diagnosis
    if top1_pct > 50:
        print(f"\n  COLLAPSED — top-1 dim dominates ({top1_pct:.1f}%)")
        print(f"  (Phase 0 collapsed encoder had top-1 = 77.8%)")
    elif top1_pct > 25:
        print(f"\n  PARTIAL — some concentration but not collapsed")
    else:
        print(f"\n  HEALTHY — variance spread across dimensions")

    return eigvals


# ---------------------------------------------------------------------------
# Check 2: Win / draw / loss separation
# ---------------------------------------------------------------------------

def check_label_separation(vecs: np.ndarray, values: np.ndarray):
    print("\n" + "="*60)
    print("CHECK 2 — Win / draw / loss separation")
    print("="*60)

    labels = np.array([_label_class(v) for v in values])
    win_vecs  = vecs[labels == "win"]
    loss_vecs = vecs[labels == "loss"]
    draw_vecs = vecs[labels == "draw"]

    print(f"  Samples — win: {len(win_vecs)}, draw: {len(draw_vecs)}, loss: {len(loss_vecs)}")

    # Centroids
    c_win  = win_vecs.mean(axis=0)
    c_loss = loss_vecs.mean(axis=0)
    c_draw = draw_vecs.mean(axis=0) if len(draw_vecs) > 0 else None

    win_loss_sim  = cosine_sim(c_win, c_loss)
    print(f"\n  Centroid cosine similarities:")
    print(f"    win  · loss  = {win_loss_sim:.4f}  (lower = better separation)")
    if c_draw is not None:
        print(f"    win  · draw  = {cosine_sim(c_win,  c_draw):.4f}")
        print(f"    loss · draw  = {cosine_sim(c_loss, c_draw):.4f}")

    # Within-class vs between-class distances (sample 500 per class)
    rng = np.random.default_rng(42)
    n_sample = min(500, len(win_vecs), len(loss_vecs))
    w = win_vecs[rng.choice(len(win_vecs),   n_sample, replace=False)]
    l = loss_vecs[rng.choice(len(loss_vecs), n_sample, replace=False)]

    within_win  = np.mean([cosine_sim(w[i], w[j])
                           for i, j in zip(range(0, n_sample, 2), range(1, n_sample, 2))])
    within_loss = np.mean([cosine_sim(l[i], l[j])
                           for i, j in zip(range(0, n_sample, 2), range(1, n_sample, 2))])
    between     = np.mean([cosine_sim(w[i], l[i]) for i in range(n_sample)])

    print(f"\n  Within-class cosine similarity (higher = tighter cluster):")
    print(f"    win  pairs: {within_win:.4f}")
    print(f"    loss pairs: {within_loss:.4f}")
    print(f"  Between-class (win vs loss): {between:.4f}")
    print(f"  Separation gap: {((within_win + within_loss) / 2 - between):.4f}")

    gap = (within_win + within_loss) / 2 - between
    if gap > 0.05:
        print(f"\n  GOOD — win/loss positions occupy distinct regions")
    elif gap > 0.01:
        print(f"\n  WEAK — some separation but small")
    else:
        print(f"\n  NONE — win/loss positions are not separated")

    return c_win, c_loss


# ---------------------------------------------------------------------------
# Check 3: Known position probes
# ---------------------------------------------------------------------------

def check_known_positions(model: PetraNet, c_win: np.ndarray, c_loss: np.ndarray):
    print("\n" + "="*60)
    print("CHECK 3 — Known position probes")
    print("="*60)

    probes = [
        ("Starting position",           chess.Board()),
        ("KQ vs K — White to move",     chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1")),
        ("KQ vs K — Black to move",     chess.Board("4k3/8/8/8/8/8/8/4K2Q b - - 0 1")),
        ("Equal endgame (KR vs KR)",    chess.Board("4k3/8/8/8/8/8/8/R3K2R w - - 0 1")),
        ("White queen up",              chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")),
        ("Black queen up",              chess.Board("4K3/8/8/8/8/8/8/q3k3 w - - 0 1")),
        ("Complex middlegame",          chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")),
    ]

    print(f"  {'Position':<35}  {'→win':>6}  {'→loss':>6}  {'value':>7}  label")
    print(f"  {'-'*35}  {'------':>6}  {'------':>6}  {'-------':>7}  -----")

    for name, board in probes:
        g   = board_to_geo(model, board)
        sw  = cosine_sim(g, c_win)
        sl  = cosine_sim(g, c_loss)
        val = model.value(board, device)
        dominant = "win" if sw > sl else "loss"
        print(f"  {name:<35}  {sw:>6.3f}  {sl:>6.3f}  {val:>+7.3f}  → {dominant}")


# ---------------------------------------------------------------------------
# Check 4: Nearest-neighbour label consistency
# ---------------------------------------------------------------------------

def check_nearest_neighbours(vecs: np.ndarray, values: np.ndarray, k: int = 5, n_probe: int = 200):
    print("\n" + "="*60)
    print("CHECK 4 — Nearest-neighbour label consistency")
    print("="*60)
    print(f"  For {n_probe} random positions: are the {k} nearest neighbours")
    print(f"  more likely to share the same label than random chance?")

    labels = np.array([_label_class(v) for v in values])
    rng    = np.random.default_rng(42)
    idxs   = rng.choice(len(vecs), n_probe, replace=False)

    # Normalise for cosine similarity
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    normed = vecs / norms

    match_rates = []
    for idx in idxs:
        sims     = normed @ normed[idx]
        sims[idx] = -1.0   # exclude self
        top_k    = np.argsort(sims)[-k:]
        same     = sum(labels[j] == labels[idx] for j in top_k)
        match_rates.append(same / k)

    mean_match = np.mean(match_rates)

    # Baseline: random label match rate
    label_counts = {l: (labels == l).sum() for l in ["win", "loss", "draw"]}
    n = len(labels)
    baseline = sum((c / n) ** 2 for c in label_counts.values())

    print(f"\n  Mean NN label match rate : {mean_match:.3f}")
    print(f"  Random baseline          : {baseline:.3f}")
    print(f"  Lift                     : {mean_match - baseline:+.3f}")

    if mean_match - baseline > 0.1:
        print(f"\n  GOOD — nearest neighbours share labels more than chance")
    elif mean_match - baseline > 0.02:
        print(f"\n  WEAK — slight tendency, not strong")
    else:
        print(f"\n  NONE — geometry does not cluster by label")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True, help="Path to model .pt")
    ap.add_argument("--dataset", required=True, help="Path to dataset .pt")
    ap.add_argument("--n",       type=int, default=5000,
                    help="Number of val positions to probe (default 5000)")
    args = ap.parse_args()

    model = PetraNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {args.model}")

    print(f"Extracting geometry vectors for {args.n} val positions ...")
    vecs, values = load_geometry_vectors(model, args.dataset, n=args.n)
    print(f"  Done. Shape: {vecs.shape}")

    check_eigenvalue_distribution(vecs)
    c_win, c_loss = check_label_separation(vecs, values)
    check_known_positions(model, c_win, c_loss)
    check_nearest_neighbours(vecs, values)

    print("\n" + "="*60)
    print("Probe complete.")
    print("="*60)


if __name__ == "__main__":
    main()
