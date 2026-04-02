"""
Geometry consistency tests for PetraNet.

Tests structural properties the geometry should satisfy if it is encoding
chess value correctly. These are model-only tests — no dataset required.
Each test is pass/fail with a numeric score so results are comparable across
model checkpoints.

Tests
-----
  1. Material monotonicity   — adding stronger pieces moves geometry toward win
  2. Piece value ordering    — Q > R > B ≈ N > P in geometry projection
  3. Antipodal symmetry      — g(pos) ≈ -g(color_flip(pos)) (STM-sign fix makes this achievable)
  4. Forced mate convergence — as mate approaches, win projection increases
  5. Transposition consistency — same position via different move orders = same geometry

Usage
-----
    python3 src/test_geometry.py --model models/zigzag/r4/best.pt
    python3 src/test_geometry.py --model models/zigzag/r6/best.pt
"""

import argparse
import os
import sys

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PetraNet, ConvBlock, ResBlock
from board import board_to_tensor
from config import device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def geo(model: PetraNet, board: chess.Board) -> np.ndarray:
    t = board_to_tensor(board).unsqueeze(0).float().to(device)
    return model.geometry(t).cpu().numpy()[0]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def win_projection(g: np.ndarray, c_win: np.ndarray, c_loss: np.ndarray) -> float:
    """Scalar projection of g onto the win/loss axis. Positive = toward win."""
    axis = c_win - c_loss
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return float(np.dot(g, axis))


def color_flip(board: chess.Board) -> chess.Board:
    """Return the board with colors swapped (mirror position).
    Under STM-relative geometry, mirror preserves the label (+1 for STM in both
    cases), so mirrored positions should have SIMILAR geometry, not antipodal.
    """
    return board.mirror()


def win_loss_centroids(model: PetraNet) -> tuple:
    """
    Compute win/loss centroids from a fixed set of unambiguous positions.
    KQ vs K (white winning) and KQ vs K (black winning) as anchors.
    """
    win_positions = [
        chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1"),  # white KQ vs K, white to move
        chess.Board("4k3/8/8/2Q5/8/8/8/4K3 w - - 0 1"),  # queen on c5
        chess.Board("8/8/8/8/3k4/8/8/3K2Q1 w - - 0 1"),  # queen on g1
        chess.Board("8/8/8/8/8/2k5/8/3K2Q1 w - - 0 1"),  # queen on g1, king c3
    ]
    loss_positions = [
        chess.Board("4K3/8/8/8/8/8/8/4k2q w - - 0 1"),  # white bare king, black has queen, white to move
        chess.Board("4K3/8/8/2q5/8/8/8/4k3 w - - 0 1"),  # queen on c5
        chess.Board("8/8/8/8/3K4/8/8/3k2q1 w - - 0 1"),  # queen on g1
        chess.Board("8/8/8/8/8/2K5/8/3k2q1 w - - 0 1"),  # queen on g1
    ]
    c_win  = np.mean([geo(model, b) for b in win_positions],  axis=0)
    c_loss = np.mean([geo(model, b) for b in loss_positions], axis=0)
    return c_win, c_loss


PASS  = "✓ PASS"
FAIL  = "✗ FAIL"
WARN  = "~ WARN"


# ---------------------------------------------------------------------------
# Test 1: Material monotonicity
# ---------------------------------------------------------------------------

def test_material_monotonicity(model: PetraNet, c_win: np.ndarray, c_loss: np.ndarray):
    print("\n" + "="*60)
    print("TEST 1 — Material monotonicity")
    print("  Adding stronger pieces should increase win projection.")
    print("="*60)

    base = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")  # bare kings

    pieces = [
        (chess.PAWN,   chess.WHITE, chess.E4, "White pawn   (e4)"),
        (chess.KNIGHT, chess.WHITE, chess.D4, "White knight (d4)"),
        (chess.BISHOP, chess.WHITE, chess.C1, "White bishop (c1)"),
        (chess.ROOK,   chess.WHITE, chess.A1, "White rook   (a1)"),
        (chess.QUEEN,  chess.WHITE, chess.D1, "White queen  (d1)"),
    ]

    base_proj = win_projection(geo(model, base), c_win, c_loss)
    print(f"\n  {'Position':<30}  {'win-proj':>8}  {'delta':>8}")
    print(f"  {'-'*30}  {'--------':>8}  {'--------':>8}")
    print(f"  {'Bare kings (baseline)':<30}  {base_proj:>+8.4f}  {'':>8}")

    projections = [base_proj]
    for piece_type, color, square, label in pieces:
        b = base.copy()
        b.set_piece_at(square, chess.Piece(piece_type, color))
        p = win_projection(geo(model, b), c_win, c_loss)
        delta = p - base_proj
        projections.append(p)
        print(f"  {label:<30}  {p:>+8.4f}  {delta:>+8.4f}")

    # Check monotonically increasing
    passing = all(projections[i] < projections[i+1] for i in range(len(projections)-1))
    result = PASS if passing else FAIL
    print(f"\n  Result: {result}  (monotonically increasing: {passing})")
    return passing


# ---------------------------------------------------------------------------
# Test 2: Piece value ordering
# ---------------------------------------------------------------------------

def test_piece_value_ordering(model: PetraNet, c_win: np.ndarray, c_loss: np.ndarray):
    print("\n" + "="*60)
    print("TEST 2 — Piece value ordering")
    print("  Geometry should respect Q > R > B ≈ N > P.")
    print("="*60)

    base = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")

    piece_positions = [
        (chess.PAWN,   chess.WHITE, chess.E4, "Pawn"),
        (chess.KNIGHT, chess.WHITE, chess.D4, "Knight"),
        (chess.BISHOP, chess.WHITE, chess.C1, "Bishop"),
        (chess.ROOK,   chess.WHITE, chess.A1, "Rook"),
        (chess.QUEEN,  chess.WHITE, chess.D1, "Queen"),
    ]

    projs = {}
    print(f"\n  {'Piece':<10}  {'win-proj':>8}")
    print(f"  {'-'*10}  {'--------':>8}")
    for piece_type, color, square, label in piece_positions:
        b = base.copy()
        b.set_piece_at(square, chess.Piece(piece_type, color))
        p = win_projection(geo(model, b), c_win, c_loss)
        projs[label] = p
        print(f"  {label:<10}  {p:>+8.4f}")

    checks = [
        ("Q > R",  projs["Queen"]  > projs["Rook"]),
        ("R > B",  projs["Rook"]   > projs["Bishop"]),
        ("R > N",  projs["Rook"]   > projs["Knight"]),
        ("B > P",  projs["Bishop"] > projs["Pawn"]),
        ("N > P",  projs["Knight"] > projs["Pawn"]),
    ]

    all_pass = True
    print()
    for label, ok in checks:
        status = PASS if ok else FAIL
        print(f"  {status}  {label}")
        if not ok:
            all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Test 3: Antipodal symmetry
# ---------------------------------------------------------------------------

def test_antipodal_symmetry(model: PetraNet):
    print("\n" + "="*60)
    print("TEST 3 — Antipodal symmetry")
    print("  Flipping colors should negate the geometry vector.")
    print("  With the STM-sign fix, g(pos) ≈ -g(color_flip(pos)) by construction.")
    print("  cosine_sim(g, g_flip) should be close to -1.")
    print("="*60)

    positions = [
        ("KQ vs K (white winning)", chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1")),
        ("Up rook",                 chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")),
        ("Up two pawns",            chess.Board("4k3/8/8/8/8/8/PP6/4K3 w - - 0 1")),
        ("Equal (KR vs KR)",        chess.Board("r3k3/8/8/8/8/8/8/R3K3 w - - 0 1")),
        ("Starting position",       chess.Board()),
    ]

    print(f"\n  {'Position':<26}  {'dot(g,flip(g))':>14}  {'cosine sim':>10}  result")
    print(f"  {'-'*26}  {'-'*14}  {'-'*10}  ------")

    cos_sims = []
    for label, board in positions:
        g      = geo(model, board)
        g_flip = geo(model, color_flip(board))
        dot    = float(np.dot(g, g_flip))
        cos    = cosine_sim(g, g_flip)
        cos_sims.append(cos)
        ok = cos < -0.5
        status = PASS if ok else (WARN if cos < 0.0 else FAIL)
        print(f"  {label:<26}  {dot:>+14.4f}  {cos:>+10.4f}  {status}")

    mean_cos = float(np.mean(cos_sims))
    overall_ok = mean_cos < -0.5
    overall = PASS if overall_ok else (WARN if mean_cos < 0.0 else FAIL)
    print(f"\n  Mean cosine similarity: {mean_cos:.4f}  →  {overall}")
    print(f"  (Perfect antipodal = -1.0, uncorrelated = 0.0, same = +1.0)")
    return overall_ok


# ---------------------------------------------------------------------------
# Test 4: Forced mate convergence
# ---------------------------------------------------------------------------

def test_forced_mate_convergence(model: PetraNet, c_win: np.ndarray, c_loss: np.ndarray):
    print("\n" + "="*60)
    print("TEST 4 — Forced mate convergence")
    print("  As mate approaches (KQ vs K), win projection should increase.")
    print("="*60)

    # KQ vs K positions at decreasing distance from mate
    # Roughly ordered from "queen just entered" to "mate in 1"
    positions = [
        ("Queen enters endgame",     chess.Board("8/8/8/8/4k3/8/8/Q3K3 w - - 0 1")),
        ("Queen closer",             chess.Board("8/8/8/4Q3/4k3/8/8/4K3 w - - 0 1")),
        ("King cooperating",         chess.Board("8/8/5Q2/8/3k4/4K3/8/8 w - - 0 1")),
        ("Near mate",                chess.Board("8/8/8/8/7k/6Q1/8/6K1 w - - 0 1")),
        ("Mate in 1",                chess.Board("7k/8/6KQ/8/8/8/8/8 w - - 0 1")),
    ]

    print(f"\n  {'Position':<28}  {'win-proj':>8}  {'value':>7}")
    print(f"  {'-'*28}  {'--------':>8}  {'-------':>7}")

    projections = []
    for label, board in positions:
        g = geo(model, board)
        p = win_projection(g, c_win, c_loss)
        v = model.value(board, device)
        projections.append(p)
        print(f"  {label:<28}  {p:>+8.4f}  {v:>+7.3f}")

    # Check general upward trend (allow one non-monotone step)
    violations = sum(projections[i] >= projections[i+1] for i in range(len(projections)-1))
    passing = violations <= 1
    result = PASS if passing else (WARN if violations <= 2 else FAIL)
    print(f"\n  Result: {result}  (non-monotone steps: {violations}/4)")
    return passing


# ---------------------------------------------------------------------------
# Test 5: Transposition consistency
# ---------------------------------------------------------------------------

def test_transposition_consistency(model: PetraNet):
    print("\n" + "="*60)
    print("TEST 5 — Transposition consistency")
    print("  Same position via different move orders should give same geometry.")
    print("="*60)

    transpositions = [
        ("Italian Game", [
            # Via 1.e4 e5 2.Nf3 Nc6 3.Bc4
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
            # Via 1.e4 e5 2.Bc4 Nc6 3.Nf3
            ["e2e4", "e7e5", "f1c4", "b8c6", "g1f3"],
        ]),
        ("Symmetric pawn structure", [
            ["e2e4", "d7d5", "d2d4", "e7e6"],
            ["d2d4", "e7e6", "e2e4", "d7d5"],
        ]),
        ("Piece development", [
            ["g1f3", "g8f6", "b1c3", "b8c6"],
            ["b1c3", "b8c6", "g1f3", "g8f6"],
        ]),
    ]

    print(f"\n  {'Transposition':<25}  {'cosine sim':>10}  {'max |Δg|':>9}  result")
    print(f"  {'-'*25}  {'-'*10}  {'-'*9}  ------")

    all_pass = True
    for name, move_seqs in transpositions:
        boards = []
        for moves in move_seqs:
            b = chess.Board()
            for m in moves:
                b.push_uci(m)
            boards.append(b)

        # Check all pairs reach the same FEN (otherwise it's not a transposition)
        fens = [b.fen().split(" ")[0] for b in boards]  # position only, ignore side/castling
        if len(set(fens)) > 1:
            print(f"  {name:<25}  {'N/A — not same pos':>21}  SKIP")
            continue

        geos = [geo(model, b) for b in boards]
        cos  = cosine_sim(geos[0], geos[1])
        max_delta = float(np.max(np.abs(geos[0] - geos[1])))
        ok = cos > 0.99
        status = PASS if ok else (WARN if cos > 0.95 else FAIL)
        if not ok:
            all_pass = False
        print(f"  {name:<25}  {cos:>+10.4f}  {max_delta:>9.4f}  {status}")

    return all_pass


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: dict):
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total  = len(results)
    passed = sum(results.values())
    print(f"\n  Passed: {passed}/{total}\n")
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class LegacyPetraNet(torch.nn.Module):
    """
    Pre-per-piece-geometry architecture (flat bottleneck, no STM-sign fix).
    Used to load checkpoints trained before commit f52b962 (e.g. zigzag r1/r2).
    Keys: bottleneck.1 = Linear(4096→128), value_head.0/2 = Linear, policy_head = Linear(128→4096).
    """
    def __init__(self, channels: int = 64, bottleneck_dim: int = 128):
        super().__init__()
        self.input_block = ConvBlock(14, channels)
        self.res_blocks  = torch.nn.Sequential(*[ResBlock(channels) for _ in range(4)])
        self.bottleneck  = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(channels * 8 * 8, bottleneck_dim),
            torch.nn.Tanh(),
        )
        self.value_head  = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )
        self.policy_head = torch.nn.Linear(bottleneck_dim, 64 * 64)

    def forward(self, x):
        x = self.input_block(x)
        x = self.res_blocks(x)
        g = self.bottleneck(x)
        return self.value_head(g), self.policy_head(g)

    def geometry(self, x):
        with torch.no_grad():
            x = self.input_block(x)
            x = self.res_blocks(x)
            return self.bottleneck(x)

    @torch.no_grad()
    def value(self, board: chess.Board, device: torch.device) -> float:
        self.eval()
        t = board_to_tensor(board).unsqueeze(0).float().to(device)
        v, _ = self.forward(t)
        return v.item()


def _load_model(path: str):
    """Load a model checkpoint, auto-detecting legacy vs current architecture."""
    sd = torch.load(path, map_location=device, weights_only=True)
    if "bottleneck.1.weight" in sd:
        model = LegacyPetraNet()
    else:
        model = PetraNet()
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model .pt")
    args = ap.parse_args()

    model = _load_model(args.model)
    model.eval()
    print(f"Loaded: {args.model}")

    print("\nBuilding win/loss centroids from KQ vs K anchor positions ...")
    c_win, c_loss = win_loss_centroids(model)
    axis_cos = cosine_sim(c_win, c_loss)
    print(f"  Win/loss centroid cosine similarity: {axis_cos:.4f}")
    print(f"  (Lower = better separated. R4 baseline: 0.869)")

    results = {}
    results["Material monotonicity"]     = test_material_monotonicity(model, c_win, c_loss)
    results["Piece value ordering"]      = test_piece_value_ordering(model, c_win, c_loss)
    results["Antipodal symmetry"]          = test_antipodal_symmetry(model)
    results["Forced mate convergence"]   = test_forced_mate_convergence(model, c_win, c_loss)
    results["Transposition consistency"] = test_transposition_consistency(model)

    print_summary(results)


if __name__ == "__main__":
    main()
