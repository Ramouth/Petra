"""
zigzag.py — Orchestrates the self-play → SF re-label → train loop.

Each round:
  1. SELF-PLAY   selfplay.py   → data/selfplay_r{n}.pt
  2. SF RELABEL  reeval_stockfish.py  → data/selfplay_r{n}_sf.pt
  3. TRAIN       train.py      → models/zigzag/r{n}/best.pt
  4. EVALUATE    evaluate.py   → gate check (>55% vs MCTS(material))

Round parameters (from ZIGZAG.md):
  Round | n_sim | SF depth | n_games | lr
    1   |   40  |    12    |   500   | 5e-4
    2   |   80  |    15    |   500   | 3e-4
    3   |  160  |    18    |   500   | 1e-4
    4   |  320  |    20    |   500   | 5e-5

CPU-only note: n_sim=320 is expensive on CPU. Cap at round 3 unless
GPU access is available. Use --rounds 1-3 for HPC CPU runs.

Usage
-----
    # Full run from round 1 (uses models/sf/best.pt as round-0 seed)
    python3 zigzag.py

    # Prototype: 50 games, start from round 1
    python3 zigzag.py --games 50 --workers 4 --rounds 1

    # Resume from a specific round
    python3 zigzag.py --start-round 2

    # Dry-run: print commands without executing
    python3 zigzag.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Round curriculum  (n_sim, sf_depth, n_games, lr)
# ---------------------------------------------------------------------------

ROUNDS = [
    (40,   12, 500, 5e-4),   # round 1
    (80,   15, 500, 3e-4),   # round 2
    (160,  18, 500, 1e-4),   # round 3
    (320,  20, 500, 5e-5),   # round 4
]

GATE_WINRATE  = 0.55   # new model must beat MCTS(material) at this rate
GATE_N_GAMES  = 100    # games for the gate evaluation
GATE_N_SIM    = 20     # n_sim for gate evaluation (fixed — measures value head, not search depth)

SRC_DIR       = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(SRC_DIR)
DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models", "zigzag")
SEED_MODEL    = os.path.join(PROJECT_ROOT, "models", "sf", "best.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], dry_run: bool) -> int:
    """Print and optionally execute a command. Returns exit code."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0
    t0 = time.time()
    result = subprocess.run(cmd)
    print(f"  (exit {result.returncode}, {time.time()-t0:.0f}s)", flush=True)
    return result.returncode


def _model_for_round(r: int) -> str:
    """Path to the best model from round r (0 = supervised seed)."""
    if r == 0:
        return SEED_MODEL
    return os.path.join(MODELS_DIR, f"r{r}", "best.pt")


def _check_gate(log_path: str) -> bool:
    """
    Parse evaluate.py output to check if the gate was passed.
    Looks for 'wr=X.XXX' on the Step 5 line and checks against threshold.
    """
    if not os.path.exists(log_path):
        print(f"  Gate log not found: {log_path}")
        return False
    with open(log_path) as f:
        for line in f:
            if "Learned value (gate)" in line and "wr=" in line:
                try:
                    wr_str = line.split("wr=")[1].split()[0]
                    wr = float(wr_str)
                    passed = wr >= GATE_WINRATE
                    print(f"  Gate result: wr={wr:.3f}  "
                          f"({'PASS' if passed else 'FAIL'}, threshold={GATE_WINRATE})")
                    return passed
                except (IndexError, ValueError):
                    pass
    print("  Could not parse gate result from log.")
    return False


# ---------------------------------------------------------------------------
# Round execution
# ---------------------------------------------------------------------------

def run_round(r: int, n_sim: int, sf_depth: int, n_games: int, lr: float,
              workers: int, dry_run: bool) -> bool:
    """
    Execute one full zigzag round. Returns True if gate passed.
    """
    print(f"\n{'='*60}")
    print(f"  ROUND {r}  |  n_sim={n_sim}  sf_depth={sf_depth}  "
          f"n_games={n_games}  lr={lr:.0e}")
    print(f"{'='*60}")

    prev_model  = _model_for_round(r - 1)
    selfplay_pt = os.path.join(DATA_DIR, f"selfplay_r{r}.pt")
    sf_pt       = os.path.join(DATA_DIR, f"selfplay_r{r}_sf.pt")
    out_dir     = os.path.join(MODELS_DIR, f"r{r}")
    gate_log    = os.path.join(PROJECT_ROOT, "logs", f"zigzag_r{r}_gate.log")

    os.makedirs(DATA_DIR,  exist_ok=True)
    os.makedirs(out_dir,   exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

    # --- 1. Self-play ---
    print(f"\n--- Step 1: Self-play ({n_games} games, n_sim={n_sim}) ---")
    rc = _run([
        sys.executable, os.path.join(SRC_DIR, "selfplay.py"),
        "--model",   prev_model,
        "--games",   str(n_games),
        "--n-sim",   str(n_sim),
        "--out",     selfplay_pt,
        "--workers", str(workers),
    ], dry_run)
    if rc != 0:
        print(f"  Self-play failed (exit {rc}). Aborting round.")
        return False

    # --- 2. SF re-label ---
    print(f"\n--- Step 2: Stockfish re-label (depth {sf_depth}) ---")
    rc = _run([
        sys.executable, os.path.join(SRC_DIR, "reeval_stockfish.py"),
        "--dataset", selfplay_pt,
        "--out",     sf_pt,
        "--depth",   str(sf_depth),
    ], dry_run)
    if rc != 0:
        print(f"  SF re-label failed (exit {rc}). Aborting round.")
        return False

    # --- 3. Train ---
    print(f"\n--- Step 3: Train (lr={lr:.0e}) ---")
    rc = _run([
        sys.executable, os.path.join(SRC_DIR, "train.py"),
        "--dataset",  sf_pt,
        "--out",      out_dir,
        "--lr",       str(lr),
        "--epochs",   "15",
        "--patience", "3",
    ], dry_run)
    if rc != 0:
        print(f"  Training failed (exit {rc}). Aborting round.")
        return False

    # --- 4. Gate evaluation ---
    print(f"\n--- Step 4: Gate evaluation ({GATE_N_GAMES} games, n_sim={GATE_N_SIM}) ---")
    new_model = _model_for_round(r)
    rc = _run([
        sys.executable, os.path.join(SRC_DIR, "evaluate.py"),
        "--model",      new_model,
        "--step",       "5",
        "--games",      str(GATE_N_GAMES),
        "--n-sim",      str(GATE_N_SIM),
        "--temp-moves", "10",
    ], dry_run)
    # evaluate.py writes its own output to stdout — capture to log as well
    # (for now just check the return code; a non-zero means GATE FAILED)
    if dry_run:
        print(f"  [dry-run] Gate assumed passed.")
        return True

    passed = rc == 0
    print(f"\n  Round {r} gate: {'PASSED' if passed else 'FAILED'}")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds",      type=int, default=None,
                    help="How many rounds to run (default: all 4)")
    ap.add_argument("--start-round", type=int, default=1,
                    help="Resume from this round (default: 1)")
    ap.add_argument("--games",       type=int, default=None,
                    help="Override n_games for all rounds (prototype: 50)")
    ap.add_argument("--workers",     type=int, default=1,
                    help="Parallel self-play workers")
    ap.add_argument("--dry-run",     action="store_true",
                    help="Print commands without executing")
    args = ap.parse_args()

    start = args.start_round
    end   = start + (args.rounds or len(ROUNDS)) - 1
    end   = min(end, len(ROUNDS))

    print(f"zigzag.py  |  rounds {start}–{end}  |  workers={args.workers}"
          + ("  [DRY RUN]" if args.dry_run else ""))

    for r in range(start, end + 1):
        n_sim, sf_depth, n_games, lr = ROUNDS[r - 1]
        if args.games:
            n_games = args.games

        prev_model = _model_for_round(r - 1)
        if not args.dry_run and not os.path.exists(prev_model):
            print(f"\nERROR: seed model not found: {prev_model}")
            print("  Round 0 requires models/sf/best.pt (run supervised training first).")
            sys.exit(1)

        passed = run_round(
            r=r, n_sim=n_sim, sf_depth=sf_depth,
            n_games=n_games, lr=lr,
            workers=args.workers, dry_run=args.dry_run,
        )

        if not passed:
            print(f"\nRound {r} gate FAILED. Stopping zigzag.")
            print("Investigate: check geometry probe, val R², and self-play game quality.")
            sys.exit(1)

        print(f"\nRound {r} complete. Model saved → {_model_for_round(r)}")

    print(f"\nZigzag complete. Ran rounds {start}–{end}.")


if __name__ == "__main__":
    main()
