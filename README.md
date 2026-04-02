# Petra

A chess engine whose positional evaluation emerges from learned geometry, not material counting or hand-crafted heuristics. The 128-dimensional bottleneck space is the engine. Every other component serves it.

**Long-term thesis:** Winning positions and losing positions should occupy opposite sides of the geometry space. MCTS selects moves that push geometry toward the winning region. After training converges, the engine should self-adjust from game outcomes alone — no external teacher.

**Target:** ~2000 ELO vs Stockfish without hand-crafted heuristics.

---

## Version History

### Version 1 — Maia-based geometry encoder (failed)
*2026-03-20 to 2026-03-24*

Built a geometry encoder (walk profile → 128-dim → value) on top of Maia-1500 policy. The geometry was supposed to drive evaluation; Maia's learned representations took over instead. Geometry became a passenger.

Root causes:
- Contrastive loss collapses on real chess data — most middlegame positions have near-identical values, so the loss provides no gradient
- A sign bug in training data (Black-to-move positions had wrong sign) made v6 appear to work — it was exploiting the mislabeling, not learning geometry
- End state: Petra ≈ Maia-1500 greedy. Geometry added nothing.

### Version 2 — PetraNet zigzag + antipodal curriculum
*2026-03-24 to 2026-03-31*

New architecture: **PetraNet** — CNN on raw board tensor (14×8×8), joint value + policy heads, 128-dim Tanh bottleneck.

**Zigzag training (Rounds R1–R6):**
Self-play generates positions the model actually reaches; Stockfish re-labels them at increasing depth. Each round: selfplay → reeval → train → gate.

| Round | n_sim | SF depth | Positions | Gate vs material | Result |
|-------|-------|----------|-----------|-----------------|--------|
| R1 | 40 | 12 | ~5k | — | baseline |
| R2 | 80 | 15 | ~5k | — | — |
| R3 | — | — | — | — | — |
| R4 | — | 20 | ~2.3k | **67.0%** (+123 ELO) | **best zigzag model** |
| R5 | — | — | ~5.2k | 37.0% vs R4 | FAIL — geometry regressed |
| R6 | — | 15 | ~5.8k | — | cold start (Tanh bottleneck fix) |

R4 is the peak zigzag model. ELO gains R1→R4 came from deeper MCTS search, not geometry improvement. The bottleneck had ReLU — all activations ≥ 0, antipodal structure geometrically impossible.

**Geometry curriculum (Stages 1–4):**
Switched to endgame-first approach. Antipodal loss (`L = max(0, cos(g, g_flip) + margin)`) explicitly forces win/loss geometry apart.

| Stage | Config | Centroid cos | Gate | Result |
|-------|--------|-------------|------|--------|
| Stage 1 | KQK, antipodal=1.0 | — | not gated | baseline |
| Stage 2 | stages 1–8, antipodal=1.0, no STM fix | -0.22 | 39.5% (-74 ELO) | FAIL — STM bug made antipodal unsatisfiable |
| Stage 3 | stages 1–8, antipodal=1.0, **STM-sign fix** | 0.03 | **54.0% (+28 ELO)** | PASS — but value head compensated for unstructured geometry ("passenger problem") |
| Stage 4 | thin value head Linear(128→1) | -0.47 | timed out | antipodal loss = 0.0000 from epoch 1; geometry 1/5 tests |

Key discovery: the STM-sign fix (multiply geometry by `stm_sign = board[:, 12, 0, 0] * 2 - 1`) is the necessary architectural condition for antipodal geometry to be learnable.

### Version 3 — STM-fix complete redo (current)
*2026-04-02 onwards. Branch: phase2.*

Complete restart with STM-sign fix as the foundation. Stage 3 geometry model serves as R0 bootstrap for self-play. Self-play curriculum with Stockfish re-labeling, geometry probe at every round.

See [LOGBOOK.md](LOGBOOK.md) for round-by-round results.

---

## Architecture

**PetraNet** (849k parameters, Stage 3 / 841k parameters, thin value head):
```
Input:        (B, 14, 8, 8) board tensor
CNN:          ConvBlock(14→64) + 4× ResBlock(64)
Bottleneck:   Linear(4096→128) + Tanh        ← geometry space
Value head:   Linear(128→64) + Tanh + Linear(64→1) + Tanh
Policy head:  Linear(128→4096)
```

**STM-sign fix** (commit f52b962):
```python
stm_sign = (board[:, 12, 0, 0] * 2 - 1)   # +1 white to move, -1 black
geometry = stm_sign * _piece_geometry(board)
```
Makes geometry STM-relative by construction. Antipodal constraint (`g(pos) ≈ -g(flip(pos))`) becomes achievable.

---

## Geometry Tests

`src/test_geometry.py` — run after every training round:

| Test | What it measures | Target |
|------|-----------------|--------|
| 1 — Material monotonicity | Adding pieces increases win projection | PASS |
| 2 — Piece value ordering | Q > R > B ≈ N > P in win projection | PASS |
| 3 — STM symmetry | Win projection positive for both colors when STM player is winning | PASS |
| 4 — Forced mate convergence | Win projection increases as mate approaches | PASS |
| 5 — Transposition consistency | Same position via different move orders = same geometry | PASS (always) |

Win/loss centroid cosine similarity: lower is better (more separated). Baseline R4: 0.869.

---

## Gate Threshold

A round passes the gate when win rate > 55% against the baseline (either material heuristic or previous round's model). Evaluated over 200 games at n_sim=50–100.

---

## Files

| File | Purpose |
|------|---------|
| `LOGBOOK.md` | Per-round results — the primary record |
| `CURRICULUM.md` | Version 2 geometry curriculum design (historical) |
| `ZIGZAG.md` | Version 2 zigzag training plan (historical) |
| `PLAN.md` / `ROADMAP.md` | Version 2 phase 2 design notes (historical) |
| `HPC_ACCESS.md` | DTU HPC connection and job submission |
| `jobs/` | LSF job scripts |
| `logs/` | HPC output logs |
| `src/` | Training, evaluation, and geometry code |
| `models/` | Saved checkpoints |
