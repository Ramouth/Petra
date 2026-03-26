# Petra — Phase 1 Roadmap

## Context

Phase 0 (2026-03-20 to 2026-03-24) failed. The geometry encoder (walk profile → contrastive loss → 128-dim space → value) cannot be trained on correct chess data — contrastive loss collapses because most middlegame positions have near-identical values. The v6 encoder that appeared to work was exploiting a sign-error bug in training data, not learning chess geometry.

Phase 0 end state: Petra = Maia-1500 policy + near-constant value function. Strictly worse than Maia-1500 greedy alone.

**Phase 1 thesis:** Replace the geometry encoder with a direct supervised value+policy network (PetraNet) trained on Lichess game outcomes. Validate with ELO before anything else. Build on what works.

---

## Phase 1 Milestones

### Milestone 1 — PetraNet architecture + MCTS ✅
CNN on raw board tensor (14×8×8). Joint value head (MSE on game outcome) and policy head (cross-entropy on move played). MCTS wired to PetraNet value + policy.

### Milestone 2 — Data pipeline with integrity validation ✅
Lichess PGN parser with 6 integrity checks (label values, sign correctness, label distribution, side-to-move balance, king presence, within-game sign consistency). Game-level train/val split (no leakage). Source: `lichess_db_standard_rated_2025-01.pgn.zst`.

### Milestone 3 — Supervised training loop ✅
Joint value + policy training. Early stopping on val loss (patience=5). Best checkpoint saved separately. Post-training sign sanity checks.

### Milestone 4 — Ablation game runner + evaluation agents ✅
Game runner for head-to-head evaluation. Evaluation agents for ELO testing.

---

## Session 1 Results (2026-03-25)

### Run 1 — Lichess game-outcome labels (150k games)
- **Dataset:** 1.19M positions, 150k games, min Elo 1500. All 6 validation checks passed.
- **Training:** 15 epochs, ~70 min/epoch on CPU. Best val loss: 3.3008.
- **Policy:** Top-1 35.8%, Top-5 70.4% — strong.
- **Value:** R²=0.181 — weak. Game outcome is too noisy a label.
- **Ablation:** MCTS(learned) lost 0-100 to MCTS(material). Gate FAILED.
- **Geometry probe:** 128-dim space healthy (top-1 eigenvalue 12.7%, not collapsed). But win/loss centroids nearly identical (cosine sim 0.9951) — space organised by board structure, not value.

### Conclusion
Policy head is a real asset. Value head needs a stronger training signal. Game outcome labels are insufficient — positions early in a game have weak correlation with the eventual winner.

---

## Session 1 Continued — SF Model (2026-03-25)

### Step 5 — Stockfish re-evaluation ✅
Re-evaluated 200k positions from dataset.pt with Stockfish depth 10. Labels replaced with `tanh(cp/400)`.

- **Label std: 0.565**, 43% decisive, 20% equal
- Saved → `dataset_sf.pt` (190k train / 10k val)

### Step 6 — Retrain on SF labels ✅
- **Best val loss: 4.2461**, R²=0.483, Top-1 21.2%, Top-5 45.0%
- All sanity checks passed. Start position value=-0.138 (slight negative bias, noted)
- Saved → `models/sf/best.pt`

### Step 7 — Ablation on SF model ✅
- Step 2 Greedy: 74% (+182 ELO)
- Step 3 MCTS(zero): 61.5% (+81 ELO)
- Step 4 MCTS(material): 89% (+363 ELO)
- Step 5 MCTS(learned) at n_sim=20: **54%** — gate FAILED (threshold 55%)
- Step 5 repeated at n_sim=100: **54.5%** — consistent positive trend, still below threshold

### Step 8 — Geometry probe on SF model ✅
- Win/loss centroid cosine: **0.9192** (vs 0.9951 for Lichess model — significant improvement)
- Top-1 eigenvalue: 23% (healthy spread)
- NN label consistency: 0.901
- Rated WEAK but meaningful separation emerging

### Decision: proceed to zigzag ✅
54-54.5% is a consistent positive signal. More supervised training on Lichess positions has diminishing returns — the model needs to see positions from its own play. Zigzag chosen.

---

## Session 2 — Zigzag Design + HPC Setup (2026-03-25)

### Fixes
- **evaluate.py**: added `--temp-moves` (default 10) — MCTSAgent was using temperature=0 making all 100 games identical. Fixed by using temperature=1 for first N half-moves.
- **data.py**: uint8 tensor storage, zst support, numpy pre-allocation
- **train.py**: `.float()` conversion for uint8 tensors

### ZIGZAG.md written ✅
Full design document: 4-round curriculum, data strategy, policy loss change, temperature schedule, LR schedule, failure modes, HPC plan, open questions.

### DTU HPC (gbar) setup ✅
- CPU-only access confirmed (no GPU for now)
- `petra-env` venv created, numpy/chess/torch installed
- Stockfish binary installed at `~/bin/stockfish`
- Project cloned to `~/Petra-Phase1`

---

## Session 3 — Zigzag Implementation (2026-03-26)

### Built ✅
- **selfplay.py**: self-play game generation. MCTS with temperature schedule, Dirichlet noise, resign threshold, position sampling. `--workers N` for HPC parallelism.
- **zigzag.py**: orchestration of the full loop (selfplay → reeval → train → gate).
- **train.py**: dense policy loss (KL over visit distributions) for self-play datasets; `--init-model` for fine-tuning from prior round.

### Round 1 prototype (50 games, n_sim=40, SF depth 12) — in progress
- Self-play: 50 games, 600 positions, 22 min ✅
- SF re-label: 600 positions at depth 12, 20s. 79% decisive ✅
- Train (fine-tune from models/sf/best.pt): val R²=0.781, all sanity checks passed ✅
- Gate evaluation (100 games, n_sim=40): **in progress** — 10/100 at 55%

---

## Next

1. Gate result for round 1 → if pass, run full 500-game round 1 on HPC with workers
2. Set up LSF job scripts for HPC submission
3. Push latest commits (offline during session 3)

---

## Milestones

### ELO
- [ ] Beat MCTS(material) at >55% — *round 1 gate* ← here now
- [ ] Beat MCTS(material) at >60% — *consistent edge*
- [ ] Beat MCTS(material) at >70% — *dominant*
- [ ] Beat Stockfish depth 1
- [ ] Beat Stockfish depth 5

### Geometry (the thesis)
- [ ] Win/loss centroid cosine < 0.85 — *currently 0.919*
- [ ] Win/loss centroid cosine < 0.70 — *strong separation*
- [ ] Separation gap > 0.15 — *currently 0.04*

### Self-play
- [ ] 1k self-play positions trained — *prototype round 1 ← here now*
- [ ] 10k self-play positions — *full HPC round 1*
- [ ] 100k self-play positions — *rounds 2–3*
- [ ] 1M MCTS simulations — *round 1 full run*
- [ ] 10M MCTS simulations — *rounds 2–4*

---

## Long-Term Phases

| Phase | Goal |
|-------|------|
| 0 | Geometry encoder on Stockfish values — **FAILED** |
| 1 | Supervised pretraining on Lichess outcomes (now) |
| 2 | Online geometry update from game outcomes |
| 3 | Self-play loop — geometry evolves from game experience |
| 4 | Full autonomy — Petra selects own curriculum |

**Success criterion:** Petra reaches ~2000 ELO vs Stockfish without hand-crafted heuristics.
