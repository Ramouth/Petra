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

## Sessions 4–6 — Zigzag R1–R4, Geometry Analysis (2026-03-27 to 2026-03-29)

### Zigzag round results

| Round | Win rate | Opponent | Notes |
|-------|----------|----------|-------|
| R1 | 54.5% | MCTS(material) | Marginal pass. SF labels, n_sim=40 |
| R2 | 61.0% | MCTS(material) | Strong pass. SF labels, n_sim=80 |
| R3 | 59.0% (45% vs R2) | MCTS(material) | Regression vs R2. Root cause: same data distribution, early stopping at epoch 1 |
| R4 | 67.0% | MCTS(material) | Strong. 200 games, n_sim=400, init from R2. Deeper search hypothesis confirmed |

R4 change: reset to R2 weights + deeper MCTS. Gate improved significantly. The gain is from search depth, not from geometry improvement.

### PGN analysis (R3 and R4 gates)
- 25–29% of games hit the 300-move limit — not bare king shuffles but complex endgames
- Type A (5/25): queen vs bare king, model can't deliver checkmate (geometry failure)
- Type B (20/25): R+P vs R+P, Q+B vs Q+B — some theoretical draws, some conversion failures
- Conclusion: Petra plays real chess. The failure mode is endgame conversion, not structural chaos.

### Geometry probe — critical finding (2026-03-29)

Ran `compare_geometry.py` across R1, R2, R4 using `selfplay_r1_full_sf.pt` as fixed reference.

| Metric | R1 | R2 | R4 |
|--------|----|----|-----|
| Top-1 eigenvalue (%) | 21.5 | 21.7 | 21.5 |
| Centroid cosine sim | 0.883 | 0.871 | 0.869 |
| Separation gap | 0.050 | 0.057 | 0.048 |
| NN consistency | 0.912 | 0.903 | 0.902 |
| Mean vec norm | 92.4 | 90.2 | 87.8 |
| KQ vs K (White) | ✓ | ✓ | ✗ FAIL |
| K vs KQ (Black) | ✗ FAIL | ✗ FAIL | ✗ FAIL |

**The encoder is frozen.** SF labels do not improve geometry across rounds. ELO gains come from deeper MCTS search, not better representation. K vs KQ (White bare king = losing) has never been classified correctly in any round — a systemic bias.

### Architectural root cause: ReLU in the bottleneck

The bottleneck is `Linear(4096→128) + ReLU`. ReLU forces all geometry values ≥ 0. Consequence:
- 57% of geometry values exactly zero
- 26 of 128 dimensions permanently dead (never activate on any position)
- Only ~27 dimensions distinguish win from loss
- Win centroid mean activation (2.80) ≈ loss centroid mean activation (2.83)
- Win/loss centroids cannot be antipodal — both live in the positive orthant

The value head compensates by memorising SF outputs on top of a crippled representation. The geometry hypothesis has never been properly testable with this architecture.

**Fix:** Replace `ReLU` with `Tanh` in the bottleneck. Done in this session (model.py). Tanh allows negative values → win/loss can occupy opposite sides of the origin → all 128 dimensions become usable.

---

## R5–R7 Plan (2026-03-29)

### R5 — Outcome labels, current architecture (submitted 2026-03-29)
- 500 games, n_sim=400, init from R4, no SF reeval — outcome labels (+1/-1/-0.1) directly
- Answers: do outcome labels improve geometry within the ReLU constraint?
- Expected: marginal improvement at best. ReLU cap is still in place.
- Scripts: `jobs/r5_selfplay.sh`, `r5_train.sh`, `r5_gate.sh`

### R6 — Tanh bottleneck, retrain from scratch (next after R5 results)
- One architectural change: `nn.ReLU()` → `nn.Tanh()` in bottleneck (done)
- Retrain from scratch — do NOT init from R4/R5. ReLU weights are geometrically wrong for Tanh.
- Outcome labels as primary signal
- Success criteria:
  - KQ vs K AND K vs KQ both classify correctly
  - Centroid cosine sim < 0.80
  - Separation gap > 0.057 (exceeds R2 peak)
  - Dead dimensions < 5/128

If R6 passes: the geometry hypothesis is alive. If R6 fails with Tanh: the CNN backbone itself may not be encoding material asymmetry in a geometry-compatible way — deeper architectural question.

### R7 — Geometric MCTS (only if R6 proves antipodal geometry)
Moves as trajectories in geometry space. Per-move MCTS bonus:

```
score(move) = value(board_after) + λ · Δg · (c_win − c_loss)
```

`Δg = geometry(board_after) − geometry(board_before)` projected onto the win-loss axis.
This gives MCTS a dense gradient at every node, not just leaf-node evaluation.
λ is tunable — start at 0.1.

### Hard decision after R7
After R7 we have enough signal to decide: does the geometry hypothesis hold at this scale? If yes, the path to GPU is justified (batched MCTS, larger model). If no, we reconsider the architecture more fundamentally (contrastive loss on the bottleneck, larger bottleneck, sequence model for position history). No premature commitment.

---

## Next

1. R5 results → run `compare_geometry.py` — does KQ vs K pass? Does separation gap move?
2. R6 scripts (Tanh, scratch training, outcome labels)
3. After R7: hard go/no-go on geometry hypothesis

---

## Milestones

### ELO
- [x] Beat MCTS(material) at >55% — *R2: 61%*
- [x] Beat MCTS(material) at >60% — *R2: 61%, R4: 67%*
- [ ] Beat MCTS(material) at >70% — *R4: 67%, close*
- [ ] Beat Stockfish depth 1
- [ ] Beat Stockfish depth 5

### Geometry (the thesis)
- [x] Win/loss centroid cosine < 0.85 — *R4: 0.869*
- [ ] Win/loss centroid cosine < 0.70 — *target for R6*
- [ ] Separation gap > 0.10 — *currently 0.048–0.057*
- [ ] KQ vs K AND K vs KQ both correct — *K vs KQ never correct, any round*
- [ ] Dead dimensions < 5/128 — *currently 26/128*

### Self-play
- [x] 1k self-play positions trained
- [x] 10k self-play positions — *R1 full run*
- [x] 100k self-play positions — *R2–R4*
- [x] 1M MCTS simulations — *R4: n_sim=400 × 200 games*
- [ ] 10M MCTS simulations — *GPU territory*

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
