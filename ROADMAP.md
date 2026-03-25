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

## Current Work

### Step 5 — Stockfish re-evaluation ✅
Re-evaluated 200k positions from dataset.pt with Stockfish depth 10. Labels replaced with `tanh(cp/400)` — direct position evaluations instead of game outcomes.

- **Label std: 0.565** (vs ~0.5 for game outcomes — wider, richer signal)
- **43% decisive** (|v|>0.5), **20% equal** (|v|<0.1)
- Saved → `dataset_sf.pt` (190k train / 10k val)

### Step 6 — Retrain on SF labels [ ]
Training in progress on `dataset_sf.pt`. ~10 min/epoch. Expected to finish overnight.

```bash
python3 src/train.py --dataset dataset_sf.pt --out models/sf/ 2>&1 | tee train_sf_run.log
```

### Step 7 — Ablation on SF model [ ]
Re-run ablation ladder. Critical gate: does MCTS(learned) beat MCTS(material)?

```bash
python3 src/evaluate.py --model models/sf/best.pt --games 100 --all-steps --n-sim 20
```

### Step 8 — Geometry probe on SF model [ ]
Re-run probe_geometry.py. Expect win/loss centroid separation to improve significantly with SF labels.

```bash
python3 src/probe_geometry.py --model models/sf/best.pt --dataset dataset_sf.pt
```

---

## Next (HPC — tomorrow)

If SF model passes the ablation gate:
1. Re-evaluate full 1.19M positions with Stockfish depth 12-15 on HPC
2. Full training run on HPC
3. Full ablation + geometry probe

SSH setup: key generated (`~/.ssh/gbar`), needs copying to DTU HPC (requires campus/VPN).

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
