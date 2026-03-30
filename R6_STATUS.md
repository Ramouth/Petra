# R6 Status Note — 2026-03-30
*Temporary. To be merged into ROADMAP.md once gate and geometry probe results are in.*

---

## The Geometry Hypothesis

Phase 1's core thesis is that a chess model's internal representation — the 128-dimensional bottleneck — should encode position value geometrically. Winning positions and losing positions should occupy opposite regions of that space. The win/loss axis should be usable: MCTS can then treat a move not just as "good value" but as "moves toward the winning region."

The geometry probe measures this with three signals:
- **Centroid cosine similarity** — how far apart win/loss centroids are (lower = better separated)
- **Separation gap** — difference in within-class vs between-class distance
- **Dead dimensions** — how many of the 128 dimensions are permanently zero

R4 state: centroid cosine 0.869, separation gap 0.048, 26/128 dead. The encoder is frozen across rounds — ELO gains from R1→R4 came entirely from deeper MCTS search, not geometry improvement.

---

## Why the Geometry Stagnated: ReLU

The bottleneck was `Linear(4096→128) + ReLU`. ReLU forces all activations ≥ 0. The consequence is geometric: win and loss positions are both forced into the positive orthant — they cannot be antipodal. With 26 permanently dead dimensions and only ~27 doing meaningful work, the value head memorised SF outputs on top of a crippled representation.

The geometry hypothesis was never testable under this architecture.

---

## The Tanh Fix

Replace `ReLU` with `Tanh` in the bottleneck. Tanh is symmetric around zero — activations live in (-1, +1), negative values are allowed. Win and loss positions can now occupy opposite sides of the origin. All 128 dimensions become usable.

This is a targeted, theoretically grounded fix with one line of code. The geometry change forces the model to rebuild its representation from scratch — hence the R6 cold start, no init from R4.

---

## What Actually Happened in R6

**Complications on the way in:**
- Reeval hit the wall time twice (3h limit, ~5,782 positions at 2.8s/pos = 4.5h needed). Third run with 5h wall succeeded.
- Stale job dependency chains caused multiple resubmissions. Lost most of the day to HPC scheduling issues.

**Dataset smaller than expected:**
- selfplay.py caps at 12 positions/game → 500 games × 12 = 5,782 positions total.
- ZIGZAG.md assumed ~30k/round. The cap was carried over from Lichess pretraining convention and never revisited.
- R6 trained on 5,782 positions. This is very small for a cold start.

**Bootstrap result:**
- Early stopped at epoch 12, best at epoch 7. Classic overfitting — training loss kept falling, val loss diverged after epoch 7.
- Label distribution skewed: 85% decisive (vs 43% in original Lichess bootstrap). Late-game R5 positions dominate.

**Sanity checks — one failure:**
```
✓ White up queen          +0.932
✗ Black up queen          +0.667  ← wrong sign
✓ KQ vs K, White to move  +0.850
✓ KQ vs K, Black to move  -0.478
```

The Black up queen failure is a value head problem, not a bottleneck problem. The value head architecture is `Linear → ReLU → Linear → Tanh`. ReLU breaks antisymmetry at the head level — if the Tanh bottleneck correctly flips sign for color-symmetric positions, the ReLU in the value head kills the negative component. R4 avoided this because its weights were fine-tuned from R2, giving the value head a learned prior that overcame the ReLU asymmetry. R6 cold-start has not learned that compensation yet.

The fix: replace `ReLU` with `Tanh` in the value head too. Deferred until the gate and geometry probe confirm the bottleneck change is producing useful signal.

---

## What We Are Waiting For

**Gate (R6 vs R4, 100 games, due ~Tue Mar 31 04:00–08:00):**
The comparison is structurally unfair to R6 — R4 is a fine-tuned model with accumulated knowledge, R6 is a cold start on 5,782 positions. Any result above 45% is not disqualifying. A result above 52% on a cold start would be a strong signal.

**Geometry probe (run after gate):**
This is the decisive test. The geometry probe is independent of the value head — it measures the bottleneck directly. If the Tanh bottleneck is working:
- Centroid cosine < 0.85 (down from R4: 0.869)
- Separation gap > 0.048 (up from R4, target: exceed R2 peak of 0.057)
- Dead dimensions < 26/128

---

## Decision Tree

```
R6 geometry probe:
│
├── Geometry improving (cosine ↓, gap ↑, dead dims ↓)
│   → Tanh bottleneck confirmed working
│   → R7: fix value head (ReLU→Tanh), 50 pos/game, depth 15 reeval
│   → Gate R7 vs R4. If R7 > R4: geometry hypothesis alive, proceed.
│
└── Geometry flat or worse (same as R4 despite Tanh change)
    → Bottleneck architecture is not the constraint
    → CNN backbone may not encode material asymmetry in geometry-compatible way
    → Commit to R4 as current best
    → Reconsider more fundamentally:
        - Contrastive loss on bottleneck
        - Larger bottleneck (256/512 dim)
        - Auxiliary geometry loss during training
        - Abandon geometry hypothesis, pursue ELO directly
```

---

## Summary

R6 is a noisy signal. The complications — wall time failures, tiny dataset, value head bug — mean the gate result will be hard to interpret cleanly. The geometry probe is what matters. If the Tanh bottleneck has moved the geometry even slightly, the path forward is clear and well-defined (R7 with known fixes). If not, Phase 1 may need a harder rethink.

R4 remains the best confirmed model. Nothing from R6 changes that yet.
