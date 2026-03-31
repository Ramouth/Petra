# Petra Phase 2 — Geometry as Driver

*Branch: phase2. Started: 2026-03-31.*

---

## Thesis

A chess engine that evaluates positions by their geometric relationship to winning and losing — not by material counting or value-head memorisation. The 128-dimensional bottleneck space is the engine. Every other component serves it.

The geometry must be trained from positions where the correct answer is unambiguous, then extended toward complexity. This is the endgame-first principle.

---

## Architecture

**PetraNet** (unchanged from Phase 1 end state):
- Input: (B, 14, 8, 8) board tensor
- CNN backbone: ConvBlock(14→64) + 4× ResBlock(64)
- Bottleneck: Linear(4096→128) + Tanh  ← the geometry space
- Value head: Linear(128→64) + Tanh + Linear(64→1) + Tanh
- Policy head: Linear(128→4096)

The bottleneck Tanh allows geometry vectors to occupy the full hypersphere. Win and loss positions can sit on opposite sides of the origin. This is the necessary (not sufficient) condition for antipodal geometry.

**Antipodal loss** (active throughout all stages):
```
L_antipodal = mean(max(0, cos(g, g_flip) + margin)) + norm_penalty
norm_penalty = mean(max(0, min_norm - ||g||)²)
```
`g_flip` is the geometry of the color-flipped board. This directly forces win/loss geometry apart. Without it, the value head can compensate and the geometry remains unstructured.

---

## Training loop

Each stage:
1. Generate positions for the current endgame class (random, no SF needed for Stages 1–3)
2. Per-epoch regeneration — fresh positions each epoch prevents memorisation
3. Train with `antipodal_weight=1.0`, `policy_weight=0.0`
4. Phase transition detection: if val loss drops >50% in one epoch, switch to tight patience (3 epochs)
5. Run `test_geometry.py` — advancement requires passing centroid cosine and color-flip thresholds
6. Gate: run step 6 eval (MCTS(geometry) vs MCTS(material)) to confirm geometry steers play

---

## Stages

### Stage 1 — KQ vs K ✅

Every non-stalemate position is a forced white win. The geometry *must* separate. No noise, no ambiguity.

**Result (2026-03-31):**
- Phase transition at epoch 8 — simultaneous collapse of value and antipodal loss
- Centroid cosine: **0.1753** (baseline was 0.981)
- Forced mate convergence: PASS
- Model: `models/geometry/stage1_best.pt`

---

### Stage 2 — KR vs K

Rook mates require king-to-edge and piece cooperation. Longer games than queen mates (up to 16 moves). Tests whether geometry encodes the concept of confinement, not just material advantage.

**Setup:**
- Init from Stage 1 weights (`models/geometry/stage1_best.pt`)
- 10k positions + 10k mirrors, regenerated each epoch
- Same antipodal loss, same hyperparameters
- Job: `jobs/stage2_geometry.sh`

**Advancement criteria:**
- Centroid cosine < 0.4
- Color flip cosine < -0.6
- `test_geometry.py` forced mate: PASS

---

### Stage 3 — K+P vs K

First draw region. Some K+P vs K positions are theoretical draws (opposition, stalemate traps). The geometry must represent three regions — win, draw, loss — not just two. Draw should sit between win and loss in the space, not coincide with either.

**New requirement:** geometry is no longer binary. The antipodal constraint stays but the margin needs tuning — draw positions should not be pushed to the losing side.

**Advancement criteria:**
- Draw positions cluster between win/loss centroids in PCA projection
- Selfplay outcome distribution matches known tablebase ratios (~60% white wins, ~40% draws)

---

### Stage 4 — KQ vs KR

First defensive piece. Black has a resource. ~15% of random positions are theoretical draws. Tests whether geometry encodes the defensive resource — a rook interposing changes the geometric reading of the position.

**New requirement:** SF depth 10 bootstrap needed (tactics matter — pure rule labels insufficient).

**Advancement criteria:**
- Selfplay draw rate converges toward ~15%
- White win rate > 70%

---

### Stages 5–7

| Stage | Class | New geometry requirement |
|-------|-------|--------------------------|
| 5 | Pawnless middlegame (2–4 pieces/side) | Piece interaction generalisation |
| 6 | Pawn endgames | Directional geometry, passed pawn distance |
| 7 | Full game | Geometry already anchored; selfplay on full positions |

Stage 7 is where the geometry transitions from endgame anchor to full-game driver. The model enters middlegame positions with correct endgame geometry already in place. The value head is no longer the only signal.

---

## Geometry MCTS

Once geometry is stable (target: after Stage 2 or Stage 3), MCTS gets a geometry bonus per move:

```
score(move) = value(board_after) + λ · (g_after · w_axis)
```

Where `w_axis = (c_win - c_loss) / ||c_win - c_loss||` is the win/loss axis in geometry space, updated after each training stage. λ starts at 0.1.

This is what step 6 eval tests on the Stage 1 model — does projecting onto the geometry axis produce better move selection than material counting?

---

## Gate metric

**Step 6: MCTS(geometry) vs MCTS(material)**

Run at the end of each stage. Both agents use the same model. One uses geometry value, one uses material count. If geometry can beat material, geometry is driving play.

Threshold: >52% to advance (50% = geometry and material are equivalent).

---

## GPU trigger

Three stages of confirmed geometry improvement (centroid cosine decreasing, color flip negative and increasing) → plan GPU migration for batched MCTS + larger model.

Stage 1: ✅. Stage 2 and Stage 3 to follow.

---

## Per-piece geometry (next architectural step)

Not yet implemented. Target: after Stage 3 confirms the current architecture can handle three-region geometry.

Each piece carries its own geometry vector. Board geometry = composition of piece geometries. Benefits:
- Piece value ordering emerges from training, not rules
- Geometry transition function `f(g_t, move) → g_{t+1}` becomes tractable: only the moved piece's vector updates
- Piece interference geometry (sliding pieces blocked by other pieces) is representable per-piece

The knight is explicitly excluded from interference geometry — it jumps, so blocking pieces do not constrain it. Its geometry is determined by square and board boundaries alone.

---

## Current state

| Item | Status |
|------|--------|
| Stage 1 KQK | ✅ Complete (centroid cosine 0.1753) |
| Per-piece geometry bottleneck | ✅ Implemented (849k params) |
| Mixed curriculum (KQK+KRK+KPK) | ✅ Ready (`jobs/stage2_geometry.sh`) |
| Step 6 eval | ⏳ Pending (rerun on HPC) |
| GPU plan | ⏳ After 3 stages confirmed |

---

## GPU scaling — collapse risks

More compute accelerates everything, including failure modes. Known risks and mitigations:

**Collapse modes:**

- **Antipodal trivial solution** — model satisfies `cos(g, g_flip) < 0` by driving geometry vectors toward zero. Symptom: antipodal loss hits zero in epoch 1 with no improvement in val loss. Norm penalty mitigates but may not be sufficient at large batch sizes.
- **Value head races ahead** — fast convergence on value loss before geometry has separated. The antipodal loss is the only geometry-independent signal; if it's weighted too low it loses the race.
- **Phase transition too fast** — on GPU, epochs take seconds. Tight patience of 3 epochs may not be enough time for geometry to stabilise after the saddle escape. Recalibrate tight patience for GPU epoch speed before long runs.
- **Selfplay position collapse** — at GPU scale, selfplay generates positions fast. Repetitive patterns can satisfy the loss without geometric variety. Per-epoch random regeneration partially protects; selfplay positions need diversity monitoring.

**Safeguards to build before GPU:**

- Track centroid cosine and color-flip cosine at every epoch — these are the real health metrics, not val loss. Geometry regression should trigger early stopping independently of loss improvement.
- Checkpoint every epoch regardless of best — the correct state may be 2 epochs before early stopping fires.
- Short calibration run first to validate tight patience at GPU epoch speed.
- The per-piece architecture is structurally more collapse-resistant than the flat bottleneck — no single shortcut produces a constant output — but this is not a guarantee.

**The honest position:** collapse dynamics at GPU scale are unknown until we run it. The goal is continuous geometry monitoring so collapse is visible before the run is lost.
