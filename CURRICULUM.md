# Geometry Curriculum — Endgame First

*Branch: geometry-first. Last updated: 2026-03-31.*

---

## Principle

Train geometry from the positions where the signal is unambiguous, using selfplay as the primary data source. Stockfish is used only to bootstrap the first stage — after that, selfplay generates its own labels because the winning side knows it is winning by the rules of chess, not by SF's evaluation.

The antipodal constraint is trained explicitly at every stage:

```
L_total = L_value + λ · L_antipodal
L_antipodal = max(0, cos(g, g_flipped) + margin)
```

Where `g_flipped` is the geometry of the color-flipped position. This forces win and loss regions apart directly. λ starts at 0.1 and increases as geometry stabilises.

---

## Stage 1 — KQ vs K

**Why first:** ~150k legal positions. Every non-stalemate position is a forced white win. The geometry *must* separate — there is no other interpretation. Perfect antipodal anchor.

**Position generation:**
- Enumerate all legal KQ vs K positions (white king + queen on any square, black king on any square, no overlap, not in illegal check)
- Filter: remove positions where black is already in checkmate or stalemate at generation time
- ~120k positions survive

**Selfplay loop:**
- White: geometry MCTS (maximise win projection onto current win/loss axis)
- Black: random legal moves
- Starting positions: random sample from enumerated KQ vs K positions
- Games are short (KQ mates in ≤10 moves from most positions with correct play)
- Label: terminal outcome (+1 white win, -1 stalemate/draw) — no SF needed

**Bootstrap (Stage 1 only):**
- Before selfplay has any geometry signal, bootstrap with SF depth 5 on a sample of 10k positions
- SF depth 5 is sufficient here — KQ vs K has no tactics to miss
- This gives the geometry axis its initial orientation so selfplay MCTS has something to follow

**Antipodal pairs:**
- Every KQ vs K position has a natural antipodal partner: swap colors (KQ belongs to Black, white has bare king)
- Sample these in paired batches during training
- L_antipodal is computed on these pairs, not random flips

**Advancement criteria (all must pass):**
- Centroid cosine < 0.5 (down from 0.981 baseline)
- Color flip cosine < -0.5 (mean across anchor positions)
- Selfplay win rate > 90% (White geometry MCTS vs random Black)
- test_geometry.py Test 3 (color flip symmetry): PASS

---

## Stage 2 — KR vs K

**Why second:** Rook mates are harder and longer than queen mates (up to 16 moves). The king must be pushed to the edge. Introduces the concept of piece cooperation — king and rook together, neither alone suffices. The geometry needs to encode king position relative to the rook's control, not just material count.

**Position generation:**
- Enumerate all legal KR vs K positions
- ~120k positions

**Selfplay loop:**
- White: geometry MCTS (init from Stage 1 weights)
- Black: geometry MCTS (same model, but minimising white win projection)
- Both sides play geometry — this is the first stage where Black is not random
- Starting positions: random KR vs K with white king not adjacent to rook (avoid trivial positions)
- Label: terminal outcome only

**Key difference from Stage 1:**
- Black now plays geometry MCTS too, creating genuine opposition
- This forces the model to learn that the winning side must *actively improve* geometry, not just wait
- The rook's interference geometry (open lines toward the black king) becomes directly relevant

**Advancement criteria:**
- Centroid cosine < 0.4
- Color flip cosine < -0.6
- White selfplay win rate > 80% (harder than KQ — some positions require long play)
- Test 4 (forced mate convergence): PASS

---

## Stage 3 — K+P vs K

**Why third:** Introduces the pawn — the piece with directional geometry (pawns only move forward). Introduces promotion. Critically: some K+P vs K positions are draws (opposition, stalemate traps). The geometry must learn the draw region, not just the win/loss axis.

**Position generation:**
- All legal K+P vs K positions with pawn not yet promoted
- Exclude trivially won (pawn on 7th with clear path) and trivially drawn positions at generation time
- ~80k contested positions

**Selfplay loop:**
- White: geometry MCTS
- Black: geometry MCTS
- Label: terminal outcome (win/draw/loss)
- Draw label = 0.0, forced win = +1.0, forced loss = -1.0

**New geometry requirement:**
- The geometry must now represent three regions, not two
- The draw region should sit between win and loss, not on either side
- This is the first test of whether per-piece geometry handles the directional pawn correctly

**Advancement criteria:**
- Selfplay outcome distribution matches known tablebase ratios (roughly 60% white wins, 40% draws in random K+P vs K)
- Color flip cosine < -0.6
- Draw positions cluster between win/loss centroids in PCA projection

---

## Stage 4 — KQ vs KR

**Why fourth:** Introduces a defensive piece. Black now has a resource — the rook can interpose, fork, or sacrifice to reach a draw. The geometry must encode the defensive resource as well as the attack. This is the first position class where Black can actually hold with correct play (~15% theoretical draws from random positions).

**Selfplay loop:**
- White: geometry MCTS
- Black: geometry MCTS (genuinely tries to hold)
- Bootstrap: SF depth 10 on 10k positions (tactics matter here — SF needed)
- Label: SF depth 10 for bootstrap, terminal outcome for ongoing selfplay

**Piece interference geometry becomes critical here:**
- The rook's interference geometry on the win/loss axis matters for Black
- A rook blocking the queen's access to the king changes the geometric reading of the position
- If interference geometry is not encoded, the model cannot represent "rook interpose blocks mate"

**Advancement criteria:**
- Draw rate in selfplay converges toward theoretical (~15%)
- White win rate > 70% (consistent with theoretical win rate from random positions)
- test_geometry.py Test 1 (material monotonicity): PASS (KQ > KR encoded)

---

## Stage 5 — Pawnless middlegame (2–4 pieces per side)

**Why fifth:** Random piece combinations. No pawns — positions are simpler to enumerate and selfplay stays in a bounded complexity. The model must now generalise piece interaction geometry beyond the specific endgame patterns learned in Stages 1–4.

**Position generation:**
- Random piece placement: 2–4 pieces per side (no pawns), kings included
- Filter illegal positions (pieces overlapping, kings adjacent, side to move in illegal check)
- 50k random positions per training batch, regenerated each epoch

**Selfplay loop:**
- Both sides: geometry MCTS
- SF depth 10 for labels (needed — these positions are not trivially evaluable)
- Games capped at 100 moves (draw by repetition or move limit)

**Advancement criteria:**
- test_geometry.py Test 2 (piece value ordering): PASS
- test_geometry.py Test 5 (transposition consistency): PASS (already passes — this is a regression check)
- Selfplay draw rate < 40% (pawnless positions should resolve)

---

## Stage 6 — Pawn endgames

**Why sixth:** Pawns add direction, promotion, and passed pawn geometry. This is the most complex endgame class. The model must encode: passed pawn distance to promotion, king proximity to pawns, pawn structure (doubled, isolated, connected). These are all interference and per-piece geometry problems.

**Position generation:**
- Random pawn endgame positions: 2–6 pawns per side, kings, no other pieces
- Filter obviously illegal positions
- Selfplay-generated positions preferred over random (richer distribution)

**Selfplay loop:**
- Both sides: geometry MCTS
- SF depth 15 for labels (pawn endgames have deep tactics)
- This is the first stage where SF depth matters meaningfully

**Advancement criteria:**
- test_geometry.py Test 4 (forced mate convergence): PASS with pawns
- Selfplay produces recognisable pawn endgame patterns (passed pawn rush, king opposition)

---

## Stage 7 — Full game

**Why last:** Only after the geometry is anchored on endgames does full-game training make sense. The geometry has a foundation. When a middlegame position simplifies into an endgame that the model knows deeply, the geometry correctly identifies it. The value head is no longer the only signal — the geometry pulls it toward the right answer.

**This is where Phase 1 full selfplay resumes**, but with a model whose geometry is already structured. The antipodal loss stays active throughout. The passenger is already in the driver's seat by the time full complexity arrives.

---

## Geometry tests at each stage

Run `test_geometry.py` at every stage gate. Expected progression:

| Stage | Centroid cos | Color flip | Test 3 | Test 4 |
|-------|-------------|------------|--------|--------|
| Baseline (R6) | 0.981 | +0.521 | FAIL | FAIL |
| After Stage 1 | < 0.5 | < -0.5 | PASS | — |
| After Stage 2 | < 0.4 | < -0.6 | PASS | PASS |
| After Stage 4 | < 0.3 | < -0.7 | PASS | PASS |
| After Stage 6 | < 0.2 | < -0.8 | PASS | PASS |

If a stage fails to move the needle after 3 training runs, the per-piece architecture or antipodal loss weight needs adjustment — not more data.

---

## What selfplay generates that Stockfish cannot

Selfplay in endgame positions generates **sequences** — the path from a contested position to a terminal outcome. This is training data for the geometry transition function `f(g_t, move) → g_{t+1}`. Every selfplay game is a sequence of (geometry, move, next_geometry) triples. Once Stage 2 or 3 has a stable geometry space, these sequences can train the transition function in parallel with the value training.

This is the path to geometry MCTS without full board forward passes at every node.
