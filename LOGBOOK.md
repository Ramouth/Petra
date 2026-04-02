# Petra — Training Logbook

Per-round record of all training runs. Each entry covers: selfplay, reeval, training, geometry probe, and gate. Rounds within each version are numbered from R1.

---

## Version 3 — STM-fix complete redo
*Started 2026-04-02. Branch: phase2.*

Foundation: Stage 3 geometry model (STM-sign fix, trained on endgame curriculum stages 1–8, gate 54% vs material) serves as **R0** bootstrap for self-play.

Gate threshold: **>55%** win rate over 200 games.

---

### R0 — Bootstrap (2026-04-02)

Stage 3 geometry model. Not a self-play round — serves as the selfplay engine for R1.

| Param | Value |
|-------|-------|
| Source | `models/geometry/stage3/best.pt` |
| Architecture | PetraNet 849k params, STM-sign fix, full value head |
| Gate (vs material, step 6) | 54.0% (+28 ELO) |
| Probe | 3/5 — PASS: color flip symmetry, forced mate conv, transposition |
| Centroid cosine | 0.03 |
| Notes | "Passenger problem": value head compensated for unstructured geometry. Gate passes but geometry representation is weak. Used as selfplay seed only. |

---

### R1 — 2026-04-02

**Selfplay**

| Param | Value |
|-------|-------|
| Model | R0 (stage3 bootstrap) |
| Games | 300 |
| n_sim | 50 |
| MAX_POSITIONS_PER_GAME | 50 |
| Positions generated | 3,522 |

**Reeval**

| Param | Value |
|-------|-------|
| SF depth | 15 |
| Positions after reeval | 3,346 train / 176 val |
| Mean value (tanh-squashed) | +0.280 |
| Decisive (|v| > 0.5) | 79.2% |

**Training**

| Param | Value |
|-------|-------|
| Init model | None (random weights) |
| LR | 1e-3 |
| Epochs | 19 (early stop, patience 5) |
| Best val loss | 7.0017 |
| Architecture | 840,961 params (thin value head) |
| Sanity check | FAIL — "Black up queen" value positive (white-biased) |

**Geometry Probe** *(test_geometry.py — corrected Test 3)*

| Test | Result |
|------|--------|
| 1 — Material monotonicity | FAIL (pawn negative delta) |
| 2 — Piece value ordering | FAIL (R > B fails) |
| 3 — Antipodal symmetry | FAIL — mean cosine n/a (old test used wrong metric; see R2) |
| 4 — Forced mate convergence | WARN (2/4 non-monotone) |
| 5 — Transposition consistency | PASS |
| **Total** | **1/5** |
| Centroid cosine | 0.8412 (baseline R4: 0.869) |

**Gate**

| Param | Value |
|-------|-------|
| Baseline | Material heuristic |
| Games | 200 |
| n_sim | 50 |
| Score | 51.7% (W=28 D=151 L=21) |
| ELO Δ | +12 |
| Verdict | **FAIL** (<55% threshold) |

**Notes:** R1 started from random weights, not from R0 geometry — explains the white-biased sanity check failure. The STM-sign fix is present in the architecture but the training signal (white-relative SF labels) dominates. Selfplay data volume is small (3.5k positions). Model is learning but weak.

---

### R2 — 2026-04-02

**Selfplay**

| Param | Value |
|-------|-------|
| Model | R1 |
| Games | 400 |
| n_sim | 100 |
| MAX_POSITIONS_PER_GAME | 50 |
| Positions generated | 18,755 |

**Reeval**

| Param | Value |
|-------|-------|
| SF depth | 15 |
| Positions after reeval | 17,818 train / 937 val |
| Mean value (tanh-squashed) | +0.116 |
| Decisive (|v| > 0.5) | 87.9% |

**Training**

| Param | Value |
|-------|-------|
| Init model | R1 |
| LR | 5e-4 |
| Epochs | 20 (full, no early stop) |
| Best val loss | 4.9079 |
| Architecture | 840,961 params (thin value head) |
| Sanity check | PASS — all sign checks correct |

**Geometry Probe** *(test_geometry.py — corrected Test 3, run locally 2026-04-02)*

| Test | Result |
|------|--------|
| 1 — Material monotonicity | FAIL (pawn negative delta) |
| 2 — Piece value ordering | **PASS** — Q > R > B > N > P all correct |
| 3 — Antipodal symmetry | **WARN** — mean cosine -0.40 (KQ vs K: -0.54 PASS; others -0.24 to -0.47 WARN) |
| 4 — Forced mate convergence | FAIL (3/4 non-monotone) |
| 5 — Transposition consistency | PASS |
| **Total** | **2/5** |
| Centroid cosine | 0.5413 (improving from R1: 0.8412) |

*Note: old Test 3 checked win-projection invariance — impossible without STM-sign fix, so always FAIL. Corrected test checks cosine_sim(g, g_flip) → target -1.0. WARN at -0.40 means partial antipodal structure learned from data alone. Old V2 R2 was at +0.64 (no structure). Genuine improvement.*

**Gate**

| Param | Value |
|-------|-------|
| Baseline | R1 |
| Games | 200 |
| n_sim | 100 |
| Score | 54.8% (W=45 D=129 L=26) |
| ELO Δ | +33 |
| Verdict | **FAIL** (<55% threshold, 0.2% short) |

**Notes:** Large jump in data volume (3.5k → 18.8k positions, 5×). Piece value ordering now fully correct — first time all five ordering tests pass. Centroid separation improving steadily. STM symmetry failure is structural: SF labels are white-relative, so geometry remains white-biased regardless of the STM-sign fix in the architecture. R2 clearly beats R1 (+33 ELO) — the model is learning. Gate missed by 0.2%.

---

## Version 2 — PetraNet zigzag + antipodal curriculum
*2026-03-24 to 2026-04-01. See ZIGZAG.md, CURRICULUM.md, PLAN.md, ROADMAP.md.*

**Key results:**

| Run | Type | Gate | Notes |
|-----|------|------|-------|
| R4 | Zigzag selfplay | **67.0% vs material** (+123 ELO) | Best zigzag model. ReLU bottleneck — geometry not antipodal-capable. |
| R5 | Zigzag selfplay | 37.0% vs R4 (-92 ELO) | Regression. Trained on geometry-modified data with STM bug. |
| Stage 2 | Endgame curriculum | 39.5% vs material (-74 ELO) | Antipodal loss + STM bug → constraint unsatisfiable. |
| Stage 3 | Endgame curriculum | **54.0% vs material** (+28 ELO) | STM-sign fix. Antipodal works. Passenger problem in value head. |
| Stage 4 | Thin value head | Timed out | Antipodal loss = 0.0 immediately. Geometry 1/5 tests. |

---

## Version 1 — Maia-based geometry encoder
*2026-03-20 to 2026-03-24.*

Geometry encoder on Maia-1500 policy. Maia representations took over; geometry was a passenger. v6 appeared to work due to a sign bug. End state: Petra ≈ Maia-1500 greedy.

---

## Open questions as of R2

1. **Antipodal symmetry**: R2 already shows -0.40 mean cosine without the STM-sign fix — partial structure learned from data. Adding the STM-sign fix to V3 (requires syncing updated model.py to HPC before R3) should push this toward -1.0. The old Test 3 was incorrectly reporting this as FAIL; corrected test shows WARN.

2. **Gate threshold**: R2 is at 54.8% vs R1, missing 55% by 0.2%. R3 with more data and higher n_sim should clear it.

3. **Data volume**: R2 used 18.8k positions. Version 2 zigzag used 2.3k positions (R4). More data is not hurting.
