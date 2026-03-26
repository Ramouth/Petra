# Petra Zigzag Training Plan

## What problem this solves

Supervised training on SF labels has a ceiling. The model learns to predict
Stockfish's evaluation of positions that appear in Lichess games. Two limits:

1. **Position distribution**: the model only sees positions humans play. Strong
   engine play reaches positions humans never visit — the model has no signal
   there.
2. **Label depth**: SF depth 10–15 has its own evaluation errors. The model
   cannot be more accurate than its teacher.

Pure self-play (AlphaZero-style) has no ceiling in principle but requires
the model to already be strong enough to generate useful games. Starting from
a weak value head (R²=0.18, game outcome labels), self-play produces noisy
outcome labels — the same problem that motivated SF re-labeling in the first
place.

The zigzag combines both:
- **Self-play** generates positions the model actually reaches — positions that
  are relevant to its current play style and strength level.
- **SF re-labeling** gives accurate values for those positions — no noisy
  outcome labels, no ceiling from human games.

Each round, the model trains on its own games evaluated by a slightly deeper
oracle. The oracle depth and the search depth both increase gradually. The
model is taught difference first, then nuance.

---

## The curriculum

| Round | n_sim (self-play) | SF depth | Games | Positions | What the model learns |
|-------|-------------------|----------|-------|-----------|----------------------|
| 0 | — | 10 | — | 200k | Bootstrap (done) |
| 1 | 40 | 12 | 500 | ~30k | Obvious tactics, clear material advantage |
| 2 | 80 | 15 | 500 | ~30k | 2–3 move combinations, simple plans |
| 3 | 160 | 18 | 500 | ~30k | Positional nuance, long-range vision |
| 4 | 320 | 20 | 500 | ~30k | Near-optimal evaluation |

**Why n_sim doubles:**
Each doubling roughly doubles the effective search depth. As the value head
gets sharper, the policy prior gets stronger, and the "information per
simulation" increases — so the model needs more simulations to move beyond
what the policy alone already knows. The search and the value head scale
together.

**Why SF depth increases:**
Once the model can predict depth-10 positions reasonably, depth-10 labels
become the ceiling — training against them adds no new gradient. Deeper
evals expose finer distinctions the model hasn't learned yet. The label
quality always needs to stay ahead of the model.

**Why 500 games per round:**
500 games at ~50 moves each ≈ 25k positions per round. With n_sim=40, each
game takes roughly 3–5 minutes on a single GPU (seconds on a cluster). On
CPU this is slow (~50 hours for round 1) — HPC is assumed for rounds 2+.
Round 1 can be prototyped on CPU at reduced scale (50 games).

---

## Data strategy

### Do we accumulate or discard?

**Discard old data.** Each round uses only the positions generated in that
round, labeled at that round's SF depth.

Reasons:
- Positions from round 1 labeled at depth 12 are inconsistent with round 3
  positions labeled at depth 18. Mixing them creates a noisy loss surface
  where two boards with the same evaluation score mean different things.
- The model in round 3 generates qualitatively different positions than in
  round 1. Round 1 positions may be blunder-filled middlegames that are no
  longer relevant to round 3's play style.
- Keeping all data would require re-labeling everything at each new depth —
  expensive and mostly wasteful for old positions the model has already
  "solved."

**Exception:** keep the original supervised dataset (200k positions, depth 10)
available as a sanity anchor for early rounds. If the model regresses on the
supervised val set, something is wrong.

### Policy target

During self-play, the MCTS visit distribution over legal moves is the policy
target — not the move played. This is the AlphaZero convention and is
strictly better than using only the played move: it contains information about
all explored alternatives, not just the chosen one.

The visit distribution is dense over the top few moves and near-zero elsewhere.
This is a much richer training signal than the one-hot policy target used in
supervised pretraining.

### Value target

After SF re-labeling: `tanh(cp / 400)` from SF at round depth. Same
convention as the bootstrap SF dataset.

**Before** re-labeling (if we ever need intermediate training): use game
outcome (+1, -1, -0.1 draw contempt). This is noisy but keeps the signs
correct.

### Position sampling from self-play games

Not every position in a game is equally informative. Conventions:

- **Skip the first 8 moves** (opening theory — low variance, model already
  knows these from Lichess pretraining).
- **Sample up to 12 positions per game** uniformly from the remaining moves
  (same as the Lichess data pipeline).
- **Resignation threshold**: if `|value| > 0.95` for 5 consecutive moves,
  declare the game over and assign the outcome. Avoids playing out hopeless
  positions for 200 moves.
- **Move limit**: 200 half-moves max. Games that reach the limit score as
  draws.

---

## The loop in detail

```
Round r:
  1. SELF-PLAY
     - Load current best model (models/zigzag/round_{r-1}/best.pt)
     - Play N games: MCTS(n_sim=N_r, temperature=1 for first 20 moves,
       then temperature=0, add_noise=True)
     - Save: (tensors, visit_distributions, outcomes, fens)
       → data/selfplay_round{r}.pt

  2. SF RE-LABEL
     - Load selfplay_round{r}.pt
     - Evaluate each FEN with Stockfish at depth D_r
     - Replace outcome values with tanh(cp/400)
     - Save: data/selfplay_round{r}_sf.pt

  3. TRAIN
     - Load models/zigzag/round_{r-1}/best.pt as starting weights
     - Train on data/selfplay_round{r}_sf.pt
       Value loss: MSE(predicted, SF_value)
       Policy loss: cross-entropy(logits, visit_distribution)  ← dense target
     - Early stopping on val loss (patience=3, tighter than supervised)
     - Save: models/zigzag/round{r}/best.pt

  4. EVALUATE
     - Run ablation step 5: MCTS(learned, n_sim=N_r) vs MCTS(material, n_sim=N_r)
     - Also: new model vs previous round model (direct progress check)
     - Gate: new model must beat previous round model at >55% to advance
     - Save results to logs/zigzag_round{r}_eval.log

  5. ACCEPT OR REJECT
     - ACCEPT: new model becomes current best, advance to round r+1
     - REJECT: keep current best, investigate (see failure modes below)
```

---

## Key design differences from supervised training

### Policy loss changes

In supervised pretraining, the policy target is one-hot (the move played in
the game). In self-play, the policy target is the full MCTS visit distribution.
This requires a change to the loss:

```python
# Supervised (current):
policy_loss = F.cross_entropy(logits, move_idx)  # one-hot target

# Self-play (new):
policy_loss = -(visit_dist * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
# Equivalent to KL(visit_dist || softmax(logits)), minimised w.r.t. logits
```

The visit distribution is sparse (most moves have 0 visits), so this is
efficient in practice. But it needs a small code change in `train.py`.

### Temperature schedule during self-play

- **Moves 1–20**: temperature=1.0 (sample from visit distribution)
  → game diversity, explore different openings
- **Moves 21+**: temperature=0.0 (greedy, pick most-visited move)
  → decisive play, fewer drawn games from mutual blundering

Dirichlet noise at the root (alpha=0.3, fraction=0.25) on every move during
self-play. This is disabled during evaluation games.

### Learning rate

After each round of self-play, the model is relatively close to a good
solution (it was already trained in round r-1). Start each round with a
lower LR than the initial supervised training:

```
Round 0 (supervised): lr = 1e-3
Round 1: lr = 5e-4
Round 2: lr = 3e-4
Round 3: lr = 1e-4
Round 4: lr = 5e-5
```

This prevents catastrophic forgetting of the supervised knowledge while
allowing the self-play signal to refine the model.

---

## selfplay.py — what needs building

This is the one missing script. It needs to:

1. Load the model
2. Play N games (configurable) using MCTS
3. For each game, record every position visited:
   - Board tensor (14, 8, 8)
   - MCTS visit distribution (4096-dim, sparse)
   - Game outcome (determined at end of game, propagated to all positions)
   - FEN (for SF re-labeling)
4. Save to a `.pt` file in the same format as `dataset.pt`

The self-play positions use the same `board_to_tensor` representation, so
`reeval_stockfish.py` can re-label them without changes. The `train.py` needs
only the policy loss modification above.

Estimated: ~200 lines.

---

## zigzag.py — thin orchestration

Calls the above scripts in sequence with the right parameters for each round.
Not complex logic — mostly argument plumbing and bookkeeping.

```python
ROUNDS = [
    # (n_sim, sf_depth, n_games, lr)
    (40,  12, 500, 5e-4),
    (80,  15, 500, 3e-4),
    (160, 18, 500, 1e-4),
    (320, 20, 500, 5e-5),
]
```

Estimated: ~100 lines.

---

## Failure modes and mitigations

### 1. Value collapse
The model regresses — value R² drops below the supervised baseline.

**Detection:** monitor R² on the original SF val set after every round.
**Mitigation:** mix 20% supervised data into each training round as an
anchor. If val R² drops below 0.4, revert to the previous round's model.

### 2. Policy collapse
The model converges to always playing the same opening, reducing game
diversity. Visit distributions become near-identical across games.

**Detection:** measure entropy of the visit distribution at the root position
across games. If mean entropy < 1.0 nat (out of ~log(20)≈3.0 for typical
20 legal moves), diversity has collapsed.
**Mitigation:** increase Dirichlet noise fraction from 0.25 to 0.4 for that
round. If persistent, add opening book randomisation (random first 4 moves).

### 3. Draw inflation
As the model gets stronger, it draws more against itself. This reduces the
value signal (draws score -0.1, a weak label).

**Detection:** draw rate > 70% in self-play games.
**Mitigation:** increase draw contempt from -0.1 to -0.3 to make the model
actively seek wins. Revisit if this causes reckless play.

### 4. Overfitting to self-play positions
The model gets very good at positions it generates but fails on novel
positions from the test set.

**Detection:** evaluate Greedy vs Random win rate after every round. If it
drops below 70%, the model has overfit to its own games.
**Mitigation:** include a random 10% sample from the original supervised
dataset in each training round.

### 5. n_sim outpaces value quality
If n_sim doubles but the value head hasn't improved, the extra simulations
just amplify whatever errors the value head already makes.

**Detection:** gate fails (new model < 55% vs previous round).
**Mitigation:** do not advance n_sim. Run an extra round at the same n_sim
and SF depth before stepping up.

---

## HPC requirements

**CPU-only access confirmed** (DTU gbar). No GPU — but the key HPC advantage
is parallelism, not raw speed. Self-play games are embarrassingly parallel
(independent processes), and Stockfish re-labeling is trivially parallel
(one SF process per core).

Each round on HPC (CPU, 32-core node assumed):

| Step | Time estimate | Parallelism | Notes |
|------|--------------|-------------|-------|
| Self-play, 500 games, n_sim=40 | ~3–4 hours | 32 games in parallel | Each game ~6 min on CPU |
| SF re-label, 30k positions, depth 12 | ~15 min | 32 Stockfish processes | Trivially parallel |
| Training, 30k positions, 10 epochs | ~2–3 hours | Single process | PyTorch CPU; main bottleneck |
| Evaluation, 100 games, n_sim=40 | ~1 hour | 4–8 parallel | |

Round 1 end-to-end: ~7 hours on HPC CPU.
Round 4 end-to-end: ~24 hours (n_sim=320, ~45 min/game, 500 games / 32 cores
= ~7 hours self-play alone).

**n_sim schedule may need to be more conservative** given CPU-only. Round 4
at n_sim=320 pushes the wall-clock limit of reasonable overnight jobs.
Consider capping at n_sim=160 (round 3) until GPU access is available.

**selfplay.py must support a `--workers N` flag** to run N games in parallel
(via multiprocessing). Each worker loads its own copy of the model. This is
the primary way to exploit HPC CPU cores.

On local CPU: prototype with 20–50 games, n_sim=40. A 50-game run takes
~5 hours — feasible overnight.

---

## Open questions to discuss

1. **Replay buffer size**: Should we keep a sliding window of the last 2 rounds
   of self-play data instead of discarding? More data = more stable training,
   but inconsistent SF depths.

2. **Mixing supervised data**: How much of the original SF-labelled data
   (200k positions at depth 10) should persist across rounds? 0%? 10%? 20%?
   Trade-off between stability and capacity to improve.

3. **Gate threshold**: 55% vs previous round model, or 55% vs MCTS(material)?
   The material baseline is fixed; the previous-round comparison measures
   direct improvement. Should we require both?

4. **n_sim schedule**: Is doubling every round right, or should it be every
   two rounds? CPU-only changes this — n_sim=320 at ~45 min/game means round
   4 self-play alone is 7+ hours even parallelised on 32 cores. Practical
   cap is probably n_sim=160 until GPU access. Could do rounds 1–3 on CPU
   HPC and defer round 4 until GPU is available.

5. **When to stop**: Is there a natural stopping condition beyond "passes the
   ELO gate"? Could track geometry space separation — stop when win/loss
   centroid cosine < 0.85 (strongly value-aware). Or just run all 4 rounds
   and evaluate.

6. **Policy loss target**: Use the raw visit counts or smooth them? Smoothing
   (e.g. temperature > 1 when computing the training target) prevents the
   model from being overconfident about moves explored only a few times.

7. **Resign threshold**: 0.95 is aggressive. At early rounds the model
   misevaluates many positions at 0.95+. Should the resign threshold relax
   in early rounds (e.g. 0.99 in round 1, 0.95 in round 4)?

---

## What's already built

| Component | Status | Notes |
|-----------|--------|-------|
| `board.py` | ✅ | Board tensor, move encoding |
| `model.py` | ✅ | PetraNet, value + policy heads |
| `mcts.py` | ✅ | AlphaZero-style search, visit distribution |
| `train.py` | ✅ (minor change) | Needs dense policy loss target |
| `reeval_stockfish.py` | ✅ | Plug in directly |
| `evaluate.py` | ✅ (fixed) | Gate test, ablation ladder |
| `probe_geometry.py` | ✅ | Geometry health check per round |
| `selfplay.py` | ❌ | Build next |
| `zigzag.py` | ❌ | Build after selfplay.py |

---

## Suggested first experiment (CPU prototype)

Before committing to HPC resources, validate the loop works at all:

```bash
# Round 1 prototype: 50 games, n_sim=40, SF depth 12
python3 src/selfplay.py \
    --model models/sf/best.pt \
    --games 50 \
    --n-sim 40 \
    --out data/selfplay_r1.pt

python3 src/reeval_stockfish.py \
    --dataset data/selfplay_r1.pt \
    --out data/selfplay_r1_sf.pt \
    --depth 12

python3 src/train.py \
    --dataset data/selfplay_r1_sf.pt \
    --out models/zigzag/r1/ \
    --lr 5e-4 \
    --epochs 10

python3 src/evaluate.py \
    --model models/zigzag/r1/best.pt \
    --step 5 \
    --games 40 \
    --n-sim 40
```

Success criterion: new model beats the SF baseline at >52% in 40 games.
Even marginal improvement confirms the loop is working and scales to HPC.
