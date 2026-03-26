#!/bin/bash
# submit_round.sh — Submit one full zigzag round as chained LSF jobs.
#
# Usage:
#   bash jobs/submit_round.sh 1          # submit round 1
#   bash jobs/submit_round.sh 2          # submit round 2
#   bash jobs/submit_round.sh 1 --dry-run
#
# Jobs are chained: each step only starts when the previous one succeeds.
# All output goes to logs/lsf_r{N}_{step}.log
#
# Round parameters (from ZIGZAG.md):
#   Round 1: n_sim=40,  sf_depth=12, n_games=500, lr=5e-4
#   Round 2: n_sim=80,  sf_depth=15, n_games=500, lr=3e-4
#   Round 3: n_sim=160, sf_depth=18, n_games=500, lr=1e-4
#   Round 4: n_sim=320, sf_depth=20, n_games=500, lr=5e-5

set -e

ROUND=${1:-1}
DRY_RUN=${2:-""}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src"
DATA="$ROOT/data"
MODELS="$ROOT/models"
LOGS="$ROOT/logs"
VENV="$ROOT/../petra-env"   # adjust if venv is elsewhere
PYTHON="$VENV/bin/python3"

mkdir -p "$LOGS" "$DATA" "$MODELS/zigzag/r$ROUND"

# --- Round parameters ---
case $ROUND in
  1) N_SIM=40;  SF_DEPTH=12; N_GAMES=500; LR=5e-4; PREV_MODEL="$MODELS/sf/best.pt" ;;
  2) N_SIM=80;  SF_DEPTH=15; N_GAMES=500; LR=3e-4; PREV_MODEL="$MODELS/zigzag/r1/best.pt" ;;
  3) N_SIM=160; SF_DEPTH=18; N_GAMES=500; LR=1e-4; PREV_MODEL="$MODELS/zigzag/r2/best.pt" ;;
  4) N_SIM=320; SF_DEPTH=20; N_GAMES=500; LR=5e-5; PREV_MODEL="$MODELS/zigzag/r3/best.pt" ;;
  *) echo "Unknown round: $ROUND"; exit 1 ;;
esac

SELFPLAY_PT="$DATA/selfplay_r${ROUND}.pt"
SF_PT="$DATA/selfplay_r${ROUND}_sf.pt"
OUT_MODEL="$MODELS/zigzag/r$ROUND"
WORKERS=32

echo "Submitting round $ROUND  (n_sim=$N_SIM, sf_depth=$SF_DEPTH, n_games=$N_GAMES, lr=$LR)"
echo "  prev model : $PREV_MODEL"
echo "  workers    : $WORKERS"
echo "  logs       : $LOGS/lsf_r${ROUND}_*.log"
[ -n "$DRY_RUN" ] && echo "  [DRY RUN — not submitting]"
echo

# ---------------------------------------------------------------------------
# Step 1: Self-play  (32 cores, ~2h wall)
# ---------------------------------------------------------------------------
SELFPLAY_CMD="cd $SRC && $PYTHON selfplay.py \
  --model $PREV_MODEL \
  --games $N_GAMES \
  --n-sim $N_SIM \
  --out $SELFPLAY_PT \
  --workers $WORKERS"

if [ -n "$DRY_RUN" ]; then
  echo "[selfplay]  $SELFPLAY_CMD"
  JID_SELFPLAY="12345"
else
  JID_SELFPLAY=$(bsub \
    -q hpc \
    -n $WORKERS \
    -R "span[hosts=1]" \
    -W 4:00 \
    -J "petra_r${ROUND}_selfplay" \
    -o "$LOGS/lsf_r${ROUND}_selfplay.log" \
    -e "$LOGS/lsf_r${ROUND}_selfplay.err" \
    "$SELFPLAY_CMD" \
    | grep -oP '(?<=Job <)\d+')
  echo "Submitted selfplay     job $JID_SELFPLAY"
fi

# ---------------------------------------------------------------------------
# Step 2: SF re-label  (1 core, ~15 min wall)
# ---------------------------------------------------------------------------
REEVAL_CMD="cd $SRC && $PYTHON reeval_stockfish.py \
  --dataset $SELFPLAY_PT \
  --out $SF_PT \
  --depth $SF_DEPTH \
  --stockfish ~/bin/stockfish"

if [ -n "$DRY_RUN" ]; then
  echo "[reeval]    $REEVAL_CMD"
  JID_REEVAL="12346"
else
  JID_REEVAL=$(bsub \
    -q hpc \
    -n 1 \
    -W 1:00 \
    -J "petra_r${ROUND}_reeval" \
    -w "done($JID_SELFPLAY)" \
    -o "$LOGS/lsf_r${ROUND}_reeval.log" \
    -e "$LOGS/lsf_r${ROUND}_reeval.err" \
    "$REEVAL_CMD" \
    | grep -oP '(?<=Job <)\d+')
  echo "Submitted reeval       job $JID_REEVAL  (after $JID_SELFPLAY)"
fi

# ---------------------------------------------------------------------------
# Step 3: Train  (1 core, ~3h wall)
# ---------------------------------------------------------------------------
TRAIN_CMD="cd $SRC && $PYTHON train.py \
  --dataset $SF_PT \
  --out $OUT_MODEL \
  --lr $LR \
  --epochs 15 \
  --patience 3 \
  --init-model $PREV_MODEL"

if [ -n "$DRY_RUN" ]; then
  echo "[train]     $TRAIN_CMD"
  JID_TRAIN="12347"
else
  JID_TRAIN=$(bsub \
    -q hpc \
    -n 1 \
    -W 4:00 \
    -J "petra_r${ROUND}_train" \
    -w "done($JID_REEVAL)" \
    -o "$LOGS/lsf_r${ROUND}_train.log" \
    -e "$LOGS/lsf_r${ROUND}_train.err" \
    "$TRAIN_CMD" \
    | grep -oP '(?<=Job <)\d+')
  echo "Submitted train        job $JID_TRAIN  (after $JID_REEVAL)"
fi

# ---------------------------------------------------------------------------
# Step 4: Gate evaluation  (4 cores, ~2h wall)
# ---------------------------------------------------------------------------
GATE_CMD="cd $SRC && $PYTHON evaluate.py \
  --model $OUT_MODEL/best.pt \
  --step 5 \
  --games 100 \
  --n-sim $N_SIM \
  --temp-moves 10"

if [ -n "$DRY_RUN" ]; then
  echo "[gate]      $GATE_CMD"
else
  JID_GATE=$(bsub \
    -q hpc \
    -n 4 \
    -R "span[hosts=1]" \
    -W 4:00 \
    -J "petra_r${ROUND}_gate" \
    -w "done($JID_TRAIN)" \
    -o "$LOGS/lsf_r${ROUND}_gate.log" \
    -e "$LOGS/lsf_r${ROUND}_gate.err" \
    "$GATE_CMD" \
    | grep -oP '(?<=Job <)\d+')
  echo "Submitted gate         job $JID_GATE  (after $JID_TRAIN)"
fi

echo
echo "Monitor with: bjobs -J 'petra_r${ROUND}*'"
echo "Logs in:      $LOGS/lsf_r${ROUND}_*.log"
