#!/bin/bash
# submit_round.sh — Submit one full R3.1 zigzag round as chained LSF jobs.
#
# Usage:
#   bash jobs/submit_round.sh 1          # round 1 — cold start, no prior model
#   bash jobs/submit_round.sh 2          # round 2 — continues from round 1
#   bash jobs/submit_round.sh 1 --dry-run
#
# Steps per round:
#   1. init    (round 1 only) — save freshly initialised PetraNet weights
#   2. selfplay               — generate games with current model
#   3. reeval                 — replace game outcomes with SF evaluation scores
#   4. train                  — train on SF-labelled positions
#   5. probe                  — test_geometry.py (structure measurement each round)
#   6. gate                   — MCTS(learned) vs MCTS(material) or vs prev round

set -e

ROUND=${1:-1}
DRY_RUN=${2:-""}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src"
DATA="$ROOT/data"
MODELS="$ROOT/models/zigzag"
LOGS="$ROOT/logs"
VENV="$ROOT/../petra-env"
PYTHON="$VENV/bin/python3"
SF_BIN="/zhome/81/b/206091/bin/stockfish"

mkdir -p "$LOGS" "$DATA" "$MODELS/r${ROUND}"

# ---------------------------------------------------------------------------
# Round parameters
# ---------------------------------------------------------------------------
case $ROUND in
  1) N_SIM=50;  SF_DEPTH=15; N_GAMES=300; LR=1e-3; EPOCHS=20; PATIENCE=5 ;;
  2) N_SIM=100; SF_DEPTH=15; N_GAMES=400; LR=5e-4; EPOCHS=20; PATIENCE=5 ;;
  3) N_SIM=200; SF_DEPTH=18; N_GAMES=500; LR=3e-4; EPOCHS=15; PATIENCE=5 ;;
  4) N_SIM=400; SF_DEPTH=20; N_GAMES=500; LR=1e-4; EPOCHS=15; PATIENCE=5 ;;
  5) N_SIM=400; SF_DEPTH=20; N_GAMES=500; LR=5e-5; EPOCHS=15; PATIENCE=5 ;;
  6) N_SIM=400; SF_DEPTH=22; N_GAMES=500; LR=2e-5; EPOCHS=15; PATIENCE=5 ;;
  *) echo "Unknown round: $ROUND (supported: 1-6)"; exit 1 ;;
esac

WORKERS=32
SELFPLAY_PT="$DATA/selfplay_r${ROUND}.pt"
SF_PT="$DATA/selfplay_r${ROUND}_sf.pt"
OUT_MODEL="$MODELS/r${ROUND}"

if [ "$ROUND" -eq 1 ]; then
  PREV_MODEL="$MODELS/r0/best.pt"
else
  PREV_MODEL="$MODELS/r$((ROUND-1))/best.pt"
fi

echo "Submitting R3.1 round $ROUND"
echo "  n_sim=$N_SIM  sf_depth=$SF_DEPTH  n_games=$N_GAMES  lr=$LR"
echo "  prev model : $PREV_MODEL"
echo "  out model  : $OUT_MODEL/best.pt"
[ -n "$DRY_RUN" ] && echo "  [DRY RUN — not submitting]"
echo

# ---------------------------------------------------------------------------
# Helper: write a job script to a temp file and submit it, returning job ID
# ---------------------------------------------------------------------------
_submit() {
  local script="$1"
  local depends="$2"
  local tmpfile
  tmpfile=$(mktemp /tmp/petra_XXXXX.sh)
  cat "$script" > "$tmpfile"
  if [ -n "$depends" ]; then
    # Inject dependency line after the last #BSUB line
    sed -i "/#BSUB/!{0,/#BSUB/!b}; \$!{/#BSUB/{n; /#BSUB/!i#BSUB -w \"done($depends)\"
}}" "$tmpfile" 2>/dev/null || true
    # Simpler approach: just append -w to bsub call
    local jid
    jid=$(bsub -w "done($depends)" < "$tmpfile" | grep -oP '(?<=Job <)\d+')
  else
    local jid
    jid=$(bsub < "$tmpfile" | grep -oP '(?<=Job <)\d+')
  fi
  rm -f "$tmpfile"
  echo "$jid"
}

# ---------------------------------------------------------------------------
# Step 0 (round 1 only): initialise fresh model weights
# ---------------------------------------------------------------------------
if [ "$ROUND" -eq 1 ]; then
  mkdir -p "$MODELS/r0"
  INIT_SCRIPT=$(mktemp /tmp/petra_init_XXXXX.sh)
  cat > "$INIT_SCRIPT" << SCRIPT
#!/bin/sh
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 2GB
#BSUB -W 0:10
#BSUB -J petra_r${ROUND}_init
#BSUB -o $LOGS/lsf_r${ROUND}_init.log
#BSUB -e $LOGS/lsf_r${ROUND}_init.err

$PYTHON - << 'EOF'
import sys; sys.path.insert(0, '$SRC')
import os, torch
from model import PetraNet
os.makedirs('$MODELS/r0', exist_ok=True)
model = PetraNet()
torch.save(model.state_dict(), '$MODELS/r0/best.pt')
print("Fresh PetraNet saved → $MODELS/r0/best.pt")
EOF
SCRIPT

  if [ -n "$DRY_RUN" ]; then
    echo "[init]  would save fresh PetraNet → $MODELS/r0/best.pt"
    JID_INIT="10000"
  else
    JID_INIT=$(bsub < "$INIT_SCRIPT" | grep -oP '(?<=Job <)\d+')
    echo "Submitted init         job $JID_INIT"
  fi
  rm -f "$INIT_SCRIPT"
  AFTER_INIT="-w done($JID_INIT)"
else
  AFTER_INIT=""
  JID_INIT=""
fi

# ---------------------------------------------------------------------------
# Step 1: Self-play
# ---------------------------------------------------------------------------
SP_SCRIPT=$(mktemp /tmp/petra_sp_XXXXX.sh)
cat > "$SP_SCRIPT" << SCRIPT
#!/bin/sh
#BSUB -q hpc
#BSUB -n $WORKERS
#BSUB -R "span[hosts=1] rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 4:00
#BSUB -J petra_r${ROUND}_selfplay
#BSUB -o $LOGS/lsf_r${ROUND}_selfplay.log
#BSUB -e $LOGS/lsf_r${ROUND}_selfplay.err

cd $SRC
$PYTHON selfplay.py \
  --model $PREV_MODEL \
  --games $N_GAMES \
  --n-sim $N_SIM \
  --out $SELFPLAY_PT \
  --workers $WORKERS
SCRIPT

if [ -n "$DRY_RUN" ]; then
  echo "[selfplay]  n_games=$N_GAMES  n_sim=$N_SIM  model=$PREV_MODEL"
  JID_SP="10001"
else
  if [ -n "$JID_INIT" ]; then
    JID_SP=$(bsub -w "done($JID_INIT)" < "$SP_SCRIPT" | grep -oP '(?<=Job <)\d+')
  else
    JID_SP=$(bsub < "$SP_SCRIPT" | grep -oP '(?<=Job <)\d+')
  fi
  echo "Submitted selfplay     job $JID_SP"
fi
rm -f "$SP_SCRIPT"

# ---------------------------------------------------------------------------
# Step 2: SF re-label
# ---------------------------------------------------------------------------
REEVAL_SCRIPT=$(mktemp /tmp/petra_rv_XXXXX.sh)
cat > "$REEVAL_SCRIPT" << SCRIPT
#!/bin/sh
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 2:00
#BSUB -J petra_r${ROUND}_reeval
#BSUB -o $LOGS/lsf_r${ROUND}_reeval.log
#BSUB -e $LOGS/lsf_r${ROUND}_reeval.err

module load gcc/12.3.0-binutils-2.40

cd $SRC
$PYTHON reeval_stockfish.py \
  --dataset $SELFPLAY_PT \
  --out $SF_PT \
  --depth $SF_DEPTH \
  --stockfish $SF_BIN
SCRIPT

if [ -n "$DRY_RUN" ]; then
  echo "[reeval]    depth=$SF_DEPTH"
  JID_REEVAL="10002"
else
  JID_REEVAL=$(bsub -w "done($JID_SP)" < "$REEVAL_SCRIPT" | grep -oP '(?<=Job <)\d+')
  echo "Submitted reeval       job $JID_REEVAL  (after $JID_SP)"
fi
rm -f "$REEVAL_SCRIPT"

# ---------------------------------------------------------------------------
# Step 3: Train
# ---------------------------------------------------------------------------
if [ "$ROUND" -eq 1 ]; then
  INIT_MODEL_FLAG=""
else
  INIT_MODEL_FLAG="--init-model $PREV_MODEL"
fi

TRAIN_SCRIPT=$(mktemp /tmp/petra_tr_XXXXX.sh)
cat > "$TRAIN_SCRIPT" << SCRIPT
#!/bin/sh
#BSUB -q hpc
#BSUB -n 2
#BSUB -R "span[hosts=1] rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 4:00
#BSUB -J petra_r${ROUND}_train
#BSUB -o $LOGS/lsf_r${ROUND}_train.log
#BSUB -e $LOGS/lsf_r${ROUND}_train.err

cd $SRC
$PYTHON train.py \
  --dataset $SF_PT \
  --out $OUT_MODEL \
  --lr $LR \
  --epochs $EPOCHS \
  --patience $PATIENCE \
  $INIT_MODEL_FLAG
SCRIPT

if [ -n "$DRY_RUN" ]; then
  echo "[train]     lr=$LR  epochs=$EPOCHS  init=${INIT_MODEL_FLAG:-none}"
  JID_TRAIN="10003"
else
  JID_TRAIN=$(bsub -w "done($JID_REEVAL)" < "$TRAIN_SCRIPT" | grep -oP '(?<=Job <)\d+')
  echo "Submitted train        job $JID_TRAIN  (after $JID_REEVAL)"
fi
rm -f "$TRAIN_SCRIPT"

# ---------------------------------------------------------------------------
# Step 4: Geometry probe
# ---------------------------------------------------------------------------
PROBE_SCRIPT=$(mktemp /tmp/petra_pb_XXXXX.sh)
cat > "$PROBE_SCRIPT" << SCRIPT
#!/bin/sh
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 2GB
#BSUB -W 0:30
#BSUB -J petra_r${ROUND}_probe
#BSUB -o $LOGS/lsf_r${ROUND}_probe.log
#BSUB -e $LOGS/lsf_r${ROUND}_probe.err

cd $SRC
$PYTHON test_geometry.py --model $OUT_MODEL/best.pt
SCRIPT

if [ -n "$DRY_RUN" ]; then
  echo "[probe]     test_geometry.py on $OUT_MODEL/best.pt"
  JID_PROBE="10004"
else
  JID_PROBE=$(bsub -w "done($JID_TRAIN)" < "$PROBE_SCRIPT" | grep -oP '(?<=Job <)\d+')
  echo "Submitted probe        job $JID_PROBE  (after $JID_TRAIN)"
fi
rm -f "$PROBE_SCRIPT"

# ---------------------------------------------------------------------------
# Step 5: Gate evaluation
# ---------------------------------------------------------------------------
if [ "$ROUND" -eq 1 ]; then
  BASELINE_FLAG=""          # round 1: vs MCTS(material)
else
  BASELINE_FLAG="--baseline-model $PREV_MODEL"   # round 2+: vs previous round
fi

GATE_SCRIPT=$(mktemp /tmp/petra_gt_XXXXX.sh)
cat > "$GATE_SCRIPT" << SCRIPT
#!/bin/sh
#BSUB -q hpc
#BSUB -n $WORKERS
#BSUB -R "span[hosts=1] rusage[mem=2GB]"
#BSUB -M 2GB
#BSUB -W 4:00
#BSUB -J petra_r${ROUND}_gate
#BSUB -o $LOGS/lsf_r${ROUND}_gate.log
#BSUB -e $LOGS/lsf_r${ROUND}_gate.err

cd $SRC
$PYTHON evaluate.py \
  --model $OUT_MODEL/best.pt \
  $BASELINE_FLAG \
  --step 5 \
  --games 200 \
  --n-sim $N_SIM \
  --workers $WORKERS \
  --pgn-out $LOGS/lsf_r${ROUND}_gate.pgn
SCRIPT

if [ -n "$DRY_RUN" ]; then
  echo "[gate]      step 5  games=200  baseline=${BASELINE_FLAG:-material}"
else
  JID_GATE=$(bsub -w "done($JID_TRAIN)" < "$GATE_SCRIPT" | grep -oP '(?<=Job <)\d+')
  echo "Submitted gate         job $JID_GATE  (after $JID_TRAIN)"
fi
rm -f "$GATE_SCRIPT"

echo
echo "Monitor with: bjobs -J 'petra_r${ROUND}*'"
echo "Logs in:      $LOGS/lsf_r${ROUND}_*.log"
