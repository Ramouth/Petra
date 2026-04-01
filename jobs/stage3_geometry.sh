#!/bin/sh
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -W 4:00
#BSUB -J petra_stage3_geometry
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_stage3_geometry.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_stage3_geometry.err

cd /zhome/81/b/206091/Petra-Phase1

echo "=== Stage 3: mixed endgame curriculum (stages 1-8), STM-aware ==="
# Fresh start — stage2 was trained with STM bug (all positions White to move).
# Fixes in this run:
#   - generate_endgame: ~50/50 White/Black to move, STM-relative labels
#   - antipodal loss: turn-flip (same position, opposite STM) instead of
#     color-flip (flipped colors had same label, contradicting antipodal constraint)
#   - geometry_value: negates projection for Black to move in evaluate.py
#
# Dataset is ~2x larger per epoch due to turn-flip variants (~64k positions).
# Reduce --endgame-positions to 8000 to keep epoch size comparable to stage2.
#
# 1: KQK   2: KRK   3: KPK        — who has piece wins
# 4: KQvKR 5: KRvKP               — stronger piece wins (fixes Q>R ordering)
# 6: KBvKP 7: KNvKP               — minor piece beats pawn (fixes B>P, N>P)
# 8: KPvKP balanced               — advancement heuristic (pawn structure geometry)
/zhome/81/b/206091/petra-env/bin/python3 src/train.py \
  --out models/geometry/stage3 \
  --epochs 80 \
  --patience 20 \
  --tight-patience 20 \
  --transition-drop 0.5 \
  --lr 1e-3 \
  --antipodal-weight 1.0 \
  --antipodal-margin 0.0 \
  --policy-weight 0.0 \
  --endgame-positions 8000 \
  --endgame-stages 1 2 3 4 5 6 7 8

echo "--- Geometry probe ---"
/zhome/81/b/206091/petra-env/bin/python3 src/test_geometry.py \
  --model models/geometry/stage3/best.pt

echo "--- Step 6: MCTS(geometry) vs MCTS(material) ---"
/zhome/81/b/206091/petra-env/bin/python3 src/evaluate.py \
  --model models/geometry/stage3/best.pt \
  --games 100 \
  --step 6 \
  --n-sim 200 \
  --workers 8
