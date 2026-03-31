#!/bin/sh
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -W 4:00
#BSUB -J petra_stage2_geometry
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_stage2_geometry.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_stage2_geometry.err

cd /zhome/81/b/206091/Petra-Phase1

echo "=== Stage 2: KR vs K geometry curriculum ==="

echo "--- Training with antipodal loss (per-epoch position regeneration) ---"
/zhome/81/b/206091/petra-env/bin/python3 src/train.py \
  --out models/geometry/stage2 \
  --epochs 80 \
  --patience 15 \
  --tight-patience 3 \
  --transition-drop 0.5 \
  --lr 1e-3 \
  --antipodal-weight 1.0 \
  --antipodal-margin 0.0 \
  --policy-weight 0.0 \
  --endgame-positions 10000 \
  --endgame-stage 2

echo "--- Geometry probe ---"
/zhome/81/b/206091/petra-env/bin/python3 src/test_geometry.py \
  --model models/geometry/stage2/best.pt
