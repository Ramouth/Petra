#!/bin/sh
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -W 2:00
#BSUB -J petra_stage1_geometry
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_stage1_geometry.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_stage1_geometry.err

cd /zhome/81/b/206091/Petra-Phase1

echo "=== Stage 1: KQ vs K geometry curriculum ==="

echo "--- Generating positions ---"
/zhome/81/b/206091/petra-env/bin/python3 src/generate_endgame.py \
  --positions 10000 \
  --out data/kqk_stage1.pt

echo "--- Training with antipodal loss ---"
/zhome/81/b/206091/petra-env/bin/python3 src/train.py \
  --dataset data/kqk_stage1.pt \
  --out models/geometry/stage1 \
  --epochs 50 \
  --patience 10 \
  --lr 1e-3 \
  --antipodal-weight 0.1 \
  --antipodal-margin 0.0 \
  --policy-weight 0.0

echo "--- Geometry probe ---"
/zhome/81/b/206091/petra-env/bin/python3 src/test_geometry.py \
  --model models/geometry/stage1/best.pt
