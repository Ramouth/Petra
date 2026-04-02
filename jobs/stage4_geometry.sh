#!/bin/sh
#BSUB -q hpc
#BSUB -n 2
#BSUB -R "span[hosts=1] rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -W 6:00
#BSUB -J petra_stage4_geometry
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_stage4_geometry.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_stage4_geometry.err

cd /zhome/81/b/206091/Petra-Phase1

echo "=== Stage 4: Thin value head — geometry must carry the representation ==="
# Change from stage3: value head reduced to Linear(128→1)+Tanh.
# The 8k-parameter hidden layer (128→64→1) let the value head compensate
# for unstructured geometry. With a linear projection only, geometry has
# no shortcut — it must organise around what predicts value.
# No --init-model: stage3 weights have incompatible value head shape.
/zhome/81/b/206091/petra-env/bin/python3 src/train.py \
  --out models/geometry/stage4 \
  --epochs 80 \
  --patience 20 \
  --tight-patience 20 \
  --transition-drop 0.5 \
  --lr 1e-3 \
  --antipodal-weight 1.0 \
  --antipodal-margin 0.0 \
  --policy-weight 0.0 \
  --endgame-positions 16000 \
  --endgame-stages 1 2 3 4 5 6 7 8

echo "--- Geometry probe ---"
/zhome/81/b/206091/petra-env/bin/python3 src/test_geometry.py \
  --model models/geometry/stage4/best.pt

echo "--- Endgame conversion eval ---"
/zhome/81/b/206091/petra-env/bin/python3 src/eval_endgame.py \
  --model models/geometry/stage4/best.pt \
  --positions 100 \
  --n-sim 200 \
  --workers 8 \
  --stages 1 2
