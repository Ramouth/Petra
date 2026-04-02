#!/bin/sh
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "span[hosts=1] rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 2:00
#BSUB -J petra_stage4_eval
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_stage4_eval.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_stage4_eval.err

cd /zhome/81/b/206091/Petra-Phase1

echo "--- Geometry probe ---"
/zhome/81/b/206091/petra-env/bin/python3 src/test_geometry.py \
  --model models/geometry/stage4/best.pt

echo "--- Endgame conversion eval ---"
/zhome/81/b/206091/petra-env/bin/python3 src/eval_endgame.py \
  --model models/geometry/stage4/best.pt \
  --positions 100 \
  --n-sim 200 \
  --workers 16 \
  --stages 1 2
