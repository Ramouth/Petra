#!/bin/sh
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=2GB]"
#BSUB -M 2GB
#BSUB -W 2:00
#BSUB -J petra_stage1_eval
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_stage1_eval.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_stage1_eval.err

cd /zhome/81/b/206091/Petra-Phase1/src

echo "=== Step 6: MCTS(geometry) vs MCTS(material) — Stage 1 model ==="
/zhome/81/b/206091/petra-env/bin/python3 evaluate.py \
  --model /zhome/81/b/206091/Petra-Phase1/models/geometry/stage1/best.pt \
  --games 100 \
  --step 6 \
  --n-sim 200 \
  --workers 8
