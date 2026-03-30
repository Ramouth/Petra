#!/bin/sh
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "span[hosts=1] rusage[mem=512MB]"
#BSUB -M 512MB
#BSUB -W 6:00
#BSUB -J petra_r3_selfplay
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_r3_selfplay.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_r3_selfplay.err

cd /zhome/81/b/206091/Petra-Phase1/src
/zhome/81/b/206091/petra-env/bin/python3 selfplay.py \
  --model /zhome/81/b/206091/Petra-Phase1/models/zigzag/r2/best.pt \
  --games 500 \
  --n-sim 160 \
  --out /zhome/81/b/206091/Petra-Phase1/data/selfplay_r3.pt \
  --workers 16
