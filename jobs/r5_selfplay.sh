#!/bin/sh
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 6:00
#BSUB -J petra_r5_selfplay
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_r5_selfplay.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_r5_selfplay.err

cd /zhome/81/b/206091/Petra-Phase1/src
/zhome/81/b/206091/petra-env/bin/python3 selfplay.py \
  --model /zhome/81/b/206091/Petra-Phase1/models/zigzag/r4/best.pt \
  --games 500 \
  --n-sim 400 \
  --out /zhome/81/b/206091/Petra-Phase1/data/selfplay_r5.pt \
  --workers 8 \
  --opening-book /zhome/81/b/206091/Petra-Phase1/data/openings_r2.txt
