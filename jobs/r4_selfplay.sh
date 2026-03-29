#!/bin/sh
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 4:00
#BSUB -J petra_r4_selfplay
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_r4_selfplay.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_r4_selfplay.err

cd /zhome/81/b/206091/Petra-Phase1/src
/zhome/81/b/206091/petra-env/bin/python3 selfplay.py \
  --model /zhome/81/b/206091/Petra-Phase1/models/zigzag/r2/best.pt \
  --games 200 \
  --n-sim 400 \
  --out /zhome/81/b/206091/Petra-Phase1/data/selfplay_r4.pt \
  --workers 8 \
  --opening-book /zhome/81/b/206091/Petra-Phase1/data/openings_r2.txt
