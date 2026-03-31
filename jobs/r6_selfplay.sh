#!/bin/sh
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "span[hosts=1] rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 4:00
#BSUB -J petra_r6_selfplay
#BSUB -w "done(petra_r6_bootstrap)"
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_r6_selfplay.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_r6_selfplay.err

cd /zhome/81/b/206091/Petra-Phase1/src
/zhome/81/b/206091/petra-env/bin/python3 selfplay.py \
  --model /zhome/81/b/206091/Petra-Phase1/models/zigzag/r6_base/best.pt \
  --games 500 \
  --n-sim 400 \
  --out /zhome/81/b/206091/Petra-Phase1/data/selfplay_r6.pt \
  --workers 16 \
  --opening-book /zhome/81/b/206091/Petra-Phase1/data/openings_r2.txt
