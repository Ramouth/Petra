#!/bin/sh
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -W 8:00
#BSUB -J petra_r6_bootstrap
#BSUB -w "done(petra_r6_reeval)"
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_r6_bootstrap.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_r6_bootstrap.err

cd /zhome/81/b/206091/Petra-Phase1/src
/zhome/81/b/206091/petra-env/bin/python3 train.py \
  --dataset /zhome/81/b/206091/Petra-Phase1/data/selfplay_r6_sf.pt \
  --out /zhome/81/b/206091/Petra-Phase1/models/zigzag/r6_base \
  --lr 1e-3 --epochs 30 --patience 5
