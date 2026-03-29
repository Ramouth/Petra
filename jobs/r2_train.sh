#!/bin/sh
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 4:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -J petra_r2_train
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_r2_train.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_r2_train.err

cd /zhome/81/b/206091/Petra-Phase1/src
/zhome/81/b/206091/Petra-Phase1/../petra-env/bin/python3 train.py \
  --dataset /zhome/81/b/206091/Petra-Phase1/data/selfplay_r2_sf.pt \
  --out /zhome/81/b/206091/Petra-Phase1/models/zigzag/r2 \
  --lr 3e-4 --epochs 15 --patience 3 \
  --init-model /zhome/81/b/206091/Petra-Phase1/models/zigzag/r1/best.pt
