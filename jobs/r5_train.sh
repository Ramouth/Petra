#!/bin/sh
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 1:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -J petra_r5_train
#BSUB -w "done(petra_r5_selfplay)"
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_r5_train.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_r5_train.err

cd /zhome/81/b/206091/Petra-Phase1/src
/zhome/81/b/206091/petra-env/bin/python3 train.py \
  --dataset /zhome/81/b/206091/Petra-Phase1/data/selfplay_r5.pt \
  --out /zhome/81/b/206091/Petra-Phase1/models/zigzag/r5 \
  --lr 3e-4 --epochs 20 --patience 5 \
  --init-model /zhome/81/b/206091/Petra-Phase1/models/zigzag/r4/best.pt
