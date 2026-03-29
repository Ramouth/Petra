#!/bin/sh
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 1:00
#BSUB -J petra_r6_train
#BSUB -w "done(petra_r6_selfplay)"
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_r6_train.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_r6_train.err

cd /zhome/81/b/206091/Petra-Phase1/src
/zhome/81/b/206091/petra-env/bin/python3 train.py \
  --dataset /zhome/81/b/206091/Petra-Phase1/data/selfplay_r6.pt \
  --out /zhome/81/b/206091/Petra-Phase1/models/zigzag/r6 \
  --lr 3e-4 --epochs 20 --patience 5 \
  --init-model /zhome/81/b/206091/Petra-Phase1/models/zigzag/r6_base/best.pt
