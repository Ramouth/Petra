#!/bin/sh
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 4:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -J petra_r3_train
#BSUB -w "done(petra_r3_reeval)" 
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_r3_train.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_r3_train.err

cd /zhome/81/b/206091/Petra-Phase1/src
/zhome/81/b/206091/petra-env/bin/python3 train.py \
  --dataset /zhome/81/b/206091/Petra-Phase1/data/selfplay_r3_sf.pt \
  --out /zhome/81/b/206091/Petra-Phase1/models/zigzag/r3 \
  --lr 1e-4 --epochs 15 --patience 3 \
  --init-model /zhome/81/b/206091/Petra-Phase1/models/zigzag/r2/best.pt
