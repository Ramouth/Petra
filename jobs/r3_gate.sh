#!/bin/sh
#BSUB -q hpc
#BSUB -n 4
#BSUB -W 2:00
#BSUB -R "span[hosts=1] rusage[mem=2GB]"
#BSUB -M 2GB
#BSUB -J petra_r3_gate
#BSUB -w "done(petra_r3_train)"
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_r3_gate.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_r3_gate.err

cd /zhome/81/b/206091/Petra-Phase1/src
/zhome/81/b/206091/petra-env/bin/python3 evaluate.py \
  --model /zhome/81/b/206091/Petra-Phase1/models/zigzag/r3/best.pt \
  --step 5 --games 100 --n-sim 160 --temp-moves 10 --workers 4
