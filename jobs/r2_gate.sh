#!/bin/sh
#BSUB -q hpc
#BSUB -n 4
#BSUB -W 4:00
#BSUB -R "span[hosts=1] rusage[mem=1GB]"
#BSUB -M 1GB
#BSUB -J petra_r2_gate
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_r2_gate.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_r2_gate.err

cd /zhome/81/b/206091/Petra-Phase1/src
/zhome/81/b/206091/Petra-Phase1/../petra-env/bin/python3 evaluate.py \
  --model /zhome/81/b/206091/Petra-Phase1/models/zigzag/r2/best.pt \
  --step 5 --games 100 --n-sim 80 --temp-moves 10 --workers 4
