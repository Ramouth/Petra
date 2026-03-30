#!/bin/sh
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=2GB]"
#BSUB -M 2GB
#BSUB -W 4:00
#BSUB -J petra_r6_gate
#BSUB -w "done(petra_r6_train)"
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_r6_gate.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_r6_gate.err

cd /zhome/81/b/206091/Petra-Phase1/src
/zhome/81/b/206091/petra-env/bin/python3 evaluate.py \
  --model /zhome/81/b/206091/Petra-Phase1/models/zigzag/r6/best.pt \
  --baseline-model /zhome/81/b/206091/Petra-Phase1/models/zigzag/r4/best.pt \
  --step 5 --games 100 --n-sim 400 --temp-moves 10 --workers 8 \
  --pgn-out /zhome/81/b/206091/Petra-Phase1/logs/lsf_r6_gate.pgn
