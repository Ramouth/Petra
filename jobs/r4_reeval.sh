#!/bin/sh
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 3:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -J petra_r4_reeval
#BSUB -w "done(petra_r4_selfplay)"
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_r4_reeval.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_r4_reeval.err

module load gcc/12.3.0-binutils-2.40

cd /zhome/81/b/206091/Petra-Phase1/src
/zhome/81/b/206091/petra-env/bin/python3 reeval_stockfish.py \
  --dataset /zhome/81/b/206091/Petra-Phase1/data/selfplay_r4.pt \
  --out /zhome/81/b/206091/Petra-Phase1/data/selfplay_r4_sf.pt \
  --depth 20 \
  --stockfish /zhome/81/b/206091/bin/stockfish
