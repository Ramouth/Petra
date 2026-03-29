#!/bin/sh
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 2:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -J petra_r2_reeval
#BSUB -o /zhome/81/b/206091/Petra-Phase1/logs/lsf_r2_reeval.log
#BSUB -e /zhome/81/b/206091/Petra-Phase1/logs/lsf_r2_reeval.err

module load gcc/12.3.0-binutils-2.40

cd /zhome/81/b/206091/Petra-Phase1/src
/zhome/81/b/206091/petra-env/bin/python3 reeval_stockfish.py \
  --dataset /zhome/81/b/206091/Petra-Phase1/data/selfplay_r2.pt \
  --out /zhome/81/b/206091/Petra-Phase1/data/selfplay_r2_sf.pt \
  --depth 15 \
  --stockfish /zhome/81/b/206091/bin/stockfish
