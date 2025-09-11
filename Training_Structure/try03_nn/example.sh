#!/usr/bin/bash

#SBATCH -J train-run
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 1-0
#SBATCH -o /data/dydtkddhkdwk/PSID_LOWPRESSURE_NN/logs/slurm-%A.out
pwd
# python /data/dydtkddhkdwk/Activelearning202509/example/main.py


python /data/dydtkddhkdwk/PSID_LOWPRESSURE_NN/train_nn_gpu.py \
  --data /data/dydtkddhkdwk/PSID_LOWPRESSURE_NN/DataSet/He_313K/He_313K_He_313_0.01_to_He_313_5_dataset.csv \
  --outdir /data/dydtkddhkdwk/PSID_LOWPRESSURE_NN/RUN_OUT \
  --trial 2 \
 --sampler random_struct 
 #--sampler qt_then_rd 
 #--sampler random_with_input 
exit 0
#   --train-ratio 0.8 \
#   --qt-frac 0.4 \
#   --n-bins 10 \
#   --gamma 0.5 \
#   --hidden1 128 \
#   --hidden2 64 \
#   --lr 0.001 \
#   --batch-size 64 \
#   --epochs 200
