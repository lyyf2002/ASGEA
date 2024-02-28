#!/bin/bash
#SBATCH -J MMKG
#SBATCH -p sugon
#SBATCH --gres=gpu:V100:1
python train.py --data_split norm --n_batch $1 --n_layer $2 --lr $3 --data_choice $4 --data_rate $5 --exp_name $6 --mm 1 --img_dim 4096