#!/bin/bash
#SBATCH -J MASEA
#SBATCH -p sugon
#SBATCH --gres=gpu:V100:1
python train.py --n_batch $1 --n_layer $2 --lr $3 --data_choice OpenEA --data_rate 0.2 --img_dim 512 --topk 0 --data_split $4 --exp_name $5 --mm 1 --MLP_num_layers 1