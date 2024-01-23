#!/bin/bash
#SBATCH -J MASEA
#SBATCH -p sugon
#SBATCH --gres=gpu:V100:1
python train.py --n_batch $1 --n_layer $2 --lr $3 --data_split $4 --exp_name $5 --mm 1 --MLP_num_layers 3 --topk 100 --data_rate 0.3 --data_choice DBP15K --img_dim 2048 --use_img_ill