#!/bin/bash
#SBATCH -J MASEA
#SBATCH -p sugon
#SBATCH --gres=gpu:V100:1
python train.py --n_batch $1 --n_layer $2 --lr $3