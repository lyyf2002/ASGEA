splits="OEA_D_W_15K_V1 OEA_D_W_15K_V2 OEA_EN_DE_15K_V1 OEA_EN_FR_15K_V1"
expn=$1
if [ ! -d "results/${expn}" ]; then
  mkdir results/${expn}
fi
if [ ! -d "results/${expn}/backup" ]; then
  mkdir results/${expn}/backup
fi
cp *.py results/${expn}/backup/

for split in $splits ; do
  python train.py --n_batch 4 --n_layer 7 --lr 0.001 --data_choice OpenEA --data_rate 0.2 --img_dim 512 --topk 0 --data_split ${split} --exp_name ${expn} --mm 1
  # echo "sbatch ${split}"
  # sbatch -o OEA_${expn}_${split}.log run_oea.slurm 4 7 0.001 ${split} ${expn}
done
