splits="zh_en ja_en fr_en"
expn=$1
if [ ! -d "results/${expn}" ]; then
  mkdir results/${expn}
fi
if [ ! -d "results/${expn}/backup" ]; then
  mkdir results/${expn}/backup
fi
cp *.py results/${expn}/backup/

for split in $splits ; do
  python train.py --n_batch 4 --n_layer 7 --lr 0.001 --data_split ${split} --exp_name ${expn} --mm 1 --MLP_num_layers 3 --topk 100 --data_rate 0.3 --data_choice DBP15K --img_dim 2048 --use_img_ill
  # echo "sbatch ${split}"
  # sbatch -o DBP_${expn}_${split}.log run_dbp.slurm 4 7 0.001 ${split} ${expn}
done
