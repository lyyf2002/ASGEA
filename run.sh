datas="FBDB15K FBYG15K"
rates="0.2 0.5 0.8"
expn=$1
if [ ! -d "results/${expn}" ]; then
  mkdir results/${expn}
fi
if [ ! -d "results/${expn}/backup" ]; then
  mkdir results/${expn}/backup
fi
cp *.py results/${expn}/backup/
for data in $datas ; do
  for rate in $rates ; do
    python train.py --data_split norm --n_batch 4 --n_layer 5 --lr 0.001 --data_choice ${data} --data_rate ${rate} --exp_name ${expn} --mm 1 --img_dim 4096
    # echo "sbatch -o ${data}_${rate}.log run.slurm 4 5 0.001 ${data} ${rate}"
    # sbatch -o ${expn}_${data}_${rate}.log run.slurm 4 5 0.001 ${data} ${rate} ${expn}
  done
done