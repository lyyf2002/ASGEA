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
      echo "sbatch -o ${data}_${rate}.log run.slurm 4 5 0.001 ${data} ${rate}"
      sbatch -o ${expn}_${data}_${rate}.log run.slurm 4 5 0.001 ${data} ${rate} ${expn}
  done
done