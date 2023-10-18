datas="FBDB15K FBYG15K"
rates="0.5 0.8"
for data in $datas ; do
  for rate in $rates ; do
      echo "sbatch -o ${data}_${rate}.log run.slurm 4 5 0.001 ${data} ${rate}"
      sbatch -o ${data}_${rate}_dev_4-1.log run.slurm 10 5 0.001 ${data} ${rate}
  done
done