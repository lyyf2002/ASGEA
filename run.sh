datas="FBDB15K FBYG15K"
rates="0.2 0.5"
for data in $datas ; do
  for rate in $rates ; do
      echo "sbatch -o ${data}_${rate}.log run.slurm 4 5 0.001 ${data} ${rate}"
      sbatch -o ${data}_${rate}.log run.slurm 4 5 0.001 ${data} ${rate}
  done
done