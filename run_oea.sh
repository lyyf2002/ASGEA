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
  echo "sbatch ${split}"
  sbatch -o OEA_${expn}_${split}.log run_oea.slurm 4 5 0.001 ${split} ${expn}
done
