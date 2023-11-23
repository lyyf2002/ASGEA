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
  echo "sbatch ${split}"
  sbatch -o DBP_${expn}_${split}.log run_dbp.slurm 2 5 0.001 ${split} ${expn}
done
