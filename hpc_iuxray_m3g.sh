#!/bin/bash
#SBATCH --job-name=iu_train_a2i2
#SBATCH --account=bw83
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16384
module load cuda
module load anaconda/2019.03-Python3.7-gcc5
source activate /projects/bw83/dzha0062/conda_envs/lzh
cd /home/dzha0062/bw83_scratch/donghao/R2Gen
bash python_iuxray_m3g.sh 2>&1 | tee output.txt