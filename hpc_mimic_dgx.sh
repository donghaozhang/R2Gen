#!/bin/bash
#SBATCH --job-name=mimic_r2gen_dgx
#SBATCH --account=bw84
#SBATCH --gres=gpu:V100:8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --constraint=V100-32G
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=dgx
#SBATCH --qos=dgx
module load cuda
module load anaconda/2019.03-Python3.7-gcc5
source activate /projects/bw83/dzha0062/conda_envs/R2Genv3
cd /home/dzha0062/bw83_scratch/donghao/R2Gen
bash python_mimic_dgx.sh 2>&1 | tee output.txt