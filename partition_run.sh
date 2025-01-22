#!/bin/bash

#SBATCH --job-name=db_partition
#SBATCH --account=bsc72
#SBATCH --chdir=.
#SBATCH --output=logs/part/%j.out
#SBATCH --error=logs/part/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --qos=acc_bscls

module purge
ml bsc/1.0
ml anaconda

source activate ProtSeq2StrucAlpha

srun /home/bsc/bsc072876/.conda/envs/ProtSeq2StrucAlpha/bin/python partitioner.py --input_dir ../ZINC20_dummy/ --output_dir ../ZINC20_partitioned/ --part_size 1000000 --pref ZINC20 --remove
