#!/bin/bash

#SBATCH --job-name=clustering
#SBATCH --account=bsc72
#SBATCH --chdir=.
#SBATCH --output=logs/cluster/%j.out
#SBATCH --error=logs/cluster/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --qos=gp_bscls

module purge
ml bsc/1.0
ml anaconda

source activate ProtSeq2StrucAlpha

srun /home/bsc/bsc072876/.conda/envs/ProtSeq2StrucAlpha/bin/python clustering.py
