#!/bin/bash

#SBATCH --job-name=ZINC_clustering1
#SBATCH --account=bsc72
#SBATCH --chdir=.
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=48:00:00
#SBATCH --qos=gp_bscls

module purge
ml intel/2023.2.0
ml cmake/3.25.1
ml impi/2021.10.0
ml mkl/2023.2.0
ml miniconda/24.1.2
ml anaconda

eval "$(conda shell.bash hook)"
source activate /gpfs/projects/bsc72/conda_envs/MolecularAnalysis

time python -u clustering_partitions.py --batch_partitions /gpfs/projects/bsc72/Libraries4DSD/ZINC20_batch_1 --cores 28 > ZINC_part_clustering.out 2>&1
