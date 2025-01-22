#!/bin/bash

#SBATCH --account=bsc72
#SBATCH --qos=gp_debug
#SBATCH --job-name=$PARTITION
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

ml intel/2023.2.0
ml cmake/3.25.1
ml impi/2021.10.0
ml mkl/2023.2.0
ml miniconda/24.1.2

eval "$(conda shell.bash hook)"
source activate /gpfs/projects/bsc72/conda_envs/MolecularAnalysis

gunzip $DATAPATH/$COMPRESS
python /gpfs/projects/bsc72/MolDBs/scripts/parse_PubChem.py $DATAPATH/$UNCOMPRESS
rm $DATAPATH/$UNCOMPRESS
