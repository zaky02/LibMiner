import os
import json

os.makedirs("rep_runs", exist_ok=True)
slurm_template = """#!/bin/bash

#SBATCH --job-name=Enamine_clust_{batch_name}
#SBATCH --account=bsc72
#SBATCH --chdir=.
#SBATCH --output=logs/%j_{batch_name}.out
#SBATCH --error=logs/%j_{batch_name}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --time=10:00:00
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

time python -u clustering_partitions.py --batch_partitions {batch_path} --cores 15 --thr -85 --clustering representative -v True
"""

batch_dir = "/gpfs/projects/bsc72/Libraries4DSD/Enamine_Real_rep_batches/"

for batch_file in os.listdir(batch_dir):
    batch_path = os.path.join(batch_dir, batch_file)
    batch_name = os.path.splitext(batch_file)[0]
    run_file_path = os.path.join("rep_runs/", f"{batch_name}.sh")

    with open(run_file_path, "w") as run_file:
        run_file.write(slurm_template.format(batch_name=batch_name, batch_path=batch_path))
