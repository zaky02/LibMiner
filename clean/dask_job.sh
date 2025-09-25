#!/bin/bash
#SBATCH --job-name=dask-test
# cada task es un worker al parecer
#SBATCH --ntasks=8
# este es el numero de threads or cores de cpus que el worker usará
# cada core tiene 2 GB (entonces 5*2=10GB de memoria máxima que puedes poner en --memory-limit)
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH --qos=gp_debug
#SBATCH --output=dask-%j.out
#SBATCH --error=dask-%j.err
#SBATCH --account=bsc72

module load anaconda   # or load your env module
source activate ai_factory     # if using conda/mamba

TMPDIR=/scratch/tmp/${USER}/${SLURM_JOBID}_dask
mkdir -p $TMPDIR
# Get the hostname of the first node in the allocation
SCHEDULER_HOST=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)

# Start scheduler in background on the first node
if [[ $SLURM_PROCID == 0 ]]; then
    dask-scheduler --host ${SCHEDULER_HOST} --port 8786 --no-dashboard &
    sleep 10  # give scheduler time to start
fi

# Start workers on all nodes
srun dask-worker tcp://${SCHEDULER_HOST}:8786 --nthreads 2 --local-directory $TMPDIR &

# Run your Python script, passing scheduler address as env variable
export DASK_SCHEDULER_ADDRESS="tcp://${SCHEDULER_HOST}:8786"
conda run -n ai_factory python deduplicate.py

rm -rf $TMPDIR