import os
import psutil
import pandas as pd
import numpy as np
from multiprocessing import Pool, current_process
from utils.memory import get_CPU_memory
from utils.bitbirch_clustering import get_bitbirch_clusters

# 1M molecules partitions occupy about 8 GB of fps
# we recomend using no more than 28 cores

def process_batch(batch_partitions, num_workers, fpsize, bf, sim, clustering, verbose):
    """
    Process a single batch file in parallel.
    """
    files = np.genfromtxt(batch_partitions, dtype=str)

    # Make sure we are processing the batch file correctly in order to
    # handle cases where the batch contains only one file
    if isinstance(files, np.ndarray) and files.ndim == 0:
        files = np.array([files])
    elif isinstance(files, str):
        files = [files]

    process_file_args = [(file, fpsize, bf, sim, clustering, verbose) for file in files] 
    with Pool(num_workers) as pool:
        result = pool.starmap_async(process_file, process_file_args)
        
        # Wait for all processes to complete and handle any exceptions
        try:
            result.get()  # This will raise any exception encountered by a worker
        except Exception as e:
            print(f"Error encountered: {e}. Stopping all execution.")
            pool.terminate()  # Terminate all processes immediately
            pool.join()  # Ensure that all processes are cleaned up
            raise  # Reraise the exception to stop execution

def is_print_process():
    return current_process().name == "ForkPoolWorker-1"

def process_file(file, fpsize, bf, sim, clustering, verbose):
    process_id = os.getpid()
    process = psutil.Process(process_id)
    try:

        if verbose:
            print(f"[PROCESS {process_id}] {process.memory_info().rss / (1024 ** 2):.2f} MB of memory used at the start")
            if is_print_process():
                get_CPU_memory()
        
        print(f'[PROCESS {process_id}] Processing file: {file}')
        df = pd.read_csv(file)
        file_id = os.path.basename(file)
        
        if verbose:
            print(f"[PROCESS {process_id}] {process.memory_info().rss / (1024 ** 2):.2f} MB of memory used after loading csv ({file_id})")
            if is_print_process():
                get_CPU_memory()
        
        print(f'[PROCESS {process_id}] Clustering ({file})')
        birch_representative_labels, birch_cluster_labels = get_bitbirch_clusters(df, file_id, fpsize, bf, sim, verbose)
        
        print(f'[PROCESS {process_id}] Adding cluster ID and representative ({file})')
        df[f'{clustering}_cluster'] = birch_cluster_labels
        df[f'{clustering}_representative'] = birch_representative_labels
        df.to_csv(file, index=False)

        print(f'[PROCESS {process_id}] Finished clustering the partition ({file})')
    
    except Exception as e:
        print(f"[PROCESS {process_id}] Error processing {file}: {str(e)}")
        raise  # Reraise the error to propagate it back and stop the pool

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Given a batch of molecular database partitions, clusterize each partition individually. Notably, 1 million fingerprints occupy approximately 8GB of memory. Therefore, using 28 cores (recommended) requires a total of 224GB.')
    requiredArguments = parser.add_argument_group('Required Arguments')
    requiredArguments.add_argument('--batch_partitions',
                                   help='Batch file with the paths to the DB partitions',
                                   required=True)
    requiredArguments.add_argument('--cores',
                                   help='',
                                   type=int,
                                   default=15,
                                   required=True)
    parser.add_argument('-v',
                        dest='verbose',
                        help='verbosity to debug',
                        default=False,
                        type=bool)
    parser.add_argument('--fpsize',
                        dest='fpsize',
                        help='Size of the fingerprint bit vector',
                        default=2048,
                        type=int)
    parser.add_argument('--branch_factor',
                        help='BIRCH branching factor parameter',
                        default=50)
    parser.add_argument('--thr',
                        help='',
                        default=-105)
    parser.add_argument('--clustering',
                        help='This is the clustering stage which can be: partition, representative or final',
                        type=str)

    args = parser.parse_args()
    batch_partitions = args.batch_partitions
    cores = args.cores
    verbose = args.verbose
    fpsize = args.fpsize
    bf = args.branch_factor
    thr = args.thr
    clustering = args.clustering

    process_batch(batch_partitions, cores, fpsize, bf, thr, clustering, verbose)
