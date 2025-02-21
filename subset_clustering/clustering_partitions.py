import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
from utils.bitbirch_clustering import get_bitbirch_clusters

# 1M molecules partitions occupy about 8 GB of fps
# we recomend using no more than 28 cores
# Find new fps calculating method

def process_batch(batch_partitions, num_workers, verbose):
    """
    Process a single batch file in parallel.
    """
    files = np.genfromtxt(batch_partitions, dtype=str)
    process_file_args = [(file, verbose) for file in files] 
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

def process_file(file, verbose):
    process_id = os.getpid()
    try:
        print(f'[PROCESS {process_id}] Processing file: {file}')
        df = pd.read_csv(file)
        file_id = os.path.basename(file)
        
        print(f'[PROCESS {process_id}] Clustering file: {file}')
        birch_representative_labels, birch_cluster_labels = get_bitbirch_clusters(df, file_id, verbose)
        
        print(f'[PROCESS {process_id}] Adding cluster ID and representative to: {file}')
        df['partition_cluster'] = birch_cluster_labels
        df['partition_representative'] = birch_representative_labels
        df.to_csv(file, index=False)
    
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
                                   required=True)
    parser.add_argument('-v',
                        dest='verbose',
                        help='verbosity to debug',
                        default=False)

    args = parser.parse_args()
    batch_partitions = args.batch_partitions
    cores = args.cores
    verbose = args.verbose

    process_batch(batch_partitions, cores, verbose)
