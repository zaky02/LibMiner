import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
from utils.bitbirch_clustering import get_bitbirch_clusters

# 1M molecules partitions occupy about 8 GB of fps
# we recomend using no more than 28 cores

def process_batch(batch_partitions, num_workers):
    """
    Process a single batch file in parallel.
    """
    files = np.genfromtxt(batch_partitions, dtype=str)
    print(files)
    with Pool(num_workers) as pool:
        pool.map(process_file, files)

def process_file(file):
    process_id = os.getpid()
    print(f'[PROCESS {process_id}] Processing file: {file}')
    df = pd.read_csv(file)
    file_id = os.path.basename(file)
    print(f'[PROCESS {process_id}] Clustering file: {file}')
    birch_centroid_labels, birch_cluster_labels = get_bitbirch_clusters(df, file_id)
    
    print(f'[PROCESS {process_id}] Adding cluster ID and centroid to: {file}')
    df['partition_cluster'] = birch_cluster_labels
    df['partition_centroid'] = birch_centroid_labels
    df.to_csv(file, index=False)


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
    args = parser.parse_args()
    batch_partitions = args.batch_partitions
    cores = args.cores

    process_batch(batch_partitions, cores)
