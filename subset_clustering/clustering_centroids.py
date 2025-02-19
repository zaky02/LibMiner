import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool
from utils.bitbirch_clustering import get_bitbirch_clusters

# Check if we are storing the values correctly in the original csv files

def process_file(file):
    df = pd.read_csv(file)
    centroid_df = df[df['partition_centroid'] == 1].copy()
    centroid_df['source'] = file
    return centroid_df

def update_file(file, centroid_df):

    print(f'Updating {file} with new centroid clusters and centroids:')
    df = pd.read_csv(file)

    file_centroids = centroid_df[centroid_df['source'] == file]
    df = pd.merge(df, file_centroids, on=['ID', 'SMILES', 'dockscore', 'partition_cluster', 'partition_centroid'])

    print(f'Mapping centroid cluster and centroid for file: {file}')
    for cluster_id in file_centroids['centroid_cluster'].unique():
        if pd.notna(cluster_id):
            cluster_rows = file_centroids[file_centroids['centroid_cluster'] == cluster_id]
            df.loc[(df['partition_cluster'].isin(cluster_rows['partition_cluster'])) &
                    (df['partition_centroid'] == 0), 'centroid_cluster'] = int(cluster_id)
            df.loc[(df['partition_cluster'].isin(cluster_rows['partition_cluster'])) &
                    (df['partition_centroid'] == 0), 'centroid_centroid'] = int(0)
    
    if 'source' in df.columns:
        df = df.drop(columns=['source'])

    df.to_csv(file, index=False)
    print(f'Updating for file: {file}, complete!')

    return file

def centroid_clustering(batch_partitions, num_workers):
    
    files = np.genfromtxt(batch_partitions, dtype=str)
    
    with Pool(num_workers) as pool:
        centroid_dfs = pool.map(process_file, files)

    centroid_df = pd.concat(centroid_dfs, ignore_index=True)
    file_id = 'centroid dataframe'

    print(f'Total number of centroids to cluster: {len(centroid_df)}')
    birch_centroid_labels, birch_cluster_labels = get_bitbirch_clusters(centroid_df, file_id)

    print(f'Adding cluster ID and centroid to: {file_id}')
    centroid_df['centroid_cluster'] = birch_cluster_labels
    centroid_df['centroid_centroid'] = birch_centroid_labels
    centroid_df['centroid_cluster'] = centroid_df['centroid_cluster'].fillna(-100)
    centroid_df['centroid_centroid'] = centroid_df['centroid_centroid'].fillna(-100)
    centroid_df['centroid_cluster'] = np.array(centroid_df['centroid_cluster']).astype(int)
    centroid_df['centroid_centroid'] = np.array(centroid_df['centroid_centroid']).astype(int)

    with Pool(num_workers) as pool:
        update = partial(update_file, centroid_df=centroid_df)
        pool.map(update, files)


if __name__ == '__main__': 
    import argparse

    parser = argparse.ArgumentParser(description='Script that clusters the centroids of the initial partition clustering.')

    requiredArguments = parser.add_argument_group('Required Arguments')
    requiredArguments.add_argument('--batch_partitions',
                                   help='Batch file with the paths to the csv files with the real centroid of its corresponding partition clustering',
                                   required=True)
    requiredArguments.add_argument('--cores',
                                   type=int,
                                   help='Number of CPUs to parallelise on',
                                   required=True)
    
    args = parser.parse_args()
    
    # Load arguments
    batch_partitions = args.batch_partitions
    cores = args.cores

    centroid_clustering(batch_partitions, cores)
