import os
import sys
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit.DataStructs import TanimotoSimilarity
import numpy as np
import bitbirch as bb
import scipy as sp
from multiprocessing import Pool


def process_batch(batch_partitions, num_workers):
    """
    Process a single batch file in parallel.
    """
    files = np.genfromtxt(batch_partitions, dtype=str)
    with Pool(num_workers) as pool:
        pool.map(process_file, files)

def process_file(file):
    process_id = os.getpid()
    print(f'[PROCESS {process_id}] Processing file: {file}')
    df = pd.read_csv(file)
    file_id = os.path.basename(file)
    centroids_dict, birch_cluster_labels = get_bitbirch_clusters(df, file_id)

    with open(file.replace('.csv', '.pkl'), 'wb') as f:
        pickle.dump(centroids_dict, f)

    df['partition_cluster'] = birch_cluster_labels
    df.to_csv(file, index=False)

def get_object_memory(obj):
    mem = sys.getsizeof(obj)
    mem = mem / (1024 ** 3)
    return mem

def generate_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fps = np.array([Chem.RDKFingerprint(mol, fpSize=1024) for mol in mols])
    del mols
    return fps

def get_bitbirch_clusters(df, file_id):
    
    print(f'Calculating fingerprints for {file_id}:')
    fps = generate_fingerprints(df.SMILES)
    mem_fps = get_object_memory(fps)
    print(f'Memory occupied by fingerprints from {file_id}: {mem_fps}')

    bitbirch = bb.BitBirch(branching_factor=50, threshold=0.65)
    print(f'Calculating clusters for {file_id}:')
    bitbirch.fit(fps)
    
    centroids = bitbirch.get_centroids()
    centroids_dict = {}
    for i, centroid in enumerate(centroids):
        centroids_dict[i] = centroid

    cluster_list = bitbirch.get_cluster_mol_ids()
    print(f'Number of clusters for {file_id}: {len(cluster_list)}')

    print(f'Saving clusters for {file_id}:')
    n_molecules = len(fps)
    cluster_labels = [0] * n_molecules
    for cluster_id, indices in enumerate(cluster_list):
        for idx in indices:
            cluster_labels[idx] = cluster_id
    
    return centroids_dict, cluster_labels

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
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
