import os
import sys
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.DataStructs.cDataStructs import CreateFromBitString
import numpy as np
import bitbirch as bb
import scipy as sp
from multiprocessing import Pool
import pandas as pd

def centroid_clustering(batch_centroids, outname):
    
    files = np.genfromtxt(batch_centroids, dtype=str)
    
    total_fps = []
    total_ids = []
    for file in files:
        partition = os.path.basename(file.replace('.pkl',''))
        
        with open(file, 'rb') as f:
            part_fps = pickle.load(f)
        
        for cluster, fp in part_fps.items():
            total_fps.append(fp)
            total_ids.append((partition, cluster))
     
    total_fps = np.array(total_fps)
    print(f'Total number of centroids to cluster: {len(total_fps)}')

    centroids_dict, clusters_labels = get_bitbirch_centroid_clusters(total_fps)

    # Get rdkit centroids_dict
    rdkit_centroid_dict = {}
    for ccluster, np_fp in centroids_dict.items():
        bitstring_fp = ''.join(map(str, np_fp))
        rdkit_fp = CreateFromBitString(bitstring_fp) 
        rdkit_centroid_dict[ccluster] = rdkit_fp
    
    path = os.path.dirname(batch_centroids) + '/'
    print(f'Saving the new centroids of the centroid clustering'
            f' to {path}{outname}')
    with open(path + outname + '_rdkit.pkl','wb') as f:
        pickle.dump(rdkit_centroid_dict, f)
    
    # Get a dictionary of dictionaries with [partition][p_clusterid] = [c_clusterid]
    pcluster_to_ccluster = {}
    for i, ccluster in enumerate(clusters_labels):
        partition, pcluster = total_ids[i]
        if partition not in pcluster_to_ccluster:
            pcluster_to_ccluster[partition] = {}
            pcluster_to_ccluster[partition][pcluster] = ccluster
        else:
            pcluster_to_ccluster[partition][pcluster] = ccluster

    print(f'Adding the centroid cluster id to molecules on each partition')
    for file in files:
        partition = os.path.basename(file.replace('.pkl',''))
        df = pd.read_csv(file.replace('.pkl','.csv'))
        pclusters = df['partition_cluster'].to_list()
        cclusters = []
        for pcluster in pclusters:
            ccluster = pcluster_to_ccluster[partition][pcluster]
            cclusters.append(ccluster)
        df['centroid_cluster'] = cclusters
        df.to_csv(file.replace('.pkl','.csv')) 

def get_object_memory(obj):
    mem = sys.getsizeof(obj)
    mem = mem / (1024 ** 3)
    return mem

def get_bitbirch_centroid_clusters(fps):
    
    mem_fps = get_object_memory(fps)
    print(f'Memory occupied by centroid fingerprints: {mem_fps}')

    bitbirch = bb.BitBirch(branching_factor=50, threshold=0.65)
    print(f'Calculating clusters of centroids:')
    bitbirch.fit(fps)

    centroids = bitbirch.get_centroids()
    centroids_dict = {}
    for i, centroid in enumerate(centroids):
        centroids_dict[i] = centroid

    cluster_list = bitbirch.get_cluster_mol_ids()
    print(f'Number of clusters of centroids: {len(cluster_list)}')

    print(f'Saving clusters:')
    n_centroids = len(fps)
    cluster_labels = [0] * n_centroids
    for cluster_id, indices in enumerate(cluster_list):
        for idx in indices:
            cluster_labels[idx] = cluster_id

    return centroids_dict, cluster_labels

if __name__ == '__main__': 
    import argparse

    parser = argparse.ArgumentParser(description='Script that clusters the centroid fingerprints stored in a pkl file.')

    requiredArguments = parser.add_argument_group('Required Arguments')
    requiredArguments.add_argument('--batch_centroids',
                                   help='Batch file with the paths to the pickle files with the centroid fingerprints of its corresponding partition clustering',
                                   required=True)
    requiredArguments.add_argument('--outname',
                                   help='Outname for pkl file of centroids of the centroid clustering',
                                   required=True)
    
    args = parser.parse_args()

    batch_centroids = args.batch_centroids
    outname = args.outname

    centroid_clustering(batch_centroids, outname)
