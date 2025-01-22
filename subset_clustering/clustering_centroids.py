import os
import sys
import pickle
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit.DataStructs import TanimotoSimilarity
import numpy as np
import bitbirch as bb
import scipy as sp
from multiprocessing import Pool

def centroid_clustering(batch_file):
    """
    OBJECTIVE:
    - READ THE PKL DICTIONARY FILE
    - CLUSTER USING BITBIRCH THE CENTROID FINGERPRINTS (VALUES)
    - TRACEBACK THE DICTIONARY KEYS SO THAT AFTER CLUSTERING THE CENTROIDS
      THE FINAL CLUSTER ID CAN BE ADDED AS A NEW COLUMN TO ALL MOLECULES 
      THAT HAD THAT CENTROID IN THEIR INITIAL CLUSTER
    - WE ALSO HAVE TO RETRIEVE THE ACTUAL CENTROIDS FROM THE CLUSTERS BY
      APPLYING THE MAX_SEPARATION CODE OF BITBIRCH BUT RETRIEVING THE MOST 
      SIMILAR MOLECULE TO THE CENTROID
    """
    files = np.genfromtxt(batch_file, dtype=str)
    total_fps = []
    total_ids = []
    for file in files:
        partition = file.replace('.pkl','')
        with open(file, 'rb') as f:
            part_fps = pickle.load(f)
        for cluster, fp in part_fps.items():
            total_fps.append(fp)
            total_ids.append((partition, cluster))
     
    total_fps = np.array(total_fps)
    print(f'Total number of centroids to cluster: {len(total_fps)}')

    centroids_dict, clusters_labels = get_bitbirch_centroid_clusters(total_fps)
    #print(centroids_dict)
    #print(clusters_labels)

    print(f'Saving the new centroids of the clustering of'
            'centroids as centroids_clustering.pkl')
    path = os.path.dirname(batch_file)
    with open(path + '/centroids_clustering.pkl','wb') as f:
        pickle.dump(centroids_dict, f)

    print(f'Adding the final cluster id to molecules on each partition')
    for file in files:
        print(file.replace('.pkl','.csv'))

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
    requiredArguments.add_argument('--batch_file',
                                   help='File containing the pickle files each with the centroid fingerprints of its corresponding partition clustering',
                                   required=True)
    
    args = parser.parse_args()

    batch_file = args.batch_file

    centroid_clustering(batch_file)
