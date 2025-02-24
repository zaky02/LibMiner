import sys
import time
import scipy
import numpy as np
import pandas as pd
from .timer import Timer
from . import bitbirch as bb
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity, CreateFromBitString, ExplicitBitVect

def get_object_memory(obj):
    mem = sys.getsizeof(obj)
    mem = mem / (1024 ** 3)
    return mem

def generate_fingerprints(smiles_list, file_id, verbose):
    
    fps = []
    for i, smile in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smile)
        fp = Chem.RDKFingerprint(mol)
        fps.append(fp) 
        if i % 5000 == 0 and i != 0 and verbose:
            print(f'{i}/{len(smiles_list)} calc. fps for {file_id}')    
    
    if verbose:
        print('Fingerprint calculation ended!')
        print('Transforming fingerprints to sparse matrix')
    
    fps_sparse = scipy.sparse.csr_matrix(fps)
    
    del fps
    return fps_sparse

def get_bitbirch_clusters(df, file_id, verbose):
    
    timer_fps = Timer(autoreset=True)
    timer_fps.start(f'Calculating fingerprints for {file_id}:')
    fps = generate_fingerprints(df.SMILES, file_id, verbose)
    timer_fps.stop()
    mem_fps = get_object_memory(fps)
    
    if verbose:
        print(f'Memory occupied by fingerprints from {file_id}: {mem_fps}')
    
    bitbirch = bb.BitBirch(branching_factor=50, threshold=0.65)
    bitbirch.fit(fps)

    centroids = bitbirch.get_centroids()

    cluster_list = bitbirch.get_cluster_mol_ids()
    print(f'Number of clusters for {file_id}: {len(cluster_list)}')

    print(f'Saving clusters for {file_id}:')
    n_molecules = fps.shape[0]
    cluster_labels = [0] * n_molecules
    representative_labels = [0] * n_molecules

    for cluster_id, indices in enumerate(cluster_list):
        for idx in indices:
            cluster_labels[idx] = cluster_id

        # Retrieving cluster fingerprints
        cluster_fps = [fps.getrow(idx).toarray().squeeze() for idx in indices]
        cluster_fps = [''.join(str(int(x)) for x in fp) for fp in cluster_fps]
        cluster_fps = [CreateFromBitString(fp) for fp in cluster_fps]

        # Retrieveing mathematical centroid fingerprint
        centroid_fp = centroids[cluster_id]
        centroid_fp = ''.join(str(int(x)) for x in centroid_fp)
        centroid_fp = CreateFromBitString(centroid_fp)

        similarities = BulkTanimotoSimilarity(centroid_fp, cluster_fps)
        
        if verbose and len(indices)>5:
            print(f'{file_id} cluster {cluster_id} with {len(similarities)} elements, its representative has a similarity of {np.max(similarities)} with its centroid')
        
        cent_idx = indices[np.argmax(similarities)]
        representative_labels[cent_idx] = 1

    return representative_labels, cluster_labels
