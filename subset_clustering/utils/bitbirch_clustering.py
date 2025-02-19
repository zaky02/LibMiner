import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity, CreateFromBitString
import numpy as np
from . import bitbirch as bb

def get_object_memory(obj):
    mem = sys.getsizeof(obj)
    mem = mem / (1024 ** 3)
    return mem

def generate_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024)
    fps = np.array([fpgen.GetFingerprint(mol) for mol in mols])
    del mols
    return fps

def get_bitbirch_clusters(df, file_id):

    print(f'Calculating fingerprints for {file_id}:')
    fps = generate_fingerprints(df.SMILES)
    mem_fps = get_object_memory(fps)
    print(f'Memory occupied by fingerprints from {file_id}: {mem_fps}')

    bitbirch = bb.BitBirch(branching_factor=50, threshold=0.65)
    bitbirch.fit(fps)

    centroids = bitbirch.get_centroids()

    cluster_list = bitbirch.get_cluster_mol_ids()
    print(f'Number of clusters for {file_id}: {len(cluster_list)}')

    print(f'Saving clusters for {file_id}:')
    n_molecules = len(fps)
    cluster_labels = [0] * n_molecules
    centroid_labels = [0] * n_molecules
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024)

    for cluster_id, indices in enumerate(cluster_list):
        for idx in indices:
            cluster_labels[idx] = cluster_id

        cluster_smiles = df.iloc[indices]['SMILES'].to_list()
        cluster_mols = [Chem.MolFromSmiles(smile) for smile in cluster_smiles]
        cluster_fps = [fpgen.GetFingerprint(mol) for mol in cluster_mols]

        centroid_fp = centroids[cluster_id]
        centroid_fp = ''.join(str(int(x)) for x in centroid_fp)
        centroid_fp = CreateFromBitString(centroid_fp)

        similarities = BulkTanimotoSimilarity(centroid_fp, cluster_fps)
        cent_idx = indices[np.argmax(similarities)]
        centroid_labels[cent_idx] = 1

    return centroid_labels, cluster_labels
