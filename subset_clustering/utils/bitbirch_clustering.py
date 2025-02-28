import gc
import os
import sys
import time
import scipy
import psutil
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from pympler import asizeof
from multiprocessing import Pool, current_process
from rdkit.Chem import rdFingerprintGenerator, AllChem, MolFromSmiles
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity, CreateFromBitString, ExplicitBitVect
from .timer import Timer
from . import bitbirch as bb
from .memory import get_CPU_memory

def get_object_memory(obj):
    mem = asizeof.asizeof(obj)
    mem = mem / (1024 ** 2)
    return mem

def is_print_process():
    return current_process().name == "ForkPoolWorker-1"

def fold_fingerprint(fp, fpsize):
    """
    """
    if fpsize % 2 != 0:
        raise ValueError("Fingerprint length must be even for equal folding.")

    half_n = fpsize // 2
    first_half = fp[:half_n]
    second_half = fp[half_n:]

    folded_fp = np.logical_or(first_half, second_half).astype(np.uint8)

    del half_n, first_half, second_half

    return folded_fp

def generate_fingerprints(smiles_list, file_id, fpsize, verbose):
    """
    """
    process_id = os.getpid()
    process = psutil.Process(process_id)
    
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=fpsize)
    
    fps = np.empty((len(smiles_list), fpsize // 4), dtype=np.uint8)
    for i, smile in enumerate(smiles_list):
        mol = MolFromSmiles(smile)
        fp = fpgen.GetFingerprintAsNumPy(mol)
        fp = fold_fingerprint(fp, fpsize)
        fp = fold_fingerprint(fp, fpsize // 2)
        fps[i] = fp

        if i % 5000 == 0 and i != 0 and verbose:
            print(f'{i}/{len(smiles_list)} calc. fps for {file_id}')
            print(f"{process.memory_info().rss / (1024 ** 2):.2f} MB for {file_id}")
        
        if i % 50000 == 0 and is_print_process():
            get_CPU_memory()

    del process_id, process, fpgen, i, smile, mol, fp
    gc.collect()
    
    return fps

def get_bitbirch_clusters(df, file_id, fpsize, bf, thr, verbose):
    """
    """
    process_id = os.getpid()
    process = psutil.Process(process_id)
    
    timer_birch = Timer(autoreset=True)
    timer_birch.start(f'[PROCESS {process_id}] BitBIRCH clustering executing for {file_id}')
    
    timer_fps = Timer(autoreset=True)
    timer_fps.start(f'[PROCESS {process_id}] Calculating fingerprints ({file_id})')
    fps = generate_fingerprints(df.SMILES, file_id, fpsize, verbose)
    timer_fps.stop()
    
    mem_fps = get_object_memory(fps)
    if verbose:
        print(f'[PROCESS {process_id}] {mem_fps} MB of memory ocupied by fps ({file_id})')
        print(f"[PROCESS {process_id}] {process.memory_info().rss / (1024 ** 2):.2f} MB of memory used after calculating fps ({file_id})")
    
    timer_fps = Timer(autoreset=True)
    timer_fps.start(f'[PROCESS {process_id}] Clustering ({file_id})')
    bitbirch = bb.BitBirch(branching_factor=int(bf), threshold=float(thr))
    bitbirch.fit(fps)
    timer_fps.stop()

    if verbose:
        print(f"[PROCESS {process_id}] {process.memory_info().rss / (1024 ** 2):.2f} MB of memory used after bitbirch clustering ({file_id})")

    centroids = bitbirch.get_centroids()

    cluster_list = bitbirch.get_cluster_mol_ids()
    
    if verbose:
        print(f"[PROCESS {process_id}] {process.memory_info().rss / (1024 ** 2):.2f} MB of memory used after loading bitbirch results" )
    
    print(f'[PROCESS {process_id}] Number of clusters for {file_id}: {len(cluster_list)}')

    print(f'[PROCESS {process_id}] Computing representatives for each cluster {file_id}')
    
    cluster_labels = [0] * fps.shape[0]
    representative_labels = [0] * fps.shape[0]

    for cluster_id, indices in enumerate(cluster_list):
        for idx in indices:
            cluster_labels[idx] = cluster_id

        # Retrieving cluster fingerprints and mathematical centroid fingerprint
        cluster_fps = [CreateFromBitString(''.join(str(int(x)) for x in fps[idx])) for idx in indices]
        centroid_fp = CreateFromBitString(''.join(str(int(x)) for x in centroids[cluster_id]))

        similarities = BulkTanimotoSimilarity(centroid_fp, cluster_fps)
        
        if verbose and len(indices)>5:
            print(f'cluster {cluster_id} with {len(similarities)} elements, representative similarity to centroid = {np.max(similarities)} ({file_id})')
        
        representative_labels[indices[np.argmax(similarities)]] = 1
        
    if verbose:
        print(f"[PROCESS {process_id}] {process.memory_info().rss / (1024 ** 2):.2f} MB of memory used after loading centroids and representatives")
    
    timer_birch.stop()
   
    return representative_labels, cluster_labels
