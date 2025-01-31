import os
import sys
import pandas as pd
import numpy as np
import pickle
import random
from multiprocessing import Pool, Manager
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
import psutil
import warnings

def process_clusters(cfp, batch_partitions, num_workers, n, min_mol, outname):
  
    with open(cfp, 'rb') as f:
        cfp = pickle.load(f)

    with Manager() as manager:
        
        with Pool(num_workers) as pool:
            # Generate tasks for each multiprocess in pool
            tasks = [(cluster_id, cfp[cluster_id], batch_partitions, n, min_mol, outname) for cluster_id in cfp.keys()]
            results = pool.starmap(get_representative, tasks)
    
    results = np.asarray(results)
    df = pd.DataFrame({'cluster':results[:,0],
        'representative_id':results[:,1],
        'representative_smiles':results[:,2],
        'cluster_size':results[:,4]})
    df = df.astype({"cluster": str, "representative_id": str,
        "representative_smiles": str, "cluster_size":int})


    df.to_csv(outname + '_clust_all.csv', index=False)

    df_subset = df[df['cluster_size'] >= min_mol]

    if len(df_subset) < n:
        warnings.warn(f"Less clusters ({len(df_subset)}) than required subset size ({n}). Maybe try reducing min_mol.", UserWarning)
        df_subset.to_csv(outname + '_clust_subset_' + str(len(df_subset)) + '.csv', index=False)
    elif len(df_subset) == n:
        df_subset.to_csv(outname + '_clust_subset_' + str(n) + '.csv', index=False)
    elif len(df_subset) > n:
        df_sorted = df_subset.sort_values(by="cluster_size", ascending=False)
        df_top_n = df_sorted.head(n)
        df_top_n.to_csv(outname + '_clust_subset_' + str(n) + '.csv', index=False)

def get_representative(cluster_id, centroid_fp, batch_partitions, n, min_mol, outname):
    
    process = psutil.Process(os.getpid())
    process_id = os.getpid()
    print(f"[PROCESS {process_id}] #### Processing cluster: {cluster_id} ####")
    #mem_info = process.memory_info()
    #print(f"Process {os.getpid()} - Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")

    partitions = np.genfromtxt(batch_partitions, dtype=str)

    representative = [cluster_id,'', '', 0, 0]
    for partition in partitions:
        #print(f"[PROCESS {process_id}] Processing partition {os.path.basename(partition)} for cluster {cluster_id}")
        part_df = pd.read_csv(partition)
        
        # Get molecules from given cluster in given partition
        part_cluster_df = part_df[part_df['centroid_cluster'] == cluster_id]
        
        if len(part_cluster_df) == 0:
            continue
        
        ids = part_cluster_df['ID'].to_list()
        smiles = part_cluster_df['SMILES'].to_list()
        del part_cluster_df
        representative[4]+=len(smiles)
         
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024)
        fps = [fpgen.GetFingerprint(mol) for mol in mols]

        similarities = BulkTanimotoSimilarity(centroid_fp, fps)
        idx = np.argmax(similarities)
        if similarities[idx] > representative[3]:
            representative[1] = ids[idx] 
            representative[2] = smiles[idx]
            representative[3] = similarities[idx]
    
    print(f"[PROCESS {process_id}] Most similar molecule to the centroid of the cluster {cluster_id} is {representative[2]} with a similarity of {representative[3]} to the centroid")
    return representative

def get_object_memory(obj):
    mem = sys.getsizeof(obj)
    mem = mem / (1024 ** 2)
    return mem

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Get a subset of cluster representatives')
    requiredArguments = parser.add_argument_group('Required Arguments')
    requiredArguments.add_argument('--cfp',
                                   help='Pickle file with fps of the centroids',
                                   required=True)
    requiredArguments.add_argument('--batch_partitions',
                                   help='Batch file with the paths to the DB partitions'
                                        ' updated with the pcluster and ccluster ids',
                                    required=True)
    requiredArguments.add_argument('-n',
                                   type=int,
                                   help='Number of cluster representatives for the subset',
                                   required=True)
    requiredArguments.add_argument('--cores',
                                   type=int,
                                   help='Number of cores to parallelise',
                                   required=True)
    parser.add_argument('--min_mol',
                        type=int,
                        help='Minimum number of molecules for a cluster to be'
                             'considered for representative extraction',
                        default=10)
    parser.add_argument('--outname',
                        help='Name of the csv file storing the representative molecules')
    
    args = parser.parse_args()

    cfp = args.cfp
    batch_partitions = args.batch_partitions
    n = args.n
    min_mol = args.min_mol
    outname = args.outname
    cores = args.cores

    process_clusters(cfp, batch_partitions, cores, n, min_mol, outname)
