import os
import pandas as pd
import numpy as np
import pickle
import random
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit.DataStructs import TanimotoSimilarity

def process_batch(batch_partitions, num_workers, cfp, n, min_mol, outname):
    files = np.genfromtxt(batch_partitions, dtype=str)
    with Pool(num_workers) as pool:
        pool.starmap(get_subset, [(cfp, part_file, n, min_mol, outname) for part_file in files])

    merge_and_select_representatives(outname, n)

def get_subset(cfp, part_file, n, min_mol, outname):
    
    process_id = os.getpid()
    print(f"[PROCESS {process_id}] Processing file: {part_file}")
    part_df = pd.read_csv(part_file)
    representatives = []

    with open(cfp, 'rb') as f:
        cfp = pickle.load(f)
    
    for keys, values in cfp.items():
        # Keys: final cluster ID
        # Values: fingerprints (numpy darray)
       
        print(f"[PROCESS {process_id}] Retriving set of molecules belonging to ccluster")
        set_mols = part_df[part_df['centroid_cluster'] == keys]

        if len(set_mols) < min_mol:
            continue

        print(f"[PROCESS {process_id}] Retrieving fingerprints of the ccluster molecules")
        mols = [Chem.MolFromSmiles(smiles) for smiles in set_mols.SMILES]
        set_mols.loc[:, 'fingerprint'] = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=1024) for mol in mols]
        X = np.array([list(fp) for fp in set_mols['fingerprint']])
         
        print(f"[PROCESS {process_id}] Retriving most similar molecule to the centroid of the ccluster")
        pop_counts = np.sum(X, axis=1)
        a_centroid = np.dot(X, values)
        
        sims_med = a_centroid / (pop_counts + np.sum(values) - a_centroid)
        sim_mol = np.argmax(sims_med)
        
        print(f"[PROCESS {process_id}] Storing n representatives of the DB")
        representatives.append(set_mols.iloc[[sim_mol]])
        representatives_df = pd.concat(representatives, ignore_index=True)
        representatives_df.to_csv(f"{outname}_{process_id}_results.csv", index=False)

def merge_and_select_representatives(outname, n):
    temp_files = [f for f in os.listdir() if f.endswith("_results.csv")]
    merged_df = pd.concat([pd.read_csv(f) for f in temp_files], ignore_index=True)

    merged_df = merged_df.drop_duplicates(subset=['centroid_cluster'])
    merged_df = merged_df.drop(columns=['fingerprint'])

    n_representatives = merged_df.sample(n=min(n, len(merged_df)))
    n_representatives.to_csv(f"{outname}_final.csv", index=False)

    for f in temp_files:
        os.remove(f)


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

    process_batch(batch_partitions, cores, cfp, n, min_mol, outname)
