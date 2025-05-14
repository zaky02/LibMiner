import os
import psutil
import pandas as pd
import numpy as np
from multiprocessing import Pool, current_process
from rdkit import Chem
from rdkit.Chem import Descriptors

def is_print_process():
    return current_process().name == "ForkPoolWorker-1"

def process_file(file, mw_threshold, verbose):
    process_id = os.getpid()
    process = psutil.Process(process_id)
    file_id = os.path.basename(file)

    if verbose:
        print(f"[PROCESS {process_id}] Memory at start: {process.memory_info().rss / (1024 ** 2):.2f} MB")
        if is_print_process():
            print_memory_summary()

    print(f"[PROCESS {process_id}] Processing file: {file}")
    df = pd.read_csv(file)

    if verbose:
        print(f"[PROCESS {process_id}] Memory after loading {file_id}: {process.memory_info().rss / (1024 ** 2):.2f} MB")
        if is_print_process():
            print_memory_summary()

    smiles_list = df['SMILES']
    ids = df['ID']
    mw_data = []

    for i, smile in enumerate(smiles_list):
        if i % 5000 == 0 and i != 0 and verbose:
            print(f'{i}/{len(smiles_list)} calc. mw for {file_id}')

        if isinstance(smile, str):
            mol = Chem.MolFromSmiles(smile)
        else:
            continue

        if mol is None:
            continue
        mw = Descriptors.MolWt(mol)

        if mw >= mw_threshold:
            mw_data.append([ids[i], smile, mw])

    print(f"[PROCESS {process_id}] Finished processing {file_id}")
    return mw_data

def process_batch(batch_partitions, num_workers, mw_threshold, verbose):
    files = np.genfromtxt(batch_partitions, dtype=str)

    if isinstance(files, np.ndarray) and files.ndim == 0:
        files = np.array([files])
    elif isinstance(files, str):
        files = [files]

    process_file_args = [(file, mw_threshold, verbose) for file in files]
    all_mw_data = []

    with Pool(num_workers) as pool:
        result = pool.starmap_async(process_file, process_file_args)

        try:
            output = result.get()
            for file_mw_data in output:
                all_mw_data.extend(file_mw_data)

        except Exception as e:
            print(f"Error encountered: {e}. Stopping execution.")
            pool.terminate()
            pool.join()
            raise

    mw_df = pd.DataFrame(all_mw_data, columns=['ID', 'SMILES', 'MW'])
    mw_df.to_csv(f'{batch_partitions}_molecules_above_threshold.csv', index=False)

    print(f"\n========== BATCH SUMMARY ==========")
    print(f"Total qualifying molecules (MW > {mw_threshold}) saved!")
    print(f"Total number of qualifying molecules: {len(all_mw_data)}")

def print_memory_summary():
    import psutil
    mem = psutil.virtual_memory()
    print(f"CPU Memory Usage: {mem.percent}% of {mem.total / (1024**3):.2f} GB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Retrieve molecules based on molecular weight thresholds from CSV files in batches and store the results in a single file.')
    requiredArguments = parser.add_argument_group('Required Arguments')
    requiredArguments.add_argument('--batch_partitions',
                                   help='Text file containing paths to CSV partitions',
                                   required=True)
    requiredArguments.add_argument('--cores',
                                   help='Number of cores for multiprocessing',
                                   type=int,
                                   default=15,
                                   required=True)
    requiredArguments.add_argument('--mw_threshold',
                                   help='MW threshold to filter the molecular DB',
                                   type=int,
                                   default=400,
                                   required=True)
    parser.add_argument('-v',
                        default=False)

    args = parser.parse_args()
    
    batch_partitions = args.batch_partitions
    cores = args.cores
    mw_threshold = args.mw_threshold
    verbose = args.v

    process_batch(batch_partitions, cores, mw_threshold, verbose)
