import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from multiprocessing import Pool
import argparse
import glob

def process_file(file_path, mw_threshold, verbose=False):
    df = pd.read_csv(file_path)
    results = []
    file_id = os.path.basename(file_path)

    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            if mw >= mw_threshold:
                results.append([row['ID'], row['SMILES'], mw])

        if verbose and i % 5000 == 0 and i != 0:
            print(f'{i}/{len(df)} calculated MWs for {file_id}')

    return results

def process_all_files(input_dir, mw_threshold, num_cores, outname, verbose=False):
    files = glob.glob(os.path.join(input_dir, '*.csv'))
    args = [(file, mw_threshold, verbose) for file in files]

    all_results = []
    with Pool(num_cores) as pool:
        for result in pool.starmap(process_file, args):
            all_results.extend(result)

    df = pd.DataFrame(all_results, columns=['ID', 'SMILES', 'MW'])
    df.to_csv(outname, index=False)
    print(f"Saved {len(df)} molecules above MW {mw_threshold} to {outname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter molecules by MW from CSV files in a directory.")
    parser.add_argument('--input_dir',
                        help='Directory containing CSV files with SMILES and ID columns',
                        required=True)
    parser.add_argument('--mw_threshold',
                        type=float,
                        help='Molecular weight threshold',
                        required=True)
    parser.add_argument('--cores',
                        type=int,
                        help='Number of CPU cores to use',
                        required=True)
    parser.add_argument('--outname',
                        help='Output CSV file name')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()
    
    input_dir = args.input_dir
    mw_threshold = args.mw_threshold
    cores = args.cores
    outname = args.outname
    verbose = args.verbose

    process_all_files(input_dir, mw_threshold, cores, outname, verbose)
