import argparse
import os 
import glob
import progressbar
import pickle
from multiprocessing import Pool
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem

def get_info_db(input_dir):
    files = glob.glob('%s/*' % input_dir)
    total_molecules = 0
    bar = progressbar.ProgressBar(maxval=len(files)).start()
    for i, file in enumerate(files):
        bar.update(i)
        molecules = os.popen(f'wc -l < {file}').read().strip()
        molecules = int(molecules)-1
        total_molecules += molecules
    bar.finish()
    return len(files), total_molecules

def compute_fp(smiles, fpgen):
    """
    Compute the Morgan fingerprint for a given smiles
    """
    # some smiles might have invalid smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = fpgen.GetSparseCountFingerprint(mol)
        return(fp)
    except Exception as e:
        print(f'Error processing SMILES {smiles}: {e}')
        return None


def process_file(file):
    """
    Process a single file and compute the fingerprints for each SMILE string
    """
    try:
        process_id = os.getpid()
        print(f'[Process {process_id}] Processing file: {file}')

        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4)

        df = pd.read_csv(file)
        smiles_list = df['SMILES'].tolist()
        ids_list = df['ID'].tolist()
        fps = [compute_fp(smiles, fpgen) for smiles in smiles_list]

        # create a dicctionary where the keys are the ids and the values the fingerprints
        fps_dict = dict(zip(ids_list, fps))
        
        with open(file.replace('.csv','.pkl'), 'wb') as f:
            pickle.dump(fps_dict, f)
    except Exception as e:
        print(f'Error processing file {file}: {e}')

def process_directory(input_dir, num_workers):
    """
    Process files in the input_dir in parallel
    """
    files = glob.glob('%s/*' % input_dir)
    
    with Pool(num_workers) as pool:
        pool.map(process_file, files)



if __name__ == '__main__':

     parser = argparse.ArgumentParser(description='')
     parser.add_argument('--db',
                         type=str,
                         help='')
     parser.add_argument('--size',
                         type=str,
                         help='')
     parser.add_argument('--cores', type=int, default=32, help='')
     
     args = parser.parse_args()
     
     #partitions, molecules = get_info_db(input_dir=args.db)

     #process_directory(args.db, args.cores)

