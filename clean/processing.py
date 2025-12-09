from pathlib import Path
import os
import time
import argparse
import pandas as pd
import dask.dataframe as dd
from rdkit import Chem
from rdkit import rdBase
from dask.distributed import Client, performance_report
import pyarrow.parquet as pq
import datamol as dm
import json
import gc
import re


def parse_args():
    parser = argparse.ArgumentParser(description='Process SMILES')
    parser.add_argument('-bs', '--blocksize', type=str, help='Block size for dask dataframe. The safest is the default 64MB',  required=False, default='64MB')
    parser.add_argument('-o','--output_path', type=str, help='Output folder for the database', required=False,
                        default='Molecular_database')
    parser.add_argument('-s', '--group_size', type=int, help='The size of the file groups to read at once, default is 100 Gb for parquet files', required=False, default=100)
    parser.add_argument("-c", "--use_cols", nargs="+", help="Columns to read", required=False, 
                        default=['ID', "SMILES"])
    parser.add_argument("-p", "--progress_file", help="file name to keep the progress of the processing", required=False, default="progress_processing.txt")
    parser.add_argument('-db', '--database', type=str, help='The json file with the databases paths',  required=False, default="db.json")
    args = parser.parse_args()
    return args.database, args.blocksize, args.output_path, args.group_size, args.use_cols, args.progress_file

# -------------------------
#  ️ Setup Dask cluster in Slurm
# -------------------------    
scheduler_address = os.environ["DASK_SCHEDULER_ADDRESS"]

client = Client(scheduler_address)    # Connect to that cluster
client.wait_for_workers(n_workers=1, timeout=180)

# -------------------------
# 1️⃣ Setup RDKit tools
# -------------------------
blocker = rdBase.BlockLogs()

PEPTIDE_SMARTS = "[NX3:1][CX3:2](=[OX1:20])[CX4:3][NX3:4][CX3:5](=[OX1:21])" # dipeptide motif
Q = Chem.MolFromSmarts(PEPTIDE_SMARTS)
unsatu = Chem.MolFromSmarts('[CD2;R0]=[CD2;R0][CD2;R0]=[CD2;R0][CD2;R0]=[CD2;R0][CD2;R0]=[CD2;R0]')

tokenizer = re.compile(
        r'(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|B|C|b|c|n|o|s|p|\(|\)|\.|\=|\#|\-|\+|\\\\|\/|\:|\~|\@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])'
    )

halogen = re.compile(r'Br|Cl|F|I')
glycol = Chem.MolFromSmarts('CCOCCOCCO')

rare = {
    'He', 'Li', 'Be', 'Ne',
    'Na', 'Mg', 'Al', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Fl', 'Lv', 'Ts', 'Og'}


def _amide_cn_bond_not_ring(mol, a_idx, c_idx):
    b = mol.GetBondBetweenAtoms(a_idx, c_idx)
    return (b is not None) and (not b.IsInRing())

def _has_nonring_carbon_on_nonamide_side(mol, carbonyl_c_idx, amide_n_idx):
    """
    Require an external, NON-RING carbon neighbor on the non-amide side of the carbonyl carbon.
    (Prevents attachment directly to aromatic/ring carbon like ...C(=O)c1...)
    """
    c = mol.GetAtomWithIdx(carbonyl_c_idx)
    for nb in c.GetNeighbors():
        nb_idx = nb.GetIdx()
        if nb_idx == amide_n_idx:
            continue
        b = mol.GetBondBetweenAtoms(carbonyl_c_idx, nb_idx)
        if b is None:
            continue
        # skip the carbonyl oxygen
        if nb.GetAtomicNum() == 8 and b.GetBondType() == Chem.BondType.DOUBLE:
            continue
        # must be carbon, not in a ring, and the bond itself not a ring bond
        if nb.GetAtomicNum() == 6 and (not nb.IsInRing()) and (not b.IsInRing()):
            return True
    return False


def is_true_peptide(mol: Chem.Mol) -> bool:
    """
    True if the molecule contains two consecutive peptide (amide) bonds where:
      - the motif NX3-C(=O)-CX4-NX3-C(=O) is present,
      - NONE of the matched atoms are in a ring,
      - each amide C-N bond is NOT a ring bond,
      - and each carbonyl carbon is 'internal' to a NON-RING carbon on its non-amide side
        (so ...C(=O)c... is rejected).
    """

    for match in mol.GetSubstructMatches(Q):
        # Build map#:target_idx dictionary from the query order
        mapnum_to_idx = {}
        for q_atom, tgt_idx in zip(Q.GetAtoms(), match):
            amap = q_atom.GetAtomMapNum()
            if amap:
                mapnum_to_idx[amap] = tgt_idx

        n1 = mapnum_to_idx[1]
        c1 = mapnum_to_idx[2]
        spacer_c = mapnum_to_idx[3]
        n2 = mapnum_to_idx[4]
        c2 = mapnum_to_idx[5]
        # oxygens are 20,21 if you need them: o1 = mapnum_to_idx[20]; o2 = mapnum_to_idx[21]

        # 1) all matched atoms (including oxygens) must be non-ring
        if any(mol.GetAtomWithIdx(idx).IsInRing() for idx in match):
            continue

        # 2) both amide C-N bonds must not be ring bonds
        if not (_amide_cn_bond_not_ring(mol, n1, c1) and _amide_cn_bond_not_ring(mol, n2, c2)):
            continue

        # 3) require a NON-RING carbon neighbor on the non-amide side of each carbonyl carbon
        if not (_has_nonring_carbon_on_nonamide_side(mol, c1, n1) and
                _has_nonring_carbon_on_nonamide_side(mol, c2, n2)):
            continue

        return True

    return False


def normalize_smiles(smi: str) -> str | None:
    """Normalize SMILES by:
    - Converting to canonical isomeric SMILES
    - Removing salts (keeping largest fragment)
    - Neutralizing charges
    
    Parameters
    ----------
    smi : str
        Input SMILES string
        
    Returns
    ------- 
    str | None
        Normalized canonical SMILES or None if invalid
    """ 
    if not smi:
        return None
    
    try:
        with dm.without_rdkit_log():
            mol = dm.to_mol(smi, ordered=True)
            if mol is None:
                return None
            
            mol = dm.fix_mol(mol, largest_only=True if "." in smi else False)
            
            mol = dm.standardize_mol(
            mol,
            disconnect_metals=False,
            normalize=True,
            reionize=True if "+" in smi or "-" in smi else False,
            uncharge=False,
            stereo=False,
            )
            
            # filtrado por moleculas que contienen isotopos o metales
            if is_true_peptide(mol): # skip true peptides
                return None
            #poli ethylene glycol
            if mol.HasSubstructMatch(glycol):
                return None
            if mol.HasSubstructMatch(unsatu):
                return None

            sma = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            
            tokens = tokenizer.findall(sma)
            
            for atom in tokens:
                if atom.strip("[]") in rare:
                    return None
                
            if len(halogen.findall(sma)) >= 6:
                return None
                                  
        return sma
    
    except Exception:
        return None
    
    
def get_hac(smi: str) -> int:
    """Calculate Heavy Atom Count (HAC) from SMILES
    
    Parameters
    ----------
    smi : str
        SMILES string
    """
    mol = Chem.MolFromSmiles(smi)
    return mol.GetNumHeavyAtoms() if mol else 0


def split_csv_groups_by_size(
    base_paths: dict[str, list[str, list[str]]],
    size_limit_gb: int = 100
) -> dict[str, list[str]]:
    """
    Dynamically split large CSV datasets into groups of ~size_limit_gb.

    Parameters
    ----------
    base_paths : dict
        e.g. {"002": "ZINC22/smiles_csv/*", "003": "Savi/*/*.csv"}
    size_limit_gb : int
        Max cumulative size per group (in GB).

    Returns
    -------
    dict
        e.g. {"002_1": ["ZINC22/smiles_csv/H04/file1.csv", ...], "003": ["Savi/a.csv", ...]}
    """
    size_limit = size_limit_gb * (1024 ** 3)
    new_mapping = {}

    for dbid, patterns in base_paths.items():
        if isinstance(patterns, str):
            patterns = [patterns]

        # Gather all CSV files from given patterns
        all_files = []
        for pattern in patterns:
            all_files.extend(Path().glob(str(pattern)))

        # Filter to only existing CSV files
        all_files = [p for p in all_files if p.is_file() and p.suffix.lower() in [".csv", ".parquet"]]

        # Collect file sizes
        file_sizes = [(p, p.stat().st_size) for p in all_files]
        file_sizes.sort(key=lambda x: x[0].as_posix())  # deterministic ordering

        # Group by cumulative size
        current_group, current_size, group_num = [], 0, 1
        for p, fsize in file_sizes:
            # if a single file is larger than threshold → its own group
            if fsize > size_limit:
                key = f"{dbid}_{group_num:02d}"
                new_mapping[key] = [str(p)]
                group_num += 1
                continue
            if current_size + fsize > size_limit and current_group:
                key = f"{dbid}_{group_num:02d}" if group_num > 1 else f"{dbid}_1"
                new_mapping[key] = [str(f) for f in current_group]
                group_num += 1
                current_group, current_size = [], 0
            current_group.append(p)
            current_size += fsize
        # Add any leftovers
        if current_group:
            key = f"{dbid}_{group_num:02d}" if group_num > 1 else dbid
            new_mapping[key] = [str(f) for f in current_group]

    return new_mapping


def write_db_by_hac(db_id: str, pattern: list[str], output_folder: Path, 
                    blocksize="64MB", use_cols: tuple[str] = ("ID", "SMILES")) -> None:
    """
    Normalize SMILES, deduplicate locally,
    split by HAC, and write to HAC-specific Parquet folders.
    
    parameters
    ---------
    db_id : str
        Database ID
    pattern : str
        Input file pattern
    output_folder : Path
        Output folder
    blocksize : str, optional 
        Dask block size
        
    Returns
    -------
    None
    
    """
    meta = {
    "ID": "string",
    "SMILES": "string",
    "db_id": "string",
    "HAC": "int64",
    }
    
    group_id = db_id
    if len(db_id.split("_")) > 1:
        group_id = db_id.split("_")[0]
        
    if Path(pattern[0]).suffix == ".csv":
        ddf = dd.read_csv(pattern, blocksize=blocksize, usecols=list(use_cols))
    elif Path(pattern[0]).suffix == ".parquet":
        ddf = dd.read_parquet(pattern, blocksize=blocksize, columns=list(use_cols))
    else:
        print(f"⚠️ Skipping unsupported file type: {pattern}")
        raise NotImplementedError("File not supported")
        
    ddf = ddf.dropna(subset=["SMILES"])

    # Normalize SMILES
    ddf["SMILES"] = ddf.map_partitions(lambda df: df["SMILES"].map(normalize_smiles), meta=("SMILES", str))
    ddf = ddf.dropna(subset=["SMILES"])
    ddf["db_id"] = group_id
    
    # HAC calculation
    ddf["HAC"] = ddf.map_partitions(lambda df: df["SMILES"].map(get_hac), meta=("HAC", int))
    ddf = ddf.astype(meta)
    # Write all partitions in parallel, grouped by HAC
    ddf.to_parquet(
        output_folder/db_id,
        write_index=False,
        partition_on=["HAC"],
        engine="pyarrow", # it seems to be faster apparently
    )

    del ddf
    client.run(gc.collect)

    print(f"DB {db_id} written to {output_folder}")

           
def check_files(pattern: list[str], db_id: str, use_cols: tuple[str] = ("ID", "SMILES")) -> list[str]:
    """
    Check if all files in pattern contain ID and SMILES columns
    """
    wrong_files = []
    right_files = []
    for f in pattern:
        f = Path(f)
        try:
            if f.suffix == ".csv":
                cols = pd.read_csv(f, nrows=0).columns
            elif f.suffix == ".parquet":
                cols = pq.ParquetFile(f).schema.names
            else:
                print(f"⚠️ Skipping unsupported file type: {f}")
                wrong_files.append(f"{f}\n")
                continue

            if not all(c in cols for c in use_cols):
                wrong_files.append(f"{f}\n") 
                continue
        except Exception as e:
            print(e)
            wrong_files.append(f"{f}\n")
            continue
        
        right_files.append(f)
        
    if wrong_files:    
        with open(f"wrong_{db_id}.txt", "w", encoding="utf-8") as wrong:
            wrong.writelines(wrong_files)
            
    return right_files
                             

# -------------------------
# 2️⃣ Read databases lazily
# -------------------------

def main():
    base_db, block_size, output_folder, group_size, use_cols, progress_file = parse_args()
    
    start = time.perf_counter()
    
    base_db = json.loads(Path(base_db).read_text())
    with performance_report(filename="dask-HAC.html"):
        db_files = split_csv_groups_by_size(base_db, group_size)
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)
        progress = Path(progress_file)
        progress.touch(exist_ok=True)
        for db_id, pattern in db_files.items():
            pattern = check_files(pattern, db_id, use_cols)    
            if f"DB {db_id} done\n" in progress.read_text():
                print(f"DB {db_id} already done, skipping.")            
                continue
            write_db_by_hac(db_id, pattern, out_path, block_size, use_cols)
            with open(progress, "a") as f:
                f.write(f"DB {db_id} done\n")
                
            
        end = time.perf_counter()
        print(f"Initial cleaning completed in {end - start:.2f} seconds")    

if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()