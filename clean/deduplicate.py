from pathlib import Path
import os
import time
import argparse
import pandas as pd
import hashlib, struct
import numpy as np
import dask.dataframe as dd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from dask.distributed import Client, performance_report

def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate SMILES')
    parser.add_argument('-bs', '--blocksize', type=str, help='Block size for dask dataframe. The safest is the default 128MB',  required=False, default='64MB')
    parser.add_argument('-o','--output_path', type=str, help='Output foldr for the database', required=False,
                        default='Molecular_database')
    parser.add_argument('-s', '--group_size', type=int, help='The size of the file groups to read at once, default is 300 Gb', required=False, default=300)
    parser.add_argument("-c", "--use_cols", nargs="+", help="Columns to read", required=False, 
                        default=['ID', "SMILES"])
    args = parser.parse_args()
    return args.blocksize, args.output_path, args.group_size, args.use_cols

# -------------------------
#  ️ Setup Dask cluster in Slurm
# -------------------------

scheduler_address = os.environ["DASK_SCHEDULER_ADDRESS"]

client = Client(scheduler_address)    # Connect to that cluster

# -------------------------
# 1️⃣ Setup RDKit tools
# -------------------------
RDLogger.DisableLog('rdApp.*')

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
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            return None
        mol = rdMolStandardize.Normalize(mol)  
        # Apply salt removal only if multiple fragments
        if "." in smi:
            lfs = rdMolStandardize.LargestFragmentChooser()
            mol = lfs.choose(mol)
        # Apply uncharging only if charges are present
        if "+" in smi or "-" in smi:
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)
        
        Chem.SanitizeMol(mol)      # ensure valid again after modifications, it is in place

        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
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
    size_limit_gb: int = 300
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
        all_files = [p for p in all_files if p.is_file() and p.suffix.lower() == ".csv"]

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


def write_db_by_hac(db_id: str, pattern: str, output_folder: Path, 
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
    if "002" in db_id:
        group_id = db_id.split("_")[0]

    ddf = dd.read_csv(pattern, blocksize=blocksize, usecols=list(use_cols))
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
    print(f"DB {db_id} written to {output_folder}")


def rename_partitions(output_folder: Path):
    """
    Adjust renaming logic for new database structure:
    - Remove db_id level
    - Rename HAC folders from HAC=value to HAC_value
    - Rename parquet files to HAC_value_dbid_X.parquet
    """
    output_folder = Path(output_folder)

    # Traverse db_id folders
    for db_folder in output_folder.iterdir():
        if not db_folder.is_dir():
            continue
        # Inside each db_id, look for HAC=* folders
        for hac_folder in db_folder.glob("HAC=*"):
            hac_value = hac_folder.name.split("=")[-1]
            new_hac_folder = output_folder / f"HAC_{hac_value}"
            new_hac_folder.mkdir(exist_ok=True, parents=True)

            # Move and rename parquet files
            for i, parquet_file in enumerate(hac_folder.glob("*.parquet"), start=1):
                new_name = new_hac_folder / f"HAC{hac_value}_db{db_folder.name.split("_")[0]}_{i:02d}.parquet"
                parquet_file.rename(new_name)
            # Remove old HAC folder
            if not any(hac_folder.iterdir()):
                hac_folder.rmdir()

        # Remove old db_id folder
        if not any(db_folder.iterdir()):
            db_folder.rmdir()

           
def check_files(pattern: list[str], db_id: str, use_cols: tuple[str] = ("ID", "SMILES")) -> list[str]:
    """
    Check if all files in pattern contain ID and SMILES columns
    """
    wrong_files = []
    right_files = []
    for f in pattern:
        try:
            cols = pd.read_csv(f, nrows=0).columns
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
                             

def assign_ids(df, partition_offsets, global_offset, meta):
    """
    Assign unique sequential IDs per partition using map_partitions.
    `partition_offsets` must be broadcast as list of integers.
    """
    # Each partition gets its starting offset from partition_offsets
    def add_ids(partition, partition_info=None):
        pid = partition_info["number"]
        offset = partition_offsets[pid]
        partition["ID"] = np.arange(len(partition), dtype=np.int64) + global_offset + offset
        return partition

    return df.map_partitions(add_ids, meta=meta)


def make_name_function(hac: int):
    def name_function(i: int) -> str:
        return f"HAC{hac}_{i:02d}.parquet"
    return name_function


def deduplicator(hac_folders: Path | str, out_path: Path | str, block_size: str = "64MB",
                 use_cols: tuple[str] = ("ID", "SMILES"), current_offset: int = 0):
    """
    deduplicate, assign unique numerical IDs and write to HAC-specific Parquet folders.
    """
    
    out_path = Path(out_path)
    hac_folders = Path(hac_folders)
    hac = hac_folders.name.split("_")[-1]
    meta = {"ID": "uint64", "SMILES": "string"}
    
    # read parquet files from a HAC  
    ddf_merged = dd.read_parquet(f"{hac_folders}/*.parquet", blocksize=block_size, 
                                columns=use_cols)

    # Deduplicate across all sources using normalized SMILES
    ddf_merged = ddf_merged.drop_duplicates(subset="SMILES").drop_duplicates(subset=["ID"])
    
    # Aim for ≤15M rows per partition because this is for each HAC
    ddf_merged = ddf_merged.repartition(partition_size="500MB")

    # Compute number of rows per partition (fast metadata op)
    partition_lengths = ddf_merged.map_partitions(len).compute()
    partition_offsets = np.insert(np.cumsum(partition_lengths[:-1]), 0, 0)
    # Assign unique IDs within Dask graph (no Python loop)
    ddf_merged = assign_ids(ddf_merged, partition_offsets, current_offset, meta)
    count = int(sum(partition_lengths))
    
    # -------------------------
    # 4️⃣ Write the database
    # -------------------------
    ddf_merged.to_parquet(
        out_path / f"cleaned/{hac}",
        write_index=False,
        compute=True,
        engine="pyarrow",
        name_function=make_name_function(hac=int(hac))     
    )
    
    return count

# -------------------------
# 2️⃣ Read databases lazily
# -------------------------
base_db = { "001": "Enamine_REAL_65B/Enamine_REAL_65B_partitioned_15M/*.csv",
            "002": "ZINC22/smiles_csv/H*/*_clean.csv",  # H04_to_H24
            "003": "Savi/*/*_clean.csv",
            "004": "PubChem/*_clean.csv",
            "005": "ChEMBL/*_clean.csv",
            "006": "DrugBank/*_clean.csv",
            "007": "Coconut/*_clean.csv",
            "008": "NPAtlas/*_clean.csv",
            "009": "ChemistriX_Clean/VIRTUAL_BIUR_POR_MW_CLEAN/*_clean.csv",
            "010": "CHIPMUNK/*_clean.csv",
            "011": "SCUBIDOO/*_clean.csv",
            "012": "SureChEMBL/*_clean.csv",
        }

def main():
    block_size, output_folder, group_size, use_cols = parse_args()
    
    start = time.perf_counter()
    
    with performance_report(filename="dask-HAC.html"):
        db_files = split_csv_groups_by_size(base_db, group_size)
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)
        progress = Path("progress.txt")
        progress.touch(exist_ok=True)
        for db_id, pattern in db_files.items():
            pattern = check_files(pattern, db_id, use_cols)    
            if f"DB {db_id} done\n" in progress.read_text():
                print(f"DB {db_id} already done, skipping.")            
                continue
            write_db_by_hac(db_id, pattern, out_path, block_size, use_cols)
            with open(progress, "a") as f:
                f.write(f"DB {db_id} done\n")
                
        rename_partitions(out_path)
            
        end = time.perf_counter()
        print(f"Initial cleaning completed in {end - start:.2f} seconds")    
    # -------------------------
    # 3️⃣ Deduplicate globally by HAC
    # -------------------------
        stats = out_path/"stats.txt"
        stats.touch(exist_ok=True)
        current_offset = 0   
        with open(stats, "r") as st:
            lines = {int(x.split(":")[0].strip("HAC")): int(x.strip().split(":")[-1]) for x in st.readlines()}
            
        hacs = sorted(out_path.glob("HAC*"), key=lambda x: int(x.name.split("_")[-1]))
        for hac_folders in hacs:
            hac = hac_folders.name.split("_")[-1]
            if f"HAC {hac}" in progress.read_text():
                print(f"HAC {hac} already done, skipping.")  
                current_offset = current_offset + lines[int(hac)]       
                continue
            
            count = deduplicator(hac_folders, out_path, block_size, use_cols, current_offset)
            current_offset = current_offset + count
            
            with open(progress, "a") as f:
                f.write(f"HAC {hac} done\n")
                
            with open(stats, "a") as w:
                w.write(f"HAC {hac}: {count}\n")
                        
        end2 = time.perf_counter()
        print(f"Completed in {end2 - end:.2f} seconds")



if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()