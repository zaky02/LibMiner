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
    args = parser.parse_args()
    return args.blocksize, args.output_path, args.group_size

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
    
    
def normalize_smiles_partition(smiles_series: pd.Series) -> pd.Series:
    """Vectorized normalization per partition."""
    return pd.Series(
        [normalize_smiles(smi) for smi in smiles_series],
        index=smiles_series.index,
        dtype="string",
    )


def get_hac(smiles_series):
    """Compute HAC for a pandas Series of SMILES in a vectorized partition."""
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_series]
    hacs = [mol.GetNumHeavyAtoms() if mol else 0 for mol in mols]
    return pd.Series(hacs, index=smiles_series.index)


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
    ddf["SMILES"] = ddf.map_partitions(lambda df: normalize_smiles_partition(df["SMILES"]), meta=("SMILES", str))
    ddf = ddf.dropna(subset=["SMILES"])
    ddf["db_id"] = group_id
    
    # HAC calculation
    ddf["HAC"] = ddf.map_partitions(lambda df: get_hac(df["SMILES"]), meta=("HAC", int))
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
        
                                        
def encode_id(df, hac: int, db_bits=5, hac_bits=10, hash_bits=49):
    """
    Vectorized 64-bit packed ID generation using 128-bit blake2b hash folded to 64 bits.

    Combines db_id, HAC, and a 49-bit hash into a unique 64-bit integer.
    When generating the search database, it will make retrieving the molecules more easily.

    Parameters
    ----------
    df : DataFrame partition (must have columns 'ID' and 'db_id')
    hac : int
        Heavy atom count to embed
    db_bits, hac_bits, hash_bits : int
        Bit allocation (must sum to <= 64)
    """
    assert db_bits + hac_bits + hash_bits <= 64, "Bit allocations exceed 64 bits"

    ids = df["ID"].to_numpy(dtype=str)
    db_ids = df["db_id"].to_numpy(dtype=np.uint64)
    n = len(ids)

    # Masks and shifts
    hash_mask = np.uint64((1 << hash_bits) - 1)
    db_mask   = np.uint64((1 << db_bits) - 1)
    db_shift  = np.uint64(hac_bits + hash_bits)
    hac_shift = np.uint64(hash_bits)
    hac_arr   = np.full(n, np.uint64(hac), dtype=np.uint64)

    # Ensure db_id fits allocated bits
    db_ids = db_ids & db_mask
    assert np.all(db_ids < (1 << db_bits)), f"db_id exceeds {2**db_bits-1}"

    # Generate 128-bit Blake2b hash and fold into 64 bits
    full_hashes = np.empty(n, dtype=np.uint64)
    for i, s in enumerate(ids):
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest()
        hi, lo = struct.unpack(">QQ", h)
        full_hashes[i] = np.uint64(hi ^ lo)

    # Keep only hash_bits bits
    hash_parts = full_hashes & hash_mask

    # Pack fields: [db_id | HAC | hash_part]
    packed_ids = (db_ids << db_shift) | (hac_arr << hac_shift) | hash_parts
    return packed_ids


def make_name_function(hac: int):
    def name_function(i: int) -> str:
        return f"HAC{hac}_{i:02d}.parquet"
    return name_function

def deduplicate(hac_folders: Path | str, block_size: str, out_path: Path | str, 
                use_cols: tuple[str] = ("ID", "SMILES")):
    """
    Normalize SMILES, deduplicate locally,
    split by HAC, and write to HAC-specific Parquet folders.
    """
    
    out_path = Path(out_path)
    hac_folders = Path(hac_folders)
    hac = hac_folders.name.split("_")[-1]
    meta = {"ID": "uint64", "SMILES": "string", "db_id": "string"}
    stats = Path("stats.txt")
    stats.touch(exist_ok=True)
    
    # read parquet files from a HAC  
    ddf_merged = dd.read_parquet(f"{hac_folders}/*.parquet", blocksize=block_size, 
                                columns=[*use_cols, "db_id"])

    # Deduplicate across all sources using normalized SMILES
    ddf_merged = ddf_merged.drop_duplicates(subset="SMILES").drop_duplicates(subset=["ID"])
    # Apply to Dask DataFrame
    ddf_merged["ID"] = ddf_merged.map_partitions(
    lambda df: pd.Series(encode_id(df, hac=int(hac)), index=df.index),
    meta=("ID", "uint64"))
    
    ddf_merged = ddf_merged.astype(meta)
    # Aim for ≤15M rows per partition because this is for each HAC
    ddf_merged = ddf_merged.repartition(partition_size="500MB")
    size = ddf_merged.shape[0]
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
    
    with open(stats, "a") as w:
        w.write(f"HAC {hac}: {size}\n")

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
    block_size, output_folder, group_size = parse_args()
    
    start = time.perf_counter()
    
    with performance_report(filename="dask-HAC.html"):
        db_files = split_csv_groups_by_size(base_db, group_size)
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)
        progress = Path("progress.txt")
        progress.touch(exist_ok=True)
        for db_id, pattern in db_files.items():
            pattern = check_files(pattern, db_id)    
            if f"DB {db_id} done\n" in progress.read_text():
                print(f"DB {db_id} already done, skipping.")            
                continue
            write_db_by_hac(db_id, pattern, out_path, block_size)
            with open(progress, "a") as f:
                f.write(f"DB {db_id} done\n")

        rename_partitions(out_path)
        
    # -------------------------
    # 3️⃣ Deduplicate globally by HAC
    # -------------------------

        end = time.perf_counter()
        print(f"Initial cleaning completed in {end - start:.2f} seconds")  
         

        for hac_folders in out_path.glob("HAC*"):
            hac = hac_folders.name.split("_")[-1]
            if f"HAC {hac}" in progress.read_text():
                print(f"HAC {hac} already done, skipping.")            
                continue
            deduplicate(hac_folders, block_size, out_path)
            with open(progress, "a") as f:
                f.write(f"HAC {hac} done\n")
                
        end2 = time.perf_counter()
        print(f"Completed in {end2 - end:.2f} seconds")



if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()