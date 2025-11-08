from pathlib import Path
import os
import time
import argparse
import pandas as pd
import dask.dataframe as dd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from dask.distributed import Client, performance_report
import pyarrow.parquet as pq

def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate SMILES')
    parser.add_argument('-bs', '--blocksize', type=str, help='Block size for dask dataframe. The safest is the default 64MB',  required=False, default='64MB')
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
            if  not int(hac_value):
                new_hac_folder = output_folder / f"wrong_HAC_{hac_value}"
            else:
                new_hac_folder = output_folder / f"HAC_{hac_value}"
                
            new_hac_folder.mkdir(exist_ok=True, parents=True)

            # Move and rename parquet files
            for i, parquet_file in enumerate(hac_folder.glob("*.parquet"), start=1):
                new_name = new_hac_folder / f"HAC{hac_value}_db{db_folder.name}_{i:02d}.parquet"
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
base_db = { "001": "Enamine_REAL_65B_parquet/*.parquet",
            "002": "ZINC22/*.csv",  # H04_to_H24
            "003": "Savi_parquet/*/*.parquet",
            "004": "PubChem_parquet/*.parquet",
            "005": "ChEMBL_parquet/*.parquet",
            "006": "DrugBank/*.parquet",
            "007": "Coconut_parquet/*.parquet",
            "008": "NPAtlas_parquet/*.parquet",
            "009": "ChemistriX_Clean/VIRTUAL_BIUR_POR_MW_CLEAN/*.parquet",
            "010": "CHIPMUNK_parquet/*.parquet",
            "011": "SCUBIDOO_parquet/*.parquet",
            "012": "SureChEMBL_parquet/*.parquet",
            "013": "MolPort_parquet/*.parquet"
        }

def main():
    block_size, output_folder, group_size, use_cols = parse_args()
    
    start = time.perf_counter()
    
    with performance_report(filename="dask-HAC.html"):
        db_files = split_csv_groups_by_size(base_db, group_size)
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)
        progress = Path("progress_processing.txt")
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

if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()