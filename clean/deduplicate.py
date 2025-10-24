from pathlib import Path
import os
import time
import argparse
import pandas as pd
import hashlib, struct
from collections import defaultdict
import dask.dataframe as dd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from dask.distributed import Client, performance_report

def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate SMILES')
    parser.add_argument('-bs', '--blocksize', type=str, help='Block size for dask dataframe. The safest is the default 128MB',  required=False, default='128MB')
    parser.add_argument('-o','--output_path', type=str, help='Output foldr for the database', required=False,
                        default='Molecular_database')
    parser.add_argument('-s', '--group_size', type=int, help='The size of the file groups to read at once, default is 400 Gb', required=False, default=400)
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
        
        if "." in smi or "+" in smi or "-" in smi:    
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
    size_limit_gb: int = 400
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
                    blocksize="128MB") -> None:
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
    "HAC": "int64",
    }

    ddf = dd.read_csv(pattern, blocksize=blocksize, usecols=["ID","SMILES"])
    ddf = ddf.dropna(subset=["SMILES"])

    # Normalize SMILES
    ddf["SMILES"] = ddf.map_partitions(lambda df: df["SMILES"].map(normalize_smiles), meta=("SMILES", str))
    ddf = ddf.dropna(subset=["SMILES"])

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

           
def check_files(pattern: list[str], db_id: str) -> list[str]:
    """
    Check if all files in pattern contain ID and SMILES columns
    """
    wrong_files = []
    right_files = []
    for f in pattern:
        try:
            cols = pd.read_csv(f, nrows=0).columns
            if not all(c in cols for c in ["ID", "SMILES"]):
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
        
                                        
def make_packed_id(s: str, db_id: int, hac: int,
                   db_bits: int = 5, hac_bits: int = 10, hash_bits: int = 49) -> int:
    """Pack db_id, HAC, and a truncated hash into a single 64-bit integer.
        The db_bits determine the maximum number allowed -> for db_bits of 5. It means db_id can
        be up to 2^db_bits -> 32 databases.
        2^hac_bits for up to 1024 HAC counts
        The hash_bits controls the uniqueness of the IDs generated -> there could be collisions with the same hash IDs although it  might be improbable for them to also have the same HAC and db_id
    
    """
    assert db_id < (1 << db_bits)
    assert hac < (1 << hac_bits)
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8)
    full_hash = struct.unpack(">Q", h.digest())[0]
    hash_part = full_hash & ((1 << hash_bits) - 1)
    packed = (db_id << (hac_bits + hash_bits)) | (hac << hash_bits) | hash_part
    return packed


def make_name_function(db_id: int, hac: int):
    def name_function(i: int) -> str:
        return f"HAC{hac}_db{db_id}_{i:02d}.parquet"
    return name_function

# -------------------------
# 2️⃣ Read databases lazily
# -------------------------
base_db = { "001": "Enamine/*_clean.csv",
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
        
        groups = defaultdict(list) 
        folders = out_path.glob("HAC*/*.parquet")
        for p in folders:
            # Extract identifier — customize this part as needed
            hac = int(p.name.split("_")[0].strp("HAC"))   # e.g., "001"
            db = int(p.name.split("_")[1].strp("db"))
            groups[(db, hac)].append(p)
            
        for (db, hac), files in groups.items():
            
            meta = {
                "ID": "int64",
                "SMILES": "string",
                }
            
            ddf_merged = dd.read_parquet(files, chunksize=block_size, 
                                        columns=["ID","SMILES"])

            # Deduplicate across all sources using normalized SMILES
            ddf_merged = ddf_merged.drop_duplicates(subset="SMILES")
            ddf_merged["ID"] = ddf_merged.map_partitions(lambda df: df["ID"].apply(make_packed_id, db_id=db, hac=hac), meta=("ID", int))
            # Aim for ≤15M rows per partition because this is for each HAC
            ddf_merged = ddf_merged.repartition(partition_size="800MB")
            ddf_merged = ddf_merged.astype(meta)
            # -------------------------
            # 4️⃣ Write the database
            # -------------------------
            ddf_merged.to_parquet(
                out_path / f"cleaned/{hac}",
                write_index=False,
                compute=True,
                engine="pyarrow",
                name_function=make_name_function(db_id=db, hac=hac)
                
            )
            
        end2 = time.perf_counter()
        print(f"Completed in {end2 - end:.2f} seconds")



if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()