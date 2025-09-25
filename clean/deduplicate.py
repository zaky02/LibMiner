from pathlib import Path
import os
import time
import dask.dataframe as dd
import dask
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from dask import delayed
from dask.distributed import Client, performance_report, wait, as_completed


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
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        rdMolStandardize.CleanupInPlace(mol)  # 
        # Apply salt removal only if multiple fragments
        if "." in smi:
            rdMolStandardize.LargestFragmentChooser().chooseInPlace(mol)
        # Apply uncharging only if charges are present
        if "+" in smi or "-" in smi:
            rdMolStandardize.Uncharger().unchargeInPlace(mol)       # ensure valid again after modifications

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
    return mol.GetNumHeavyAtoms()

def rename_partitions(output_folder: Path | str, hac: int):
    """Rename partitions to HAC_xx_01.parquet, HAC_xx_02.parquet, etc.
    
    Parameters
    ----------
    hac : int
        HAC value
    output_folder : Path
        Output folder path
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(f"{output_folder}/HAC_{hac:02d}").glob("*.parquet"))
    for i, f in enumerate(files, start=1):
        new_name = output_folder / f"HAC_{hac:02d}/HAC{hac:02d}_{i:02d}.parquet"
        f.rename(new_name)
        
# -------------------------
# 2️⃣ Read databases lazily
# -------------------------
start = time.perf_counter()
with performance_report(filename="dask-clean.html"):
    db_files = {
        "001": "Enamine/*.csv",
        "002": "ZINC/*.csv",
        "003": "Savi/*.csv",
        "004": "PubChem/*.csv",
        "005": "ChEMBL/*.csv",
        "006": "DrugBank/*.csv",
        "007": "Coconut/*.csv",
        "008": "NPAtlas/*.csv",
        "009": "ChemistriX/*.csv",
        "010": "CHIPMUNK/*.csv",
        "011": "SCUBIDOO/*.csv",
        "012": "SureChEMBL/*.csv",
    }

    processed_paths = []

    # Batch size can match #workers if desired, but each DB is processed fully partitioned
    for db_id, pattern in db_files.items():
        print(f"Processing DB {db_id} ...")
        out_path = Path(f"Deduplicated/cleaned_{db_id}/")
        out_path_missing = Path(f"Deduplicated/missing_{db_id}/")
        out_path.mkdir(parents=True, exist_ok=True)
        out_path_missing.mkdir(parents=True, exist_ok=True)
        if len(list(out_path.glob("*.parquet"))) > 0:
            print(f"  Skipping {db_id}, already processed.")
            processed_paths.append(out_path)
            continue
        # Read CSV lazily with small blocks
        ddf = dd.read_csv(pattern, blocksize="512MB")
        ddf["db_id"] = db_id

        # Normalize and deduplicate
        ddf["SMILES"] = ddf["SMILES"].map(normalize_smiles, meta=("SMILES", str))
        ddf = ddf.dropna(subset=["SMILES"])
        ddf_missing = ddf[ddf["SMILES"].isna()]
        ddf = ddf.drop_duplicates(subset="SMILES")

        # Submit writing task asynchronously, each partition uses a different worker
        future = client.compute(ddf.to_parquet(out_path, write_index=False, compute=False))
        future_missing = client.compute(
        ddf_missing.to_parquet(out_path_missing, write_index=False, compute=False)
        )
        # Wait until this DB is done before moving to the next
        wait([future, future_missing])
        future.release()  # free memory on workers
        future_missing.release()
        processed_paths.append(out_path)   

end = time.perf_counter()
print(f"Initial cleaning completed in {end - start:.2f} seconds")       
# -------------------------
# 4️⃣ Merge and deduplicate
# -------------------------
with performance_report(filename="dask-deduplicate.html"):
    # After writing, read all processed DBs lazily
    ddf_merged = dd.read_parquet([f"{p}/*.parquet" for p in processed_paths])

    # Deduplicate across all sources using normalized SMILES
    ddf_merged = ddf_merged.drop_duplicates(subset="SMILES")

    ddf_merged["HAC"] = ddf_merged["SMILES"].map(get_hac, meta=("HAC", int))

    # -------------------------
    # 4️⃣ Write the database
    # -------------------------
    # Lazy group by HAC

    min_HAC = ddf_merged.HAC.min().compute()
    max_HAC = ddf_merged.HAC.max().compute()
    output_folder = Path("molecular_database/")
    for hac in range(min_HAC, max_HAC + 1):
        ddf_hac = ddf_merged[ddf_merged.HAC == hac]
        ddf_hac = ddf_hac.reset_index(drop=True)
        ddf_hac = ddf_hac.drop(columns=["HAC"])
        
        # Write partitioned into smaller chunks (≤15M rows)
        n_parts = int(len(ddf_hac) / 15e6)
        ddf_hac = ddf_hac.repartition(npartitions=n_parts)
        # Write to Parquet asynchronously
        out_path = output_folder / f"HAC_{hac:02d}/"
        out_path.mkdir(exist_ok=True)
        future = client.compute(ddf_hac.to_parquet(out_path, write_index=False, compute=False))
        
        # Wait and release to free memory
        wait(future)
        future.release()
    
        print(f"HAC {hac} processed at {out_path}") 
        rename_partitions(output_folder, hac)

end2 = time.perf_counter()
print(f"Completed in {end2 - end:.2f} seconds")

client.shutdown()