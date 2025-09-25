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

def write_db_by_hac(db_id: str, pattern: str, output_folder: Path, output_folder_missing: Path):
    """
    Process one database: normalize SMILES, deduplicate locally,
    split by HAC, and append to HAC-specific Parquet folders.
    Parameters
    ----------
    db_id : str
        Database ID
    pattern : str
        Path to CSV file patterns
    output_folder : Path
        Output folder path    
    output_folder_missing : Path
        Output folder for incorrect SMILES
    """
    ddf = dd.read_csv(pattern, blocksize="512MB")
    ddf["db_id"] = db_id

    # Normalize
    ddf["SMILES"] = ddf["SMILES"].map(normalize_smiles, meta=("SMILES", str))

    # Separate missing SMILES
    ddf_missing = ddf[ddf["SMILES"].isna()]
    ddf = ddf.dropna(subset=["SMILES"])

    # Deduplicate within this DB only
    ddf = ddf.drop_duplicates(subset="SMILES")

    # HAC calculation
    ddf["HAC"] = ddf["SMILES"].map(get_hac, meta=("HAC", int))

    # Write missing SMILES
    ddf_missing.to_parquet(
        output_folder_missing / f"{db_id}/",
        write_index=False,
        compute=True,
        append=True,
    )


    # Partition and write by HAC
    for hac in range(min_HAC-1, max_HAC + 1):
        out_path = output_folder / f"HAC_{hac:02d}/"
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Filter by HAC
        if hac < min_HAC:
            ddf_hac = ddf[ddf.HAC < min_HAC]
        elif hac == max_HAC:
            ddf_hac = ddf[ddf.HAC > max_HAC]
        else:
            ddf_hac = ddf[ddf.HAC == hac]

        if len(ddf_hac) == 0:
            continue

        # Repartition into manageable chunks (optional)
        n_parts = max(1, int(len(ddf_hac) / 1e6))  # Aim for ≤1M rows per partition because this is for each db
        ddf_hac = ddf_hac.repartition(npartitions=n_parts)

        # Append to existing Parquet files
        ddf_hac.to_parquet(
            out_path,
            write_index=False,
            compute=True,
            append=True,
        )

        rename_partitions(output_folder, hac)
        print(f"DB {db_id} → HAC {hac} written to {out_path}")
       
# -------------------------
# 2️⃣ Read databases lazily
# -------------------------
start = time.perf_counter()

min_HAC = 5
max_HAC = 45

with performance_report(filename="dask-HAC.html"):
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

    # Batch size can match #workers if desired, but each DB is processed fully partitioned
    out_path = Path("Molecular_database")
    out_path_missing = Path("Molecular_database/missing/")
    for db_id, pattern in db_files.items():
        print(f"Processing DB {db_id} ...")
    
        # Skip if already processed
        if (out_path_missing / db_id).exists():
            print(f"  Skipping {db_id}, already processed.")
            continue

        write_db_by_hac(db_id, pattern, out_path, out_path_missing)

end = time.perf_counter()
print(f"Initial cleaning completed in {end - start:.2f} seconds")       
# -------------------------
# 4️⃣ Merge and deduplicate
# -------------------------
with performance_report(filename="dask-deduplicate.html"):
    # After writing, read all processed DBs lazily

    for hac in range(min_HAC-1, max_HAC + 1):
        ddf_merged = dd.read_parquet([f"{out_path / f'HAC_{hac:02d}/'}*.parquet"], chunksize="512MB", 
                                     columns=["ID","SMILES"])

        # Deduplicate across all sources using normalized SMILES
        ddf_merged = ddf_merged.drop_duplicates(subset="SMILES")
        
        
        # -------------------------
        # 4️⃣ Write the database
        # -------------------------
        ddf_merged.to_parquet(
            out_path / f"cleaned/HAC_{hac:02d}",
            write_index=False,
            compute=True,
            append=True,
        )
end2 = time.perf_counter()
print(f"Completed in {end2 - end:.2f} seconds")

client.shutdown()