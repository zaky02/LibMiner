import dask.dataframe as dd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from dask import delayed, compute
from pathlib import Path

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
        # Apply salt removal only if multiple fragments
        if "." in smi:
            lfs = rdMolStandardize.LargestFragmentChooser()
            mol = lfs.choose(mol)

        # Apply uncharging only if charges are present
        if "+" in smi or "-" in smi:
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)
        
        if "." in smi or "+" in smi or "-" in smi:    
            Chem.SanitizeMol(mol)          # ensure valid again after modifications

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


# -------------------------
# 2️⃣ Read databases lazily
# -------------------------
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

ddfs = {}
for db_id, path in db_files.items():
    ddfs[db_id] = dd.read_csv(path)
    ddfs[db_id]["db_id"] = db_id  # add provenance

# -------------------------
# 3️⃣ Wrap processing per DB
# -------------------------
@delayed
def process_db(ddf: dd.DataFrame, db_id: int) -> str:
    """Process and write a single database
    
    Parameters
    ----------
    ddf : dd.DataFrame
        Dask DataFrame of the database
    db_id : int
        Database identifier
    """
    # Normalize SMILES
    ddf["SMILES"] = ddf["SMILES"].map(normalize_smiles, meta=("SMILES", str))
    ddf = ddf.dropna(subset=["SMILES"])
    ddf = ddf.drop_duplicates(subset="SMILES")  # intra-DB dedup
    ddf["HAC"] = ddf["SMILES"].map(get_hac, meta=("HAC", int))
    out_path = f"cleaned_{db_id}/"
    ddf.to_parquet(out_path, write_index=False)
    return out_path

tasks = [process_db(ddf, db_id) for db_id, ddf in ddfs.items()]

# Trigger all DBs to process concurrently
compute(*tasks)

# -------------------------
# 4️⃣ Merge and deduplicate
# -------------------------
# After writing, read all processed DBs lazily
ddf_merged = dd.read_parquet("processed_*/")

# Deduplicate across all sources using normalized canonical SMILES
ddf_merged = ddf_merged.drop_duplicates(subset="canonical_smiles")

# -------------------------
# 4️⃣ Write the database
# -------------------------
# Lazy group by HAC
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

min_HAC = ddf_merged.HAC.min().compute()
max_HAC = ddf_merged.HAC.max().compute()
output_folder = Path("molecular_database/")
for hac in range(min_HAC, max_HAC + 1):
    ddf_hac = ddf_merged[ddf_merged.HAC == hac]
    ddf_hac = ddf_hac.reset_index(drop=True)

    # Write partitioned into smaller chunks (≤15M rows)
    n_parts = int(len(ddf_hac) / 15e6)
    ddf_hac.repartition(npartitions=n_parts).to_parquet(
        output_folder / f"HAC_{hac:02d}/",
        write_index=False
    )

    rename_partitions(output_folder, hac)

