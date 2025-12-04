import datamol as dm
from pathlib import Path
import pandas as pd


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



def stereo_expansion(df: pd.DataFrame, n_variants: int=20, 
                     timeout_seconds: int=None, n_jobs: int=-1) -> pd.DataFrame:
    """
    Generate n_variants stereoisomers

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing SMILES
    n_variants : int, optional
        The number of stereoisomers, by default 20
    timeout_seconds : int, optional
        the number of seconds, by default None
    n_jobs : int, optional
        the number of jobs, by default -1 which uses all cores
    
    Returns
    -------
    pd.DataFrame
        The dataframe with expanded stereoisomers
    """

    mols = dm.from_df(df, smiles_column="SMILES") 
    inputs = [{"mol": mol, "clean_it":False, "n_variants":n_variants, 
               "timeout_seconds":timeout_seconds} for mol in mols]
    
    # generate stereoisomers
    out = dm.parallelized( 
    dm.enumerate_stereoisomers,
    inputs,
    progress=False,
    n_jobs=n_jobs,
    scheduler="threads",
    arg_type="kwargs",
    )
    
    smiles = {dm.to_smiles(mols[i], canonical=True, isomeric=True): 
        dm.to_df(mol, n_jobs=n_jobs, smiles_column="SMILES") for i, mol in enumerate(out)}
    
    return pd.concat(smiles)
