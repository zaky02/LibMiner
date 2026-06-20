import datamol as dm
from pathlib import Path
import pandas as pd
from functools import partial
from molfeat.trans.base import MoleculeTransformer
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Sequence
from collections import defaultdict
import tables as tb


def convert_folder(hac_folders: Path | str, keep: list[str] = []) -> dict[str, list[Path]]:
    hac_folders = Path(hac_folders)
    pa = list(hac_folders.glob("*.parquet"))
    clas = defaultdict(list)
    for p in pa:
        db = p.stem.split("_")[1].split("_")[0].strip("db")
        
        if keep and db not in keep: continue

        clas[db].append(p)
    return clas


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



def stereo_expansion(smiles: list[str], n_variants: int=20, 
                     timeout_seconds: int=None, n_jobs: int=-1) -> pd.DataFrame:
    """
    Generate n_variants stereoisomers

    Parameters
    ----------
    smiles : list[str]
        The input list of SMILES strings
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

    mols = [dm.to_mol(smi) for smi in smiles]
    # generate stereoisomers
    out = dm.parallelized( 
    partial(dm.enumerate_stereoisomers, clean_it=False, n_variants=n_variants, 
            timeout_seconds=timeout_seconds),
    mols,
    progress=False,
    n_jobs=n_jobs,
    scheduler="threads",
    )
    
    smiles = {dm.to_smiles(mols[i], canonical=True, isomeric=True): 
        dm.to_df(mol, n_jobs=n_jobs, smiles_column="SMILES") for i, mol in enumerate(out)}
    
    return pd.concat(smiles)


def convert_hac_to_mw(hac: int):
    """Convert HAC to MW threshold"""
    return (hac * 12) + (2 * hac)


def convert_mw_to_hac(mw: float | None):
    """Convert MW threshold to HAC"""
    if mw is None:
        return mw
    return int(mw / 14)    


@dataclass
class Rerank:
    """Rerank search results based on calculated properties"""
    
    feature: str = "scaffoldkeys"
    n_jobs: int = 1
    
    def property_calculation(self,
                             smiles: list[str],
                             index: Sequence | None = None) -> pd.DataFrame:
        """
        Calculate properties for a list of SMILES, 
        which can be used to rerank the results from the similarity search
        """
        transformer = MoleculeTransformer(featurizer=self.feature, dtype=int, 
                                          n_jobs=self.n_jobs)
        features = transformer(smiles)
        
        return pd.DataFrame(features, columns=transformer.columns, 
                            index=smiles if index is None else index)

    def normalized_distances_to_reference(self, vectors, reference_idx=0):
        """
        1. Normalize features to comparable ranges
        2. Compute distances in normalized space
        """
        # Method A: Min-max normalization per feature
        min_vals = vectors.min(axis=0)
        max_vals = vectors.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        
        normalized = (vectors - min_vals) / range_vals
        
        # Reference in normalized space
        ref_norm = normalized[reference_idx]
        
        # Distances in normalized space
        distances = np.sqrt(np.sum((normalized - ref_norm) ** 2, axis=1))
        #ranking = np.argsort(distances)
        
        return distances
    
    def compute(self,
                smiles: list[str],
                index: Sequence | None = None,
                reference_idx: int = 0) -> pd.DataFrame:
        """
        Compute normalized distances to a reference molecule
        """
        features_df = self.property_calculation(smiles, index=index)
        vectors = features_df.values
        
        distances = self.normalized_distances_to_reference(vectors,
                                                           reference_idx=reference_idx)
    
        features_df["distances"] = distances
         
        return features_df.sort_values("distances")



def decode_fp_row(row: np.void, fp_fields: list[str]) -> np.ndarray:
    chunks = np.array([row[f] for f in fp_fields], dtype=np.uint64)
    chunks = chunks.byteswap()  # ← fix: little-endian → big-endian bytes
    return np.unpackbits(chunks.view(np.uint8), bitorder="big")


def decode_all_fps(h5_path: str) -> tuple[np.ndarray, np.ndarray]:
    with tb.open_file(h5_path, "r") as f:
        raw = f.root.fps[:]
        fp_fields = [n for n in raw.dtype.names if n not in ("fp_id", "popcnt")]

    fp_ids = raw["fp_id"]
    chunks = np.stack([raw[f] for f in fp_fields], axis=1).astype(np.uint64)
    chunks = chunks.byteswap()  # ← fix

    raw_bytes = chunks.view(np.uint8).reshape(len(raw), -1)
    fps = np.unpackbits(raw_bytes, axis=1, bitorder="big")

    return fp_ids, fps

def encode_fps_batch(
    fps: np.ndarray,
    fp_ids: np.ndarray,
    fp_fields: list[str],
) -> np.ndarray:
    n = len(fps)
    packed_bytes = np.packbits(fps.astype(np.uint8), axis=1, bitorder="big")
    chunks = packed_bytes.view(np.uint64).byteswap()  # ← fix: back to little-endian

    popcnts = fps.sum(axis=1).astype(np.int64)
    dtype = [("fp_id", "<i8")] + [(f, "<u8") for f in fp_fields] + [("popcnt", "<i8")]

    out = np.empty(n, dtype=dtype)
    out["fp_id"]  = fp_ids.astype(np.int64)
    out["popcnt"] = popcnts
    for i, f in enumerate(fp_fields):
        out[f] = chunks[:, i]

    return out