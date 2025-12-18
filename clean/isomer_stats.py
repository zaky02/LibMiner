import dask.dataframe as dd
from dask.distributed import Client, performance_report
import os
from pathlib import Path
import argparse
import dask
import pandas as pd
import logging
from utils import convert_folder
from itertools import combinations


def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate SMILES')
    parser.add_argument('-bs', '--blocksize', type=str, help='Block size for dask dataframe. The safest is the default 64MB',  required=False, default='64MB')
    parser.add_argument('-dp','--database_path', type=str, help='The folder path for the database', required=False,
                        default='Molecular_database')
    parser.add_argument('-n','--nostereo', type=str, help='The column name of the canonical smiles', required=False,
                        default='nostereo_SMILES')
    
    args = parser.parse_args()
    return args.blocksize, args.database_path, args.nostereo


scheduler_address = os.environ["DASK_SCHEDULER_ADDRESS"]
client = Client(scheduler_address)    # Connect to that cluster
client.wait_for_workers(n_workers=1, timeout=180)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def compute_internal_duplication(
    db_paths: dict[str, str],
    smiles_cols: str = "canonical_SMILES",
    block_size: str = "64MB",
    batch_size: int = 2,
):
    """
    Compute internal deduplication statistics for many databases in smaller batches.
    Returns a pandas Series of unique counts and a dict of deduplicated Dask DataFrames.
    """
    dedup_dfs = {}
    counts = {}

    # Convert items to a list so we can slice batches
    db_items = list(db_paths.items())

    for i in range(0, len(db_items), batch_size):
        batch = db_items[i : i + batch_size]
        lazy_results = []

        # Build each batch of Dask operations
        for db_id, path in batch:
            df = dd.read_parquet(path, columns=smiles_cols, blocksize=block_size)
            # El drop ID nunca debe hacerse en el pairwise
            df_dedup = df.drop_duplicates(subset=smiles_cols)
            unique = df_dedup.map_partitions(len).sum()
            dedup_dfs[db_id] = df_dedup
            lazy_results.append(unique)

        # Compute this batch in one go
        computed_values = dask.compute(*lazy_results)
        counts.update(dict(zip([db_id for db_id, _ in batch], computed_values)))

    return pd.Series(counts, name="after_internal_deduplication"), dedup_dfs


def get_overlap_by_merge(db1: str, db2: str, 
                        dedup_dfs: dict[str, dd.DataFrame], 
                        smiles_col: str ="nostereo_SMILES", 
                        on_disk: bool=False):

    df1 = dedup_dfs[db1]
    df2 = dedup_dfs[db2]

    shuffle_method = "disk" if on_disk else "tasks"
    overlap = dd.merge(df1, df2, on=smiles_col, how="inner", shuffle_method=shuffle_method)
    
    return overlap.map_partitions(len).sum() # Returns a Dask DataFrame, not computed!


def get_pairwise_overlaps(
    dedup_dfs: dict[str, dd.DataFrame],
    smiles_col: str ="nostereo_SMILES",
    on_disk: bool=False
    ):
    
    overlaps={}
    pairs = [sorted(x) for x in combinations(dedup_dfs.keys(), 2)]
    for db1, db2 in pairs:  # run n bacthes at a time
        overlap = get_overlap_by_merge(db1, db2, dedup_dfs, smiles_col, on_disk)
        overlaps[f"{db1}_{db2}"] = overlap
    
    return pd.Series(overlaps, name="after_pairswise_deduplication")


def main():
    block_size, database_path, smiles_col = parse_args()
    
    with performance_report(filename="dask-isomer.html"):
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        database_path = Path(database_path)
        output_stats = database_path/"isomer_duplicates"
        progress = Path("progress_isomers.txt")
        progress.touch(exist_ok=True)
        hacs = sorted(database_path.glob("HAC_*"), key=lambda x: int(x.name.split("_")[-1]))
        
        logger.info(f"start isomer deduplication: {database_path}")
        size_limit = 50 * (1024 ** 3)   
        for hac_folders in hacs:
            hac = hac_folders.name.split("_")[-1]
        
            if f"HAC {hac} done" in progress.read_text():
                print(f"HAC {hac} already done, skipping.")     
                continue
            
            ## look at the file size to decide if on disk or not
            file_sizes = sum([p.stat().st_size for p in hac_folders.glob("*.parquet")])
            batch_size = 4
            on_disk = False
            if file_sizes >= size_limit:
                batch_size = 2
                on_disk=True
                
            out_parq = output_stats / hac 
            out_parq.mkdir(parents=True, exist_ok=True)
            
            classified_folders = convert_folder(hac_folders)
            logger.info(f"computing internal stats {hac}")
            internal_counts, dedup_dfs = compute_internal_duplication(classified_folders, smiles_col, block_size, 
                                                                      batch_size)
            
            internal_counts
            
            logger.info(f"computing database redundancy {hac}")
            overlaps = get_pairwise_overlaps(dedup_dfs, smiles_col, on_disk)
            dedup_dfs.clear()

            internal_counts.to_csv(output_stats/hac/"internal_duplication.csv")
            overlaps.to_csv(output_stats/hac/"pairwise_duplication.csv")

            with open(progress, "a") as f:
                f.write(f"HAC {hac} done\n")
    
        for n in ["pairwise_duplication.csv", "internal_duplication.csv"]:
            files = Path(output_stats).glob(f"*/{n}")
            pd.concat({f.parents.name: pd.read_csv(f, index_col=0) for f in files}).to_csv(output_stats/f"total_{n}")     
            
if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()