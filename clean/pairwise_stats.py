import dask.dataframe as dd
from dask.distributed import Client, performance_report
import os
from pathlib import Path
import argparse
from collections import defaultdict
from itertools import combinations
import dask
import pandas as pd
import shutil
import logging
import gc
from typing import Sequence
from utils import convert_folder


def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate SMILES')
    parser.add_argument('-bs', '--blocksize', type=str, help='Block size for dask dataframe. The safest is the default 64MB',  required=False, default='64MB')
    parser.add_argument('-dp','--database_path', type=str, help='The folder path for the database', required=False,
                        default='Molecular_database')
    parser.add_argument('-s','--smiles_col', type=str, help='The column name of the smiles', required=False,
                        default='SMILES')
    parser.add_argument('-op','--output_parquet', type=str, help="The name for the redundant_smiles file",   required=False, default='redudant_smiles.parquet')
    parser.add_argument('-id','--id_col', type=str, help='The column name for the original IDs from the databases', required=False, default='ID')
    parser.add_argument("-ld", "--large_dbs", nargs='*', default=["001", "002", "014", "003"],
                        help="List of database IDs considered large for on-disk processing.")
    args = parser.parse_args()
    return args.blocksize, args.database_path, args.smiles_col, args.output_parquet, args.id_col, args.large_dbs


scheduler_address = os.environ["DASK_SCHEDULER_ADDRESS"]
client = Client(scheduler_address)    # Connect to that cluster
client.wait_for_workers(n_workers=1, timeout=180)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_count(hac_folders, smiles_col="SMILES", block_size="64MB"):
    # Read only necessary columns
    ddf_merged = dd.read_parquet(f"{hac_folders}/*.parquet", blocksize=block_size,
                                 columns=["db_id", smiles_col])
    
    # Compute counts per db_id per partition
    def partition_counts(df):
        return df.groupby("db_id")[smiles_col].count()
    
    # this is a partition groupby
    counts_per_partition = ddf_merged.map_partitions(partition_counts)
    final_count = counts_per_partition.groupby("db_id").sum().to_frame().rename(columns={smiles_col: "dropna_counts"}).compute()

    del ddf_merged
    client.run(gc.collect)

    return final_count


def compute_internal_duplication(
    db_paths: dict[str, str],
    smiles_cols: str = "SMILES",
    block_size: str = "64MB",
    batch_size: int = 2,
    id_cols: str = "ID"
):
    """
    Compute internal deduplication statistics for many databases in smaller batches.
    Returns a pandas Series of unique counts
    """
    counts = {}

    # Convert items to a list so we can slice batches
    db_items = list(db_paths.items())

    for i in range(0, len(db_items), batch_size):
        batch = db_items[i : i + batch_size]
        lazy_results = []

        # Build each batch of Dask operations
        for db_id, path in batch:
            df = dd.read_parquet(path, columns=[smiles_cols, id_cols], blocksize=block_size)
            # El drop ID nunca debe hacerse en el pairwise
            df_dedup = df.drop_duplicates(subset=smiles_cols)
            unique = df_dedup.map_partitions(len).sum()
            #dedup_dfs[db_id] = df_dedup
            lazy_results.append(unique)

        # Compute this batch in one go
        computed_values = dask.compute(*lazy_results)
        counts.update(dict(zip([db_id for db_id, _ in batch], computed_values)))

    return pd.Series(counts, name="after_internal_deduplication")


def get_overlap_by_merge(db1: str, db2: str, 
                        db_paths: dict[str, str], 
                        smiles_col: str ="SMILES",
                        id_cols: str = "ID",
                        block_size: str = "64MB",
                        large_dbs: Sequence[str] = ("001", "002", "014", "003"),
                        on_disk: bool=False):
    
    out_dir =  Path(f"tmp/{db1}_{db2}")
    
    if out_dir.exists():
        return out_dir
    
    df1 = dd.read_parquet(db_paths[db1], columns=[smiles_col, id_cols], blocksize=block_size).drop_duplicates(subset=smiles_col)
    df2 = dd.read_parquet(db_paths[db2], columns=[smiles_col, id_cols], blocksize=block_size).drop_duplicates(subset=smiles_col)

    # Use the merge and let Dask manage the partitioning/shuffle
    # If len(df1) and len(df2) are both large, this is the most Dask-idiomatic way.
    # The 'on_disk' flag already handles the memory spill.
    
    shuffle_method = "disk" if on_disk else "tasks"
    overlap = dd.merge(df1, df2, on=smiles_col, how="inner", shuffle_method=shuffle_method)
    
    if on_disk and (db1 in large_dbs or db2 in large_dbs):
        overlap.to_parquet(out_dir, write_index=False, compute=True)
        return out_dir
    del df1, df2
    client.run(gc.collect)
    return overlap # Returns a Dask DataFrame, not computed!


def get_overlapping_databases(
    db_paths: dict[str, str],
    smiles_col: str ="SMILES",
    id_cols: str = "ID",
    block_size: str = "64MB",
    large_dbs: Sequence[str] = ("001", "002", "014", "003"),
    on_disk: bool=False
    ):
    
    overlaps={}
    pairs = [sorted(x) for x in combinations(db_paths.keys(), 2)]
    for db1, db2 in pairs:  # run n bacthes at a time
        overlap = get_overlap_by_merge(db1, db2, db_paths, smiles_col, id_cols, block_size, large_dbs, on_disk)
        overlaps[f"{db1}_{db2}"] = overlap
    
    return overlaps


def count_reundancy(
    overlaps: dict[str, dd.DataFrame | pd.DataFrame | Path],
    smiles_col: str = "SMILES"
    ):
    
    overlap_counts = {}
    smiles_to_dbs = defaultdict(set)  
    for pair, df in overlaps.items():
        db1, db2 = pair.split("_")
        if isinstance(df, Path):
            df = dd.read_parquet(df, columns=[smiles_col])                 
        sm = df[smiles_col].compute()
        overlap_counts[pair] = len(sm)
        for smi in sm:
            smiles_to_dbs[smi].update([db1, db2])
        del sm, df
        
    client.run(gc.collect)
    
    return smiles_to_dbs, overlap_counts


def save_redundancy(
    overlaps: dict[str, dd.DataFrame | pd.DataFrame | Path],
    smiles_col: str ="SMILES",
    output: str | Path ="redundant_smiles.parquet"):
    """
    Save the redundancy information to a parquet file and return the counts as a pandas Series.
    """
    smiles_to_dbs, counts = count_reundancy(overlaps, smiles_col)
    smiles_overlap_df = pd.DataFrame({
                "SMILES": list(smiles_to_dbs.keys()),
                "Databases": [",".join(sorted(list(v))) for v in smiles_to_dbs.values()]
                    })
    
    smiles_overlap_df.to_parquet(output)
            
    return pd.Series(counts, name="redundant_count")


def main():
    block_size, database_path, smiles_col, out_parquet, id_col, large_dbs = parse_args()
    
    with performance_report(filename="dask-pairwise.html"):
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        database_path = Path(database_path)
        output_stats = database_path/"pairwise_stats"
        progress = Path("progress_pairwise.txt")
        progress.touch(exist_ok=True)
        hacs = sorted(database_path.glob("HAC_*"), key=lambda x: int(x.name.split("_")[-1]))
        
        logger.info(f"start pairwise deduplication: {database_path}")
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
                
            (output_stats/hac).mkdir(parents=True, exist_ok=True)
            out_parq = output_stats / hac / out_parquet 
            logger.info(f"counting stats {hac}")
            sta = compute_count(hac_folders, smiles_col, block_size)
            
            classified_folders = convert_folder(hac_folders)
            logger.info(f"computing internal stats {hac}")
            internal_counts = compute_internal_duplication(classified_folders, smiles_col,  block_size, 
                                                                      batch_size, id_col)
            
            logger.info(f"computing database redundancy {hac}")
            overlaps = get_overlapping_databases(classified_folders, smiles_col, id_col, block_size, large_dbs, on_disk)

            logger.info(f"Save database redundancy {hac}")
            redundant_counts = save_redundancy(overlaps, smiles_col, out_parq)
            
            overlaps.clear()
            # save to disk the results
            pd.concat([sta, internal_counts], axis=1).to_csv(output_stats/hac/"internal_duplication.csv")
            redundant_counts.to_csv(output_stats/hac/"pairwise_duplication.csv")

            with open(progress, "a") as f:
                f.write(f"HAC {hac} done\n")
            
            if Path(f"tmp").exists():
                shutil.rmtree(f"tmp", ignore_errors=True)
                
        for n in ["pairwise_duplication.csv", "internal_duplication.csv"]:
            files = Path(output_stats).glob(f"*/{n}")
            pd.concat({f.parents.name: pd.read_csv(f, index_col=0) for f in files}).to_csv(output_stats/f"total_{n}")
        
if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()