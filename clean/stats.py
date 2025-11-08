import dask.dataframe as dd
from dask.distributed import Client, performance_report
import os
from pathlib import Path
import argparse
import json
from collections import defaultdict
import dask
from itertools import islice
from itertools import combinations
import dask
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate SMILES')
    parser.add_argument('-bs', '--blocksize', type=str, help='Block size for dask dataframe. The safest is the default 64MB',  required=False, default='64MB')
    parser.add_argument('-dp','--database_path', type=str, help='The folder path for the database', required=False,
                        default='Molecular_database')
    parser.add_argument('-s','--smiles_col', type=str, help='The column name of the smiles', required=False,
                        default='SMILES')
    parser.add_argument('-b','--batch_size', type=int, help='Batch size to compute overlap', required=False,
                        default=2)
    parser.add_argument('-op','--output_parquet', type=str, help="The name for the redundant_smiles file",   required=False, default='redudant_smiles.parquet')
    
    args = parser.parse_args()
    return args.blocksize, args.database_path, args.smiles_col, args.batch_size, args.output_parquet



scheduler_address = os.environ["DASK_SCHEDULER_ADDRESS"]
client = Client(scheduler_address)    # Connect to that cluster


def compute_count(hac_folders, block_size="64MB"):
    # Read only necessary columns
    ddf_merged = dd.read_parquet(f"{hac_folders}/*.parquet", blocksize=block_size,
                                 columns=["db_id", "SMILES"])
    
    # Compute counts per db_id per partition
    def partition_counts(df):
        return df.groupby("db_id")["SMILES"].count()
    
    # this is a partition groupby
    counts_per_partition = ddf_merged.map_partitions(partition_counts).compute()
    
    # Combine counts across partitions
    return counts_per_partition.groupby("db_id").sum().to_frame().rename({"SMILES":"total_counts"}, axis=1)


def convert_folder(hac_folders: Path | str):
    hac_folders = Path(hac_folders)
    pa = list(hac_folders.glob("*.parquet"))
    clas = defaultdict(list)
    for p in pa:
        db = p.stem.split("_")[1].split("_")[0].strip("db")
        clas[db].append(p)
    return clas


def compute_internal_duplication(
    db_paths: dict[str, str],
    smiles_col: str = "SMILES",
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
            df = dd.read_parquet(path, columns=[smiles_col], blocksize=block_size)
            df_dedup = df.drop_duplicates(subset=[smiles_col])
            unique = df_dedup.map_partitions(len).sum()
            dedup_dfs[db_id] = df_dedup
            lazy_results.append(unique)

        # Compute this batch in one go
        computed_values = dask.compute(*lazy_results)
        counts.update(dict(zip([db_id for db_id, _ in batch], computed_values)))

    # Return as a tidy Series
    counts_series = pd.Series(counts, name="internal_counts")

    return counts_series, dedup_dfs


def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def compute_database_redundancy(classified_folders: dict[str, str], dedup_dfs, 
                                n:int=2, smiles_col: str ="SMILES", 
                               output: str | Path ="redundant_smiles.parquet"):
    """_summary_

    Args:
        classified_folders (dict[str, str]): _description_
        dedup_dfs (_type_): _description_
        n (int, optional): _description_. Defaults to 2.
        smiles_col (str, optional): _description_. Defaults to "SMILES".
        output (str | Path, optional): _description_. Defaults to "redundant_smiles.parquet".

    Returns:
        _type_: _description_
    """
    overlaps={}
    counts = {}
    smiles_to_dbs = defaultdict(set)
    pairs = list(combinations(classified_folders, 2))
    for batch in batched(pairs, n):  # run n bacthes at a time
        futures = []
        for db1, db2 in batch:
            overlap = dd.merge(dedup_dfs[db1], dedup_dfs[db2], on=smiles_col, how="inner")
            futures.append(overlap)
        
        results = dask.compute(*futures)
        for (db1, db2), res in zip(batch, results):
            overlaps[f"{db1}_{db2}"] = res
    
    for pair, df in overlaps.items():
        counts[pair] = df.shape[0]
        db1, db2 = pair.split("_")
        for smi in df["SMILES"]:
            smiles_to_dbs[smi].update([db1, db2])
            
    smiles_overlap_df = pd.DataFrame({
                "SMILES": list(smiles_to_dbs.keys()),
                "Databases": [",".join(sorted(list(v))) for v in smiles_to_dbs.values()]
                    })
    
    smiles_overlap_df.to_parquet(output)        
    return smiles_overlap_df, pd.Series(counts, name="redundant_count")


def main():
    block_size, database_path, smiles_col, batch_size, out_parquet = parse_args()
    with performance_report(filename="dask-stats.html"):
            
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        database_path = Path(database_path)
        output_stats = database_path/"stats"
        progress = Path("progress_stats.txt")
        progress.touch(exist_ok=True)
        hacs = sorted(database_path.glob("HAC_*"), key=lambda x: int(x.name.split("_")[-1]))
        
        print(f"start count from: {database_path}")
            
        for hac_folders in hacs:
            hac = hac_folders.name.split("_")[-1]

            if f"HAC {hac} done" in progress.read_text():
                print(f"HAC {hac} already done, skipping.")     
                continue
            
            (output_stats/hac).mkdir(parents=True, exist_ok=True)
            out_parq = output_stats / hac / out_parquet 
               
            sta = compute_count(hac_folders, block_size)
            
            classified_folders = convert_folder(hac_folders)
            internal_counts, dedup_dfs = compute_internal_duplication(classified_folders, smiles_col, block_size, batch_size)
            redundante_smiles, redundant_counts = compute_database_redundancy(classified_folders, dedup_dfs,
                                                                              batch_size, smiles_col, out_parq)
            
            pd.concat([sta, internal_counts], axis=1).to_csv(output_stats/hac/"after_before_counts.csv")
            redundant_counts.to_csv(output_stats/hac/"overlaping_counts.csv")

            with open(progress, "a") as f:
                f.write(f"HAC {hac} done\n")
        
if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()