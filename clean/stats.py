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

    args = parser.parse_args()
    return args.blocksize, args.database_path



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
    return counts_per_partition.groupby("db_id").sum().to_dict()


def convert_folder(hac_folders: Path | str):
    hac_folders = Path(hac_folders)
    pa = list(hac_folders.glob("*.parquet"))
    clas = defaultdict(list)
    for p in pa:
        db = p.stem.split("_")[1].split("_")[0].strip("db")
        clas[db].append(p)
    return clas


def compute_internal_duplication(db_paths: dict[str, str], smiles_col: str = "SMILES"):

    dedup_dfs = {}
    lazy_results = []

    # Build all lazy computations first
    for db_id, path in db_paths.items():
        df = dd.read_parquet(path, columns=[smiles_col])
        
        df_dedup = df.drop_duplicates(subset=[smiles_col])
        unique = df_dedup.map_partitions(len).sum()
        dedup_dfs[db_id] = df_dedup

        # Collect both lazy results for single batch compute
        lazy_results.extend(unique)

    # Compute all totals and uniques at once
    computed_values = dask.compute(*lazy_results)

    # Assign results back in the same order
    counts = dict(zip(db_paths.keys(), computed_values))

    return counts, dedup_dfs


def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def compute_database_redundancy(classified_folders, dedup_dfs, n=5, smiles_col="SMILES", 
                               output="redundant_smiles.parquet"):
    
    overlaps={}
    counts = {}
    smiles_to_dbs = defaultdict(set)
    pairs = list(combinations(classified_folders, 2))
    for batch in batched(pairs, n):  # run 3 at a time
        futures = []
        for db1, db2 in batch:
            overlap = dd.merge(dedup_dfs[db1], dedup_dfs[db2], on=smiles_col, how="inner")
            futures.append(overlap)
        
        results = dask.compute(*futures)
        for (db1, db2), res in zip(batch, results):
            overlaps[f"{db1}_{db2}"] = res
    
    for pair, df in overlaps.items():
        counts[pairs] = df.shape[0]
        db1, db2 = pair.split("_")
        for smi in df["SMILES"]:
            smiles_to_dbs[smi].update([db1, db2])
            
    smiles_overlap_df = pd.DataFrame({
                "SMILES": list(smiles_to_dbs.keys()),
                "Databases": [",".join(sorted(list(v))) for v in smiles_to_dbs.values()]
                    })
    
    smiles_overlap_df.to_parquet(output)        
    return smiles_overlap_df, counts


def main():
    block_size, database_path = parse_args()
    with performance_report(filename="dask-stats.html"):
            
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        database_path = Path(database_path)
        database_path.mkdir(parents=True, exist_ok=True)
        progress = Path("progress_stats.txt")
        progress.touch(exist_ok=True)
        stats = database_path/"hac_stats.txt"
        stats.touch(exist_ok=True)
        hacs = sorted(database_path.glob("HAC*"), key=lambda x: int(x.name.split("_")[-1]))
        
        with open(stats, "a") as w:
            w.write(f"start count from: {database_path}\n")
            
        for hac_folders in hacs:
            hac = hac_folders.name.split("_")[-1]
            if f"HAC {hac} before" in progress.read_text():
                print(f"HAC {hac} already done, skipping.")     
                continue    
            sta = compute_count(hac_folders, block_size)
            
            classified_folders = convert_folder(hac_folders)
            internal_counts, dedup_dfs = compute_internal_duplication(classified_folders)
            redundante_smiles, redundant_counts = compute_database_redundancy(classified_folders, dedup_dfs)
            
            with open(stats, "a") as w:
                w.write(f"AC {hac} # {json.dumps(sta)}\n")
            
            with open(progress, "a") as f:
                f.write(f"HAC {hac} done\n")
        
if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()