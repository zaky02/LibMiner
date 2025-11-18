import dask.dataframe as dd
from dask.distributed import Client, performance_report
import os
from pathlib import Path
import argparse
from collections import defaultdict
import dask
from itertools import islice
from itertools import combinations
import dask
import pandas as pd
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate SMILES')
    parser.add_argument('-bs', '--blocksize', type=str, help='Block size for dask dataframe. The safest is the default 64MB',  required=False, default='64MB')
    parser.add_argument('-dp','--database_path', type=str, help='The folder path for the database', required=False,
                        default='Molecular_database')
    parser.add_argument('-s','--smiles_col', type=str, help='The column name of the smiles', required=False,
                        default='SMILES')
    parser.add_argument('-op','--output_parquet', type=str, help="The name for the redundant_smiles file",   required=False, default='redudant_smiles.parquet')
    parser.add_argument('-sm','--small_threshold', type=int, 
                        help='The threshold to consider a dataframe small', required=False,
                        default=1_000_000)
    
    args = parser.parse_args()
    return args.blocksize, args.database_path, args.smiles_col, args.output_parquet, args.small_threshold



scheduler_address = os.environ["DASK_SCHEDULER_ADDRESS"]
client = Client(scheduler_address)    # Connect to that cluster
client.wait_for_workers(n_workers=1, timeout=180)


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
    small_threshold: int=1_000_000
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
        
    # leave the small databases computed    
    for db_id, unique_count in counts.items():
        if unique_count <= small_threshold:
            dedup_dfs[db_id] = dedup_dfs[db_id].compute()

    return pd.Series(counts, name="internal_counts"), dedup_dfs


def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def get_overlap(db1: str, db2: str, 
                dedup_dfs: dict[str, dd.DataFrame], 
                counts: dict[str, int], 
                hac: str,
                smiles_col: str ="SMILES", 
                small_threshold: int=1_000_000):

    df1 = dedup_dfs[db1]
    df2 = dedup_dfs[db2]

    len1 = counts[db1]
    len2 = counts[db2]

    # choose smaller df
    if len1 <= len2:
        small, big = df1, df2
    else:
        small, big = df2, df1

    # CASE 1 - small fits fully in memory  best performance, no shuffle
    if min(len1, len2) <= small_threshold:
        small_list = small[smiles_col].to_list()
        return big[big[smiles_col].isin(small_list)]

    # CASE 2 - both large chunked approach without shuffle
    parts = max(int(min(len1, len2) / small_threshold), 1)
    small = small.repartition(npartitions=parts)

    #overlaps = []
    for i, sma_part in enumerate(small.to_delayed()):
        # convert each small partition to pandas list (critical!)
        sma_list = sma_part[smiles_col].compute().tolist()
        # Safe: partition-local isin, NO shuffle
        part_overlap = big[big[smiles_col].isin(sma_list)]
        out_path = f"tmp/{hac}/{db1}_{db2}/part_{i}"
        part_overlap.to_parquet(out_path, write_index=False)
        #overlaps.append(part_overlap)

    return dd.read_parquet(f"tmp/{hac}/{db1}_{db2}/part_*/*.parquet")


def get_overlapping_databases(
    dedup_dfs: dict[str, dd.DataFrame],
    counts: dict[str, int], 
    hac: str,
    n:int=2, smiles_col: str ="SMILES",
    small_threshold=1_000_000
    ):
    
    output_dir = Path("tmp") / hac
    output_dir.mkdir(exist_ok=True, parents=True)
    overlaps={}
    pairs = list(combinations(dedup_dfs.keys(), 2))
    for batch in batched(pairs, n):  # run n bacthes at a time
        futures = []
        for db1, db2 in batch:
            overlap = get_overlap(db1, db2, dedup_dfs, counts, hac, smiles_col, 
                                  small_threshold)
            futures.append(overlap)
            
        #results = dask.compute(*futures)
        for (db1, db2), res in zip(batch, futures):
            overlaps[f"{db1}_{db2}"] = res

    return overlaps


def count_reundancy(
    overlaps: dict[str, dd.DataFrame | pd.DataFrame],
    counts: dict[str, int],
    smiles_col: str = "SMILES",
    small_threshold=1_000_000
    ):
    
    overlap_counts = {}
    smiles_to_dbs = defaultdict(set)  
    for pair, df in overlaps.items():
        db1, db2 = pair.split("_")
        if all(i <= small_threshold for i in [counts[u] for u in [db1, db2]]):
            overlap_counts[pair] = df.shape[0]
            sm = df[smiles_col]
        else:
            overlap_counts[pair] = df.map_partitions(len).sum().compute()
            sm = df[smiles_col].compute()
            
        for smi in sm:
            smiles_to_dbs[smi].update([db1, db2])
        del sm
    return smiles_to_dbs, overlap_counts

def count_redundancy_ondisk( 
    smiles_col: str,
    output_dir = Path("tmp")):
    
    counts = {}
    smiles_to_dbs = defaultdict(set)  
    for f in output_dir.glob("*.parquet"):
        db1, db2 = f.stem.split("_")
        df = pd.read_parquet(f, columns=[smiles_col])
        counts[f.stem] = len(df)
        for smi in df[smiles_col]:
            smiles_to_dbs[smi].update([db1, db2])
        del df  # free memory
    shutil.rmtree(output_dir)
    return smiles_to_dbs, counts

def save_redundancy(
    overlaps: dict[str, dd.DataFrame | pd.DataFrame],
    counts: dict[str, int],
    smiles_col: str ="SMILES",
    output: str | Path ="redundant_smiles.parquet",
    small_threshold=1_000_000): 
    
    smiles_to_dbs, counts = count_reundancy(overlaps, counts, smiles_col, small_threshold)
    
    smiles_overlap_df = pd.DataFrame({
                "SMILES": list(smiles_to_dbs.keys()),
                "Databases": [",".join(sorted(list(v))) for v in smiles_to_dbs.values()]
                    })
    
    smiles_overlap_df.to_parquet(output)        
    return smiles_overlap_df, pd.Series(counts, name="redundant_count")


def main():
    block_size, database_path, smiles_col, out_parquet, small_threshold = parse_args()
    
    with performance_report(filename="dask-stats.html"):
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        database_path = Path(database_path)
        output_stats = database_path/"stats"
        progress = Path("progress_stats.txt")
        progress.touch(exist_ok=True)
        hacs = sorted(database_path.glob("HAC_*"), key=lambda x: int(x.name.split("_")[-1]))
        
        print(f"start count from: {database_path}")
        size_limit = 50 * (1024 ** 3)   
        for hac_folders in hacs:
            hac = hac_folders.name.split("_")[-1]
        
            if f"HAC {hac} done" in progress.read_text():
                print(f"HAC {hac} already done, skipping.")     
                continue
            ## look at the file size to decide if on disk or not
            file_sizes = sum([p.stat().st_size for p in hac_folders.glob("*.parquet")])
            batch_size = 3
            if file_sizes > size_limit:
                batch_size = 1
            
            (output_stats/hac).mkdir(parents=True, exist_ok=True)
            out_parq = output_stats / hac / out_parquet 
            print(f"counting stats {hac}")
            sta = compute_count(hac_folders, block_size)
            
            classified_folders = convert_folder(hac_folders)
            print(f"computing internal stats {hac}")
            internal_counts, dedup_dfs = compute_internal_duplication(classified_folders, 
                                                                      smiles_col, 
                                                                      block_size, 
                                                                      batch_size)
            
            print(f"computing database redundancy {hac}")
            overlaps = get_overlapping_databases(dedup_dfs, internal_counts, hac,
                                                 batch_size, smiles_col, small_threshold)
            redundante_smiles, redundant_counts = save_redundancy(overlaps,
                                                                  internal_counts, 
                                                                  smiles_col, 
                                                                  out_parq, 
                                                                  small_threshold)

            # save to disk the results
            pd.concat([sta, internal_counts], axis=1).to_csv(output_stats/hac/"after_before_counts.csv")
            redundant_counts.to_csv(output_stats/hac/"overlaping_counts.csv")

            with open(progress, "a") as f:
                f.write(f"HAC {hac} done\n")
            
            if Path(f"tmp/{hac}").exists():
                shutil.rmtree(f"tmp/{hac}", ignore_errors=True)

        
if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()