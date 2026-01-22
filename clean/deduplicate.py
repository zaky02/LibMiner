""" A script to deduplicate SMILES in a molecular database using Dask.
It is used 2 times during the creation of the database -> the first time is on the SMILES column.
You can use the default options

For the second time you have to change the following options:
- --use_cols nostereo_SMILES
- --drop_cols nostereo_SMILES
- --assign_ids (to set it to true)
- --meta '{"nostereo_SMILES": "string", "num_ID": "int64"}'
- --input_path to the output_path of the first deduplication, if default would be 'Molecular_database/deduplicated'
- --output_path to a different folder, for example 'Molecular_database/deduplicated_nostereo'

"""
from pathlib import Path
import os
import time
import argparse
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, performance_report
import pandas as pd
import gc
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate SMILES')
    parser.add_argument('-bs', '--blocksize', type=str, 
                        help='Block size for dask dataframe. The safest is the default 64 MB',  
                        required=False, default='64MB')
    parser.add_argument('-o','--output_path', type=str, help='Output folder for the deduplicated database', required=False, default='Molecular_database/deduplicate_canonical')
    parser.add_argument('-i','--input_path', type=str, help='Input folder where the database is', required=False,
                        default='Molecular_database')
    parser.add_argument('-s', '--repartition_size', type=int, 
                        help='The number of rows for each partition and file', 
                        required=False, default=10_000_000)
    parser.add_argument("-c", "--use_cols", nargs="+", help="Columns to read", required=False, 
                        default=["ID", "SMILES", "nostereo_SMILES", "db_id"])
    parser.add_argument("-d", "--drop_cols", type=str, help="Column to drop the duplicate", required=False, 
                        default='SMILES')
    parser.add_argument('-a','--assign_ids', action='store_true', help='Whether to assign unique numerical IDs', required=False)
    parser.add_argument('-m','--meta', type=json.loads, 
                        help='Metadata dictionary for dask dataframe from a json string', required=False, 
                        default={"ID": "string", "SMILES": "string", "db_id": "string", 
                                 "nostereo_SMILES": "string"})
    args = parser.parse_args()
    return args.blocksize, args.output_path, args.repartition_size, args.use_cols, args.input_path, args.drop_cols, args.assign_ids, args.meta

# -------------------------
#  ️ Setup Dask cluster in Slurm
# -------------------------

scheduler_address = os.environ["DASK_SCHEDULER_ADDRESS"]

client = Client(scheduler_address)    # Connect to that cluster
client.wait_for_workers(n_workers=1, timeout=180)

# -------------------------
# 1️⃣ Setup RDKit tools
# -------------------------   
def assign_ids(df, partition_offsets, global_offset, meta):
    """
    Assign unique sequential IDs per partition using map_partitions.
    `partition_offsets` must be broadcast as list of integers.
    """
    # Each partition gets its starting offset from partition_offsets
    def add_ids(partition, partition_info=None):
        pid = partition_info["number"]
        offset = partition_offsets[pid]
        partition["num_ID"] = np.arange(len(partition), dtype=np.int64) + global_offset + offset
        return partition

    return df.map_partitions(add_ids, meta=meta)


def make_name_function(hac: int):
    def name_function(i: int) -> str:
        return f"HAC{hac}_{i:02d}.parquet"
    return name_function


def deduplicator(hac_folders: Path | str, 
                 out_path: Path | str, 
                 block_size: str = "64MB", 
                 repartition_size: str = 10_000_000,
                 use_cols: tuple[str] = ("ID", "SMILES", "nostereo_SMILES", "db_id"), 
                 current_offset: int = 0, 
                 drop: str="SMILES",
                 if_assign_ids: bool=False,
                 meta: dict = {"ID": "string", "SMILES": "string", 
                               "db_id": "string", "nostereo_SMILES": "string"}) -> int:
    
    """
    Deduplicate, assign unique numerical IDs and write to HAC-specific Parquet folders.
    """
    
    out_path = Path(out_path)
    hac_folders = Path(hac_folders)
    hac = hac_folders.name.split("_")[-1]
    
    # read parquet files from a HAC  
    ddf_merged = dd.read_parquet(f"{hac_folders}/*.parquet", blocksize=block_size, 
                                columns=list(use_cols))
    
    # Deduplicate across all sources using normalized SMILES
    ddf_merged = ddf_merged.drop_duplicates(subset=drop)

    # Compute number of rows per partition (fast metadata op)
    partition_lengths = ddf_merged.map_partitions(len).compute()
    count = int(sum(partition_lengths))
    if count == 0:
        return count
    
    partition_offsets = np.insert(np.cumsum(partition_lengths[:-1]), 0, 0)
    # Assign unique IDs within Dask graph (no Python loop)
    if if_assign_ids:
        ddf_merged = assign_ids(ddf_merged, partition_offsets, current_offset, meta)
    
    n = max(1, int(count / repartition_size))
    # Aim for ≤15M rows per partition because this is for each HAC
    ddf_merged = ddf_merged.repartition(npartitions=n) # use num partitions instead of MB because it will cause memory problems apparently
    # -------------------------
    # 4️⃣ Write the database
    # -------------------------
    ddf_merged.to_parquet(
        out_path / f"HAC_{hac}",
        write_index=False,
        compute=True,
        engine="pyarrow",
        name_function=make_name_function(hac=int(hac))     
    )
    
    del ddf_merged
    client.run(gc.collect)
    
    return count

# -------------------------
# 2️⃣ Read databases lazily
# -------------------------

def main():
    block_size, output_folder, repartition_size, use_cols, input_path, drop_cols, if_assign_ids, meta = parse_args()
    
    start = time.perf_counter()
    
    with performance_report(filename="dask-deduplicate.html"):
        
        input_path = Path(input_path)
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)
        progress = out_path / Path("progress_deduplicate.txt")
        progress.touch(exist_ok=True)
        stats = out_path/"count_stats.txt"
        stats.touch(exist_ok=True)
        current_offset = 0   
        with open(stats, "r") as st:
            lines = {int(x.split("#")[0].strip("HAC")): 
                     int(x.strip().split("#")[-1]) for x in st.readlines()}
            
        hacs = sorted(input_path.glob("HAC_*"), key=lambda x: int(x.name.split("_")[-1]))
        for hac_folders in hacs:
            hac = hac_folders.name.split("_")[-1]
            if f"HAC {hac}" in progress.read_text():
                print(f"HAC {hac} already done, skipping.")  
                current_offset = current_offset + lines[int(hac)]       
                continue
            
            count = deduplicator(hac_folders, out_path, 
                                 block_size, repartition_size, 
                                 use_cols, current_offset, drop_cols,
                                 if_assign_ids, meta)
            
            current_offset = current_offset + count
            
            with open(progress, "a") as f:
                f.write(f"HAC {hac} done\n")
                
            with open(stats, "a") as w:
                w.write(f"HAC {hac}# {count}\n")
            
        end = time.perf_counter()
        print(f"Initial cleaning completed in {end - start:.2f} seconds")    
   
        with open(stats, "r") as st:
            lines = pd.Series({int(x.split("#")[0].strip("HAC")): 
                     int(x.strip().split("#")[-1]) for x in st.readlines()})
            lines.to_csv(out_path/"dedup_counts.csv")
            
if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()