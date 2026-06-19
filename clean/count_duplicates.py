""" A script to deduplicate SMILES in a molecular database using Dask.
It is used 2 times during the creation of the database -> the first time is on the SMILES column.
You can use the default options

For the second time you have to change the following options:
- --use_cols nostereo_SMILES db_id
- --drop_cols nostereo_SMILES
- --assign_ids (to set it to true)
- --meta '{"nostereo_SMILES": "string", "num_ID": "int64", "db_id": "string"}'
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
from itertools import pairwise


def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate SMILES')
    parser.add_argument('-bs', '--blocksize', type=str, 
                        help='Block size for dask dataframe. The safest is the default 64 MB',  
                        required=False, default='64MB')
    parser.add_argument('-i','--input_path', type=str, help='Input folder where the database is', required=False,
                        default='Molecular_database')
    parser.add_argument("-c", "--use_cols", nargs="+", help="Columns to read", required=False, 
                        default=["nostereo_SMILES", "num_ID"])
    
    
    args = parser.parse_args()
    return args.blocksize, args.use_cols, args.input_path

# -------------------------
#  ️ Setup Dask cluster in Slurm
# -------------------------

scheduler_address = os.environ["DASK_SCHEDULER_ADDRESS"]

client = Client(scheduler_address)    # Connect to that cluster
client.wait_for_workers(n_workers=1, timeout=180)

# -------------------------
# 1️⃣ Setup RDKit tools
# -------------------------   


def deduplicator(hac_folders: Path | str, 
                 block_size: str = "64MB", 
                 use_cols: tuple[str] = ("nostereo_SMILES", "num_ID")) -> int:
    
    """
    Deduplicate, assign unique numerical IDs and write to HAC-specific Parquet folders.
    """
    
    out_path = Path(out_path)
    hac_folders = Path(hac_folders)
    hac = hac_folders.name.split("_")[-1]
    
    # read parquet files from a HAC  
    ddf_merged = dd.read_parquet(f"{hac_folders}/*.parquet", blocksize=block_size, 
                                columns=list(use_cols))
    partition_lengths = ddf_merged.map_partitions(len).compute()
    count1 = int(sum(partition_lengths))
    # Deduplicate across all sources using normalized SMILES
    ddf_merged = ddf_merged.drop_duplicates(subset="num_ID")

    # Compute number of rows per partition (fast metadata op)
    partition_lengths = ddf_merged.map_partitions(len).compute()
    count2 = int(sum(partition_lengths))
    count = count1 - count2

    del ddf_merged
    client.run(gc.collect)
    
    return count

# -------------------------
# 2️⃣ Read databases lazily
# -------------------------

def main():
    block_size,  use_cols, input_path= parse_args()
    
    start = time.perf_counter()
    
    with performance_report(filename="dask-deduplicate.html"):
        
        input_path = Path(input_path)
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        progress = Path("progress_numid.txt")
        progress.touch(exist_ok=True)

        current_offset = 0   
        hacs = sorted(input_path.glob("HAC_*"), key=lambda x: int(x.name.split("_")[-1]))
        for hac_folders in hacs:
            hac = hac_folders.name.split("_")[-1]
            if f"HAC {hac}" in progress.read_text():
                print(f"HAC {hac} already done, skipping.")    
                continue
            
            count = deduplicator(hac_folders, block_size,  use_cols)

            
            with open(progress, "a") as f:
                f.write(f"HAC {hac}: {count} done\n")
                
            
        end = time.perf_counter()
        print(f"Initial cleaning completed in {end - start:.2f} seconds")    

            
if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()