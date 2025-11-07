import dask.dataframe as dd
from dask.distributed import Client, performance_report
import os
from pathlib import Path
import argparse
import json
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate SMILES')
    parser.add_argument('-bs', '--blocksize', type=str, help='Block size for dask dataframe. The safest is the default 64MB',  required=False, default='64MB')
    parser.add_argument('-o','--output_path', type=str, help='Output foldr for the database', required=False,
                        default='Molecular_database')

    args = parser.parse_args()
    return args.blocksize, args.output_path



scheduler_address = os.environ["DASK_SCHEDULER_ADDRESS"]

client = Client(scheduler_address)    # Connect to that cluster

def compute_stats(hac_folders, block_size="64MB"):
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


def main():
    block_size, output_folder = parse_args()
    with performance_report(filename="dask-stats.html"):
            
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)
        progress = Path("progress_stats.txt")
        progress.touch(exist_ok=True)
        stats = out_path/"hac_stats.txt"
        stats.touch(exist_ok=True)
        hacs = sorted(out_path.glob("HAC*"), key=lambda x: int(x.name.split("_")[-1]))
        hacs_cleaned = sorted(out_path.glob("cleaned/HAC*"), key=lambda x: int(x.name.split("_")[-1]))
        for hac_folders in hacs:
            hac = hac_folders.name.split("_")[-1]
            if f"HAC {hac} before" in progress.read_text():
                print(f"HAC {hac} already done, skipping.")     
                continue    
            sta = compute_stats(hac_folders, block_size)
            
            with open(stats, "a") as w:
                w.write(f"HAC {hac} previous# {json.dumps(sta)}\n")
            
            with open(progress, "a") as f:
                f.write(f"HAC {hac} before done\n")
        
        for folder_cleaned in hacs_cleaned:
            hac = folder_cleaned.name.split("_")[-1]
            if f"HAC {hac} cleaned" in progress.read_text():
                print(f"HAC {hac} cleaned already done, skipping.")     
                continue    
            sta = compute_stats(folder_cleaned, block_size)
            
            with open(stats, "a") as w:
                w.write(f"HAC {hac} after# {json.dumps(sta)}\n")
            
            with open(progress, "a") as f:
                f.write(f"HAC {hac} cleaned done\n")


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()