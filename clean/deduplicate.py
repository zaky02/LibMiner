from pathlib import Path
import os
import time
import argparse
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, performance_report

def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate SMILES')
    parser.add_argument('-bs', '--blocksize', type=str, help='Block size for dask dataframe. The safest is the default 128MB',  required=False, default='64MB')
    parser.add_argument('-o','--output_path', type=str, help='Output foldr for the database', required=False,
                        default='Molecular_database')
    parser.add_argument('-s', '--repartition_size', type=str, help='The size for each of the files default is 500MB', required=False, default="500MB")
    parser.add_argument("-c", "--use_cols", nargs="+", help="Columns to read", required=False, 
                        default=['ID', "SMILES"])
    args = parser.parse_args()
    return args.blocksize, args.output_path, args.repartition_size, args.use_cols

# -------------------------
#  ️ Setup Dask cluster in Slurm
# -------------------------

scheduler_address = os.environ["DASK_SCHEDULER_ADDRESS"]

client = Client(scheduler_address)    # Connect to that cluster

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


def deduplicator(hac_folders: Path | str, out_path: Path | str, block_size: str = "64MB", 
                 repartition_size: str = "500MB",
                 use_cols: tuple[str] = ("ID", "SMILES"), current_offset: int = 0):
    """
    deduplicate, assign unique numerical IDs and write to HAC-specific Parquet folders.
    """
    
    out_path = Path(out_path)
    hac_folders = Path(hac_folders)
    hac = hac_folders.name.split("_")[-1]
    meta = {"ID": "string", "SMILES": "string", "db_id": "string", "num_id": "int64"}
    
    # read parquet files from a HAC  
    ddf_merged = dd.read_parquet(f"{hac_folders}/*.parquet", blocksize=block_size, 
                                columns=[*use_cols, "db_id"])

    # Deduplicate across all sources using normalized SMILES
    ddf_merged = ddf_merged.drop_duplicates(subset="SMILES").drop_duplicates(subset=["ID"])
    
    # Aim for ≤15M rows per partition because this is for each HAC
    ddf_merged = ddf_merged.repartition(partition_size=repartition_size)

    # Compute number of rows per partition (fast metadata op)
    partition_lengths = ddf_merged.map_partitions(len).compute()
    partition_offsets = np.insert(np.cumsum(partition_lengths[:-1]), 0, 0)
    # Assign unique IDs within Dask graph (no Python loop)
    ddf_merged = assign_ids(ddf_merged, partition_offsets, current_offset, meta)
    count = int(sum(partition_lengths))
    
    # -------------------------
    # 4️⃣ Write the database
    # -------------------------
    ddf_merged.to_parquet(
        out_path / f"cleaned/{hac}",
        write_index=False,
        compute=True,
        engine="pyarrow",
        name_function=make_name_function(hac=int(hac))     
    )
    
    return count

# -------------------------
# 2️⃣ Read databases lazily
# -------------------------

def main():
    block_size, output_folder, repartition_size, use_cols = parse_args()
    
    start = time.perf_counter()
    
    with performance_report(filename="dask-deduplicate.html"):
        
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)
        progress = Path("progress_deduplicate.txt")
        progress.touch(exist_ok=True)
        stats = out_path/"stats.txt"
        stats.touch(exist_ok=True)
        current_offset = 0   
        with open(stats, "r") as st:
            lines = {int(x.split(":")[0].strip("HAC")): int(x.strip().split(":")[-1]) for x in st.readlines()}
            
        hacs = sorted(out_path.glob("HAC*"), key=lambda x: int(x.name.split("_")[-1]))
        for hac_folders in hacs:
            hac = hac_folders.name.split("_")[-1]
            if f"HAC {hac}" in progress.read_text():
                print(f"HAC {hac} already done, skipping.")  
                current_offset = current_offset + lines[int(hac)]       
                continue
            
            count = deduplicator(hac_folders, out_path, block_size, repartition_size, 
                                 use_cols, current_offset)
            
            current_offset = current_offset + count
            
            with open(progress, "a") as f:
                f.write(f"HAC {hac} done\n")
                
            with open(stats, "a") as w:
                w.write(f"HAC {hac}: {count}\n")
            
        end = time.perf_counter()
        print(f"Initial cleaning completed in {end - start:.2f} seconds")    
   

if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()