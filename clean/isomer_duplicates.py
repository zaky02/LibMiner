import dask.dataframe as dd
from dask.distributed import Client, performance_report
import os
from pathlib import Path
import argparse
import dask
import pandas as pd
import shutil
import logging
import gc
from utils import convert_folder
from typing import Sequence


def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate SMILES')
    parser.add_argument('-bs', '--blocksize', type=str, help='Block size for dask dataframe. The safest is the default 64MB',  required=False, default='64MB')
    parser.add_argument('-dp','--database_path', type=str, help='The folder path for the database', required=False,
                        default='Molecular_database')
    parser.add_argument('-c','--canonical', type=str, help='The column name of the canonical smiles', required=False,
                        default='CANONICAL_SMILES')
    parser.add_argument("-C", "--catalogue_cols", nargs="+", help="The columns that identifies the original database entries", required=False, default=['ID', "SMILES"])
    
    args = parser.parse_args()
    return args.blocksize, args.database_path, args.canonical, args.catalogue_cols


scheduler_address = os.environ["DASK_SCHEDULER_ADDRESS"]
client = Client(scheduler_address)    # Connect to that cluster
client.wait_for_workers(n_workers=1, timeout=180)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_name_function(hac: str, db: str):
    def name_function(i: int) -> str:
        return f"HAC{hac}_db{db}_{i:02d}.parquet"
    return name_function


def aggregate_catalogue_information(
    db_paths: dict[str, str],
    output_path: Path | str,
    canonical_cols: str = "CANONICAL_SMILES",
    block_size: str = "64MB",
    catalog_cols: Sequence[str] = ("ID", "SMILES"),
    on_disk: bool = False
):
    """
    Aggregate the different SMILES and IDs under the same canonical smile (strip of isomeric information) for those commercial databases.
    """

    # Convert items to a list so we can slice batches
    shuffle_method = "disk" if on_disk else "tasks"
    
    for db_id, path in db_paths.items():
        df = dd.read_parquet(path, columns=[canonical_cols, *catalog_cols], blocksize=block_size)
        # drop duplicate by ID and SMILES just in case there is duplicates
        df = df.drop_duplicates(subset=catalog_cols[0]).drop_duplicates(subset=catalog_cols[1]) 
        df_dedup = df.groupby(canonical_cols).agg({col: list for col in catalog_cols}, 
                                                    split_out=df.npartitions, shuffle=shuffle_method)
        

        df_dedup.to_parquet(output_path, 
                            write_index=False, 
                            name_function=make_name_function(hac=output_path.name, db=db_id)
                            )


def main():
    block_size, database_path, canonical, catalogue_cols = parse_args()
    
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

            on_disk = False
            if file_sizes >= size_limit:

                on_disk = True
                
            out_parq = output_stats / hac 
            out_parq.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"deduplicating {hac}")
            classified_folders = convert_folder(hac_folders)

            aggregate_catalogue_information(classified_folders, out_parq, canonical, block_size, catalogue_cols, on_disk)
    
            
            with open(progress, "a") as f:
                f.write(f"HAC {hac} done\n")
    
        
if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()