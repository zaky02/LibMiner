from pathlib import Path
import os
import time
from rdkit import RDLogger
import argparse
import pyarrow.parquet as pq
from FPSim2.scripts.create_fpsim2_fp_db import create_db_file_parallel
import ray


def parse_args():
    parser = argparse.ArgumentParser(description='Clusterize SMILES by generating fingerprints')
    parser.add_argument('-o', '--output_smi', type=str, help='The output .smi filename', 
                        required=False, default='all_molecules.smi')
    parser.add_argument('-i','--input_path', type=str, help='The molecular database folder path', required=False,
                        default='Molecular_database')
    parser.add_argument('-b','--batch_size', type=int, help='bacth size', required=False,
                        default=100_000)
    parser.add_argument('-fp','--fp_size', type=int, help='Fingerprint size', required=False,
                        default=512)
    parser.add_argument('-oh', '--output_hdf', type=str, help='The output .h5 filename', 
                        required=False, default='fp_db.h5')
    parser.add_argument("-c", "--cpus", type=int, help="number of processors to run the create_db_file_parallel",
                        default=4)
    args = parser.parse_args()
    return args.input_path, args.output_smi, args.batch_size, args.fp_size, args.output_hdf, args.cpus


@ray.remote
def convert_parquet_to_smi_chunk(parquet_path, out_dir, smiles_col="SMILES", id_col="ID", 
                                 batch_size=100_000):
    """
    Convert a single Parquet file to a temporary .smi chunk file.
    Executed as a Ray task.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    smi_temp = Path(out_dir) / (Path(parquet_path).stem + ".smi")

    buffer = []
    with open(smi_temp, "w", encoding="utf-8") as f:
        for batch in parquet_file.iter_batches(columns=[smiles_col, id_col], batch_size=batch_size):
            df = batch.to_pandas()
            lines = [f"{row[smiles_col]} {(row[id_col])}\n" for _, row in df.iterrows()]
            buffer.extend(lines)

            # Flush periodically to keep memory small
            if len(buffer) >= 1_500_000:
                f.writelines(buffer)
                buffer.clear()

        if buffer:
            f.writelines(buffer)

    return str(smi_temp)


def ray_parquet_to_smi(parquet_files, out_smi, smiles_col="SMILES", id_col="ID", batch_size=100_000):
    """
    Convert multiple Parquet files to a single .smi file using Ray for concurrency.
    Each file is processed in parallel as a Ray task.
    """
    out_dir = Path(out_smi).parent / "_tmp_smi_chunks"
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"🚀 Launching {len(parquet_files)} Ray tasks...")

    # Step 1. Launch Ray tasks for each Parquet file
    futures = [
        convert_parquet_to_smi_chunk.remote(pq_file, str(out_dir), smiles_col, id_col, batch_size)
        for pq_file in parquet_files
    ]

    # Step 2. Collect all temporary chunk file paths
    chunk_files = ray.get(futures)

    # Step 3. Merge all .smi chunks into the final file
    print("🧷 Combining temporary chunks...")
    with open(out_smi, "w", encoding="utf-8") as out_f:
        for chunk_file in chunk_files:
            with open(chunk_file, "r", encoding="utf-8") as cf:
                out_f.writelines(cf)
            os.remove(chunk_file)

    out_dir.rmdir()
    print(f"✅ Done! Combined {len(chunk_files)} Parquet files into {out_smi}")

    # Shutdown Ray when done (optional if you’ll reuse the cluster)
    ray.shutdown()
    

def main():
    
    input_folder, output_smi, batch_size, fp_size, output_hdf, cpus = parse_args()
    RDLogger.DisableLog('rdApp.*')
    
    start = time.perf_counter()
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    # Batch size can match #workers if desired, but each DB is processed fully partitioned
    input_path = Path(input_folder) / "cleaned"
    parquet_files = input_path.glob("HAC*/*.parquet")
    ray_parquet_to_smi(parquet_files, output_smi, batch_size=batch_size)
    
    Path(output_hdf).parent.mkdir(parents=True, exist_ok=True)
    
    create_db_file_parallel(
    smi_file=output_smi, 
    out_file=output_hdf, 
    fp_type="Morgan", 
    fp_params={"radius": 2, "fpSize": fp_size}, 
    full_sanitization=False, 
    num_processes=cpus
)
    
    Path(output_smi).unlink(missing_ok=True)
    end = time.perf_counter()
    print(f"Completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()