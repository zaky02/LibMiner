from pathlib import Path
import os
import time
from rdkit import RDLogger
import argparse
import pyarrow.parquet as pq
from FPSim2.scripts.create_fpsim2_fp_db import create_db_file_parallel, merge_db_files, count_rows, calculate_chunks, read_chunk, create_db_file, generate_time_hash
import ray
import json
import pyarrow.compute as pc
import pyarrow as pa


def parse_args():
    parser = argparse.ArgumentParser(description='Create a fingerprint database from SMILES')
    parser.add_argument('-o', '--output_smi', type=str, help='The output .smi filename', 
                        required=False, default='all_molecules.smi')
    parser.add_argument('-i','--input_path', type=str, help='The molecular database folder path', required=False,
                        default='Molecular_database')
    parser.add_argument('-b','--batch_size', type=int, help='bacth size', required=False,
                        default=100_000)
    parser.add_argument('-fp','--fp_parm', type=json.loads, help='Fingerprint params', required=False,
                        default={"radius": 2, "fpSize": 1024})
    parser.add_argument('-ft','--fp_type', type=str, help='Fingerprint type supported by FPSim2', required=False,
                        default="Morgan")
    parser.add_argument('-oh', '--output_hdf', type=str, help='The output .h5 filename', 
                        required=False, default='fp_db.h5')
    parser.add_argument("-c", "--cpus", type=int, help="number of processors to run the create_db_file_parallel",
                        default=100)
    args = parser.parse_args()
    return args.input_path, args.output_smi, args.batch_size, args.fp_parm, args.output_hdf, args.cpus, args.fp_type


@ray.remote
def convert_parquet_to_smi_chunk(parquet_path: str | Path, out_dir: str | Path, 
                                 smiles_col: str="nostereo_SMILES", 
                                 id_col: str="num_ID", 
                                 batch_size: int=100_000):
    """
    Convert a single Parquet file to a temporary .smi chunk file.
    Executed as a Ray task.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    smi_temp = Path(out_dir) / (Path(parquet_path).stem + ".smi")

    with open(smi_temp, "w", encoding="utf-8") as f:
        for batch in parquet_file.iter_batches(columns=[smiles_col, id_col], batch_size=batch_size):
            smiles = batch.column(smiles_col)
            ids = batch.column(id_col).cast(pa.string())
            lines = pc.binary_join_element_wise(smiles, ids, "\n", " ")
            f.writelines(lines.to_pylist())
            
    return str(smi_temp)

def ray_parquet_to_smi(parquet_files: list[str | Path], 
                       out_smi: str | Path, 
                       smiles_col: str="nostereo_SMILES", 
                       id_col: str="num_ID", 
                       batch_size: int=100_000):
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
    
    
def sort_function(x: Path) -> tuple[int, int]:
    """Sort function for sorting Parquet files."""
    parts = x.stem.split('_')
    hac_part = parts[0]
    db_part = parts[1]
    hac_value = int(hac_part.replace('HAC', '').replace('wrongHAC', '0'))
    db_value = int(db_part.replace('db', ''))
    return (hac_value, db_value)


@ray.remote
def create_db_chunk(args):
    mols_source, chunk_range, fp_type, fp_params, full_sanitization = args
    time_hash = generate_time_hash()
    out_file = f"temp_chunk_{chunk_range[0]}_{time_hash}.h5"

    rows = read_chunk(mols_source, chunk_range[0], chunk_range[1])
    create_db_file(
        rows,
        out_file,
        mol_format="smiles",
        fp_type=fp_type,
        fp_params=fp_params,
        sort_by_popcnt=False,
        full_sanitization=full_sanitization,
    )
    return out_file


def create_db_file_parallel(
    smi_file, out_file, fp_type, fp_params, full_sanitization, num_processes
):
    total_mols = count_rows(smi_file)
    chunks = calculate_chunks(total_mols, num_processes)
    futures = [create_db_chunk.remote((smi_file, chunk, fp_type, fp_params, full_sanitization)) for chunk in chunks]
    tmp_filenames = ray.get(futures)

    merge_db_files(tmp_filenames, out_file, sort_by_popcnt=True)

    # Clean up temporary files
    for temp_file in tmp_filenames:
        os.remove(temp_file)


def main():
    
    input_folder, output_smi, batch_size, fp_params, output_hdf, cpus, fp_type = parse_args()
    RDLogger.DisableLog('rdApp.*')
    
    start = time.perf_counter()
    ray.init(address="auto", log_to_driver=False)
    print(ray.cluster_resources())
    
    if not Path(output_smi).exists():
        print("Converting Parquet files to SMILES...")
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        input_path = Path(input_folder)
        parquet_files = sorted(filter(lambda x: 4 <= int(x.parent.name.split("_")[-1]) <= 80 , input_path.glob("HAC_*/*.parquet")), key=sort_function)
        ray_parquet_to_smi(parquet_files, output_smi, batch_size=batch_size)
        
    Path(output_hdf).parent.mkdir(parents=True, exist_ok=True)
    
    if not Path(output_hdf).exists():
        print("Creating fingerprint database...")
        create_db_file_parallel(
        smi_file=output_smi, 
        out_file=output_hdf, 
        fp_type=fp_type, 
        fp_params=fp_params, 
        full_sanitization=False, 
        num_processes=cpus
    )
    
        Path(output_smi).unlink(missing_ok=True)
        
    end = time.perf_counter()
    print(f"Completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()