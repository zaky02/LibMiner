"""
Create a fingerprint database from SMILES stored in Parquet files using MPI for parallelism.
Each MPI rank processes a subset of Parquet files, converts them to SMILES format, and then creates fingerprint chunks.
The final fingerprint database is merged from all chunks.

"""
from pathlib import Path
import os
import time
import sys
from rdkit import RDLogger
import argparse
import pyarrow.parquet as pq
from FPSim2.scripts.create_fpsim2_fp_db import merge_db_files, count_rows, calculate_chunks, read_chunk, create_db_file
import json
import pyarrow.compute as pc
import pyarrow as pa


def parse_args():
    parser = argparse.ArgumentParser(description='Create a fingerprint database from SMILES')
    parser.add_argument('-o', '--output_smi', type=str, help='The output .smi filename', 
                        required=False, default='all_molecules.smi')
    parser.add_argument('-i','--input_path', type=str, help='The molecular database folder path', required=False,
                        default='Molecular_database')
    parser.add_argument('-b','--batch_size', type=int, help='batch size', required=False,
                        default=100_000)
    parser.add_argument('-fp','--fp_param', type=json.loads, help='Fingerprint params', required=False,
                        default={"radius": 2, "fpSize": 1024})
    parser.add_argument('-ft','--fp_type', type=str, help='Fingerprint type supported by FPSim2', required=False,
                        default="Morgan")
    parser.add_argument('-oh', '--output_hdf', type=str, help='The output .h5 filename', 
                        required=False, default='fp_db.h5')
    parser.add_argument('--stage', type=str, choices=['convert', 'fingerprint', 'merge_smi', 'merge_fp'], 
                        help='Processing stage', required=True)

    args = parser.parse_args()
    return args


def sort_function(x: str | Path) -> tuple[int, int]:
    """Sort function for sorting Parquet files."""
    parts = Path(x).stem.split('_')
    hac_part = parts[0]
    db_part = parts[1]
    hac_value = int(hac_part.replace('HAC', '').replace('wrongHAC', '0'))
    db_value = int(db_part.replace('db', ''))
    return (hac_value, db_value)


def convert_parquet_to_smi_chunk(parquet_path: str | Path, out_dir: str | Path, 
                                 smiles_col: str="nostereo_SMILES", 
                                 id_col: str="num_ID", 
                                 batch_size: int=100_000):
    """
    Convert a single Parquet file to a temporary .smi chunk file.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    smi_temp = Path(out_dir) / (Path(parquet_path).stem + ".smi")
    
    if smi_temp.exists():
        print(f"Chunk {smi_temp} already exists, skipping.")
        return str(smi_temp)
    
    print(f"Processing {parquet_path} -> {smi_temp}")
    
    with open(smi_temp, "w", encoding="utf-8") as f:
        for batch in parquet_file.iter_batches(columns=[smiles_col, id_col], batch_size=batch_size):
            smiles = batch.column(smiles_col)
            ids = batch.column(id_col).cast(pa.string())
            lines = pc.binary_join_element_wise(smiles, ids, "\n", " ")
            f.writelines(lines.to_pylist())
            
    print(f"Completed {smi_temp}")
    return str(smi_temp)


def stage_convert_parquet(args):
    """Stage 1: Convert parquet files to SMI chunks (parallelized via SLURM array)"""
    
    # Get SLURM array task ID and count
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    array_size = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    
    print(f"[Task {task_id}/{array_size}] Starting parquet conversion")
    
    input_path = Path(args.input_path)
    parquet_files = sorted(
        filter(lambda x: 4 <= int(x.parent.name.split("_")[-1]) <= 80, 
               input_path.glob("HAC_*/*.parquet")), 
        key=sort_function
    )
    
    out_dir = Path(args.output_smi).parent / "_tmp_smi_chunks"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Assign files to this task
    my_files = parquet_files[task_id::array_size]
    
    print(f"[Task {task_id}] Processing {len(my_files)} files")
    
    for pq_file in my_files:
        try:
            convert_parquet_to_smi_chunk(
                pq_file,
                out_dir,
                batch_size=args.batch_size,
            )
        except Exception as e:
            print(f"[Task {task_id}] ERROR processing {pq_file}: {e}")
            sys.exit(1)
    
    print(f"[Task {task_id}] Completed all assigned files")


def stage_merge_smi(args):
    """Stage 2: Merge all SMI chunks into final file (single job)"""
    
    print("Merging SMI chunks...")
    
    input_path = Path(args.input_path)
    parquet_files = sorted(
        filter(lambda x: 4 <= int(x.parent.name.split("_")[-1]) <= 80, 
               input_path.glob("HAC_*/*.parquet")), 
        key=sort_function
    )
    
    out_dir = Path(args.output_smi).parent / "_tmp_smi_chunks"
    
    # Collect all expected chunk files
    chunk_files = sorted([
        out_dir / (Path(pq_file).stem + ".smi") 
        for pq_file in parquet_files
    ], key=sort_function)
    
    # Check for missing chunks
    missing = [f for f in chunk_files if not f.exists()]
    if missing:
        print("ERROR: Missing chunk files:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)
    
    # Merge chunks
    print(f"Merging {len(chunk_files)} chunks into {args.output_smi}")
    with open(args.output_smi, "w", encoding="utf-8") as out_f:
        for chunk_file in chunk_files:
            print(f"  Adding {chunk_file.name}")
            with open(chunk_file, "r", encoding="utf-8") as cf:
                out_f.writelines(cf)
            os.remove(chunk_file)
    
    # Clean up temp directory
    if out_dir.exists() and not any(out_dir.iterdir()):
        out_dir.rmdir()
    
    print(f"SMILES file created at {args.output_smi}")


def stage_create_fingerprints(args):
    """Stage 3: Create fingerprint chunks (parallelized via SLURM array)"""
    
    # Get SLURM array task ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    array_size = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    
    print(f"[Task {task_id}/{array_size}] Starting fingerprint generation")
    
    TMP_DIR = Path("tmp_chunks")
    TMP_DIR.mkdir(exist_ok=True)
    
    # Count total molecules
    total_mols = count_rows(args.output_smi)
    print(f"[Task {task_id}] Total molecules: {total_mols}")
    
    # Calculate chunks
    chunks = calculate_chunks(total_mols, array_size, m=1)
    chunk_list = list(enumerate(chunks))
    
    # This task processes only its assigned chunk
    if task_id >= len(chunk_list):
        print(f"[Task {task_id}] No chunk assigned (total chunks: {len(chunk_list)})")
        return
    
    chunk_id, chunk = chunk_list[task_id]
    
    final_file = TMP_DIR / f"chunk_{chunk_id}.h5"
    tmp_file = TMP_DIR / f"chunk_{chunk_id}.h5.tmp"
    
    # Check if already completed
    if final_file.exists():
        print(f"[Task {task_id}] Chunk {chunk_id} already completed")
        return
    
    # Clean up any stale temp file
    if tmp_file.exists():
        print(f"[Task {task_id}] Removing stale temp file")
        tmp_file.unlink()
    
    try:
        print(f"[Task {task_id}] Creating chunk {chunk_id} (rows {chunk[0][0]}-{chunk[-1][1]})")
        
        create_db_file(
            read_chunk(args.output_smi, chunk[0], chunk[1]),
            str(tmp_file),
            mol_format="smiles",
            fp_type=args.fp_type,
            fp_params=args.fp_param,
            sort_by_popcnt=False,
            full_sanitization=False,
        )
        
        # Atomic move ensures completion
        tmp_file.replace(final_file)
        
        print(f"[Task {task_id}] Successfully created chunk {chunk_id}")
        
    except Exception as e:
        print(f"[Task {task_id}] ERROR creating chunk {chunk_id}: {e}")
        if tmp_file.exists():
            tmp_file.unlink()
        sys.exit(1)


def stage_merge_fingerprints(args):
    """Stage 4: Merge fingerprint chunks into final database (single job)"""
    
    print("Merging fingerprint chunks...")
    
    TMP_DIR = Path("tmp_chunks")
    
    array_size = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    
    # We need to know how many chunks were created
    # This should match the array size used in stage 3
    chunk_files = sorted(TMP_DIR.glob("chunk_*.h5"), 
                        key=lambda x: int(x.stem.split('_')[-1]))
    
    if not chunk_files:
        print("ERROR: No chunk files found")
        sys.exit(1)
    
    print(f"Found {len(chunk_files)} chunk files")
    
    # Check for any missing chunks
    chunk_ids = sorted([int(f.stem.split('_')[-1]) for f in chunk_files])
    expected_ids = list(range(array_size))
    
    missing = set(expected_ids) - set(chunk_ids)
    if missing:
        print(f"ERROR: Missing chunks: {sorted(missing)}")
        sys.exit(1)
    
    # Merge all chunks
    print(f"Merging {len(chunk_files)} chunks into {args.output_hdf}")
    merge_db_files([str(f) for f in chunk_files], args.output_hdf, sort_by_popcnt=True)
    
    print(f"Fingerprint database created at {args.output_hdf}")
    
    # Clean up
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    
    if TMP_DIR.exists() and not any(TMP_DIR.iterdir()):
        TMP_DIR.rmdir()
    
    # Optionally remove SMI file
    # Path(args.output_smi).unlink(missing_ok=True)


def main():
    args = parse_args()
    RDLogger.DisableLog('rdApp.*')
    
    start = time.perf_counter()
    
    if args.stage == 'convert':
        stage_convert_parquet(args)
    elif args.stage == 'merge_smi':
        stage_merge_smi(args)
    elif args.stage == 'fingerprint':
        stage_create_fingerprints(args)
    elif args.stage == 'merge_fp':
        stage_merge_fingerprints(args)
    else:
        print(f"Unknown stage: {args.stage}")
        sys.exit(1)
    
    end = time.perf_counter()
    print(f"Stage '{args.stage}' completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()