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
from mpi4py import MPI


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

    args = parser.parse_args()
    return args.input_path, args.output_smi, args.batch_size, args.fp_parm, args.output_hdf, args.fp_type


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
    
    if smi_temp.exists():
        return str(smi_temp)  # restart-safe
    
    with open(smi_temp, "w", encoding="utf-8") as f:
        for batch in parquet_file.iter_batches(columns=[smiles_col, id_col], batch_size=batch_size):
            smiles = batch.column(smiles_col)
            ids = batch.column(id_col).cast(pa.string())
            lines = pc.binary_join_element_wise(smiles, ids, "\n", " ")
            f.writelines(lines.to_pylist())
            
    return str(smi_temp)

def parquet_to_smi_per_rank(rank,
                            size,
                            parquet_files: list[str | Path], 
                            out_dir: str | Path, 
                            smiles_col: str="nostereo_SMILES", 
                            id_col: str="num_ID", 
                            batch_size: int=100_000):
    """
    Convert multiple Parquet files to a single .smi file using Ray for concurrency.
    Each file is processed in parallel as a Ray task.
    """

    # ---- Deterministic file assignment ----
    my_files = parquet_files[rank::size]
    # Step 1. Launch Ray tasks for each Parquet file
    my_chunks = []
    for pq_file in my_files:
        try:
            chunk = convert_parquet_to_smi_chunk(
                pq_file,
                out_dir,
                smiles_col,
                id_col,
                batch_size,
            )
            my_chunks.append(chunk)
        except Exception as e:
            print(f"[Rank {rank}] ERROR processing {pq_file}: {e}")

    return my_chunks
    

def merge_smi_file(all_chunks: list[list[str]], 
                   out_smi: str | Path, 
                   parquet_files: list[str | Path],
                   out_dir: str | Path):
    
    all_chunks_flat = sorted([c for sub in all_chunks for c in sub], key=sort_function)
    
    expected = {str((out_dir / (Path(pq_file).stem + ".smi"))) for pq_file in parquet_files}

    missing = expected - set(all_chunks_flat)

    if missing:
        print("[MPI] WARNING: Missing chunks:")
        for m in sorted(missing):
            print(" ", m)

        print("\n[MPI] Not merging. Re-run the job to resume.")
        sys.exit(1)
        
    print("🧷 Combining temporary chunks...")
    with open(out_smi, "w", encoding="utf-8") as out_f:
        for chunk_file in all_chunks_flat:
            with open(chunk_file, "r", encoding="utf-8") as cf:
                out_f.writelines(cf)
            os.remove(chunk_file)
            
    out_dir.rmdir()            

def sort_function(x: str | Path) -> tuple[int, int]:
    """Sort function for sorting Parquet files."""
    parts = Path(x).stem.split('_')
    hac_part = parts[0]
    db_part = parts[1]
    hac_value = int(hac_part.replace('HAC', '').replace('wrongHAC', '0'))
    db_value = int(db_part.replace('db', ''))
    return (hac_value, db_value)


def get_chunks(rank, size, comm, output_smi):
    if rank == 0:
        print("[Rank 0] Starting count_rows()", flush=True)
        total_mols = count_rows(output_smi)
        print(f"[Rank 0] count_rows done: {total_mols}", flush=True)
    else:
        total_mols = None

    total_mols = comm.bcast(total_mols, root=0)

    num_chunks = int(size / 4)
    chunks = calculate_chunks(total_mols, num_chunks)
    all_chunks = list(enumerate(chunks))
    my_chunks = all_chunks[rank::size]
    print(
        f"[Rank {rank}] Assigned {len(my_chunks)} chunks with IDs {" ".join([str(c[0]) for c in my_chunks])}",
        flush=True
    )

    return my_chunks, num_chunks

def run_chunks_per_rank(
    rank: int,
    my_chunks: list[tuple[int, list[tuple[int, int]]]],
    fp_type: str,
    fp_params: dict,
    output_smi: str | Path,
    TMP_DIR: Path,
):
    """
    Create chunks of H5 files with atomic completion guarantees.
    """
    completed_files = []

    for chunk_id, chunk in my_chunks:
        final_file = TMP_DIR / f"chunk_{chunk_id}.h5"
        tmp_file = TMP_DIR / f"chunk_{chunk_id}.h5.tmp"

        # If final file exists, it is guaranteed complete
        if final_file.exists():
            print(f"[Rank {rank}] Chunk {chunk_id} already completed, skipping.")
            completed_files.append(str(final_file))
            continue

        # Clean up stale temp file
        if tmp_file.exists():
            print(f"[Rank {rank}] Found incomplete temp file for chunk {chunk_id}, removing.")
            tmp_file.unlink()

        try:
            print(f"[Rank {rank}] Creating chunk {chunk_id}")

            create_db_file(
                read_chunk(output_smi, chunk[0], chunk[1]),
                str(tmp_file),
                mol_format="smiles",
                fp_type=fp_type,
                fp_params=fp_params,
                sort_by_popcnt=False,
                full_sanitization=False,
            )

            # Atomic move: guarantees completeness
            tmp_file.replace(final_file)

            completed_files.append(str(final_file))
            print(f"[Rank {rank}] Finished chunk {chunk_id}")

        except Exception as e:
            print(f"[Rank {rank}] Error creating chunk {chunk_id}: {e}", flush=True)
            # Ensure no corrupted temp file remains
            if tmp_file.exists():
                tmp_file.unlink()

    return completed_files


def create_db_file_merged(
    all_files, num_chunks, output_hdf, TMP_DIR
):
    
    all_files_flat = sorted([f for sublist in all_files for f in sublist], key=lambda x: int(Path(x).stem.split('_')[-1]))
        # Ensure all expected chunks exist
    expected = {
        (TMP_DIR/f"chunk_{chunk_id}.h5")
        for chunk_id in range(num_chunks)
    }

    missing = expected - set(all_files_flat)

    if missing:
        print("[MPI] WARNING: Missing chunks:")
        for m in sorted(missing):
            print(" ", m)

        print("\n[MPI] Not merging. Re-run the job to resume.")
        sys.exit(1)
    
    print("Merging all chunks into final database...")
    merge_db_files(all_files_flat, output_hdf, sort_by_popcnt=True)
    print(f"Fingerprint database created at {output_hdf}")
    
    # Clean up temporary files
    for temp_file in all_files_flat:
        os.remove(temp_file)
        
    TMP_DIR.rmdir()


def main():
    
    input_folder, output_smi, batch_size, fp_params, output_hdf, fp_type = parse_args()
    RDLogger.DisableLog('rdApp.*')
    
    start = time.perf_counter()

    # Init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if not Path(output_smi).exists():
        
        # Batch size can match #workers if desired, but each DB is processed fully partitioned
        input_path = Path(input_folder)
        parquet_files = sorted(filter(lambda x: 4 <= int(x.parent.name.split("_")[-1]) <= 80 , input_path.glob("HAC_*/*.parquet")), key=sort_function)
        out_dir = Path(output_smi).parent / "_tmp_smi_chunks"
        # ---- Create temp dir once ----
        if rank == 0:
            print("Converting Parquet files to SMILES...")
            out_dir.mkdir(exist_ok=True, parents=True)
            
        comm.Barrier()
        
        my_chunks = parquet_to_smi_per_rank(rank, size, parquet_files, out_dir, batch_size=batch_size)
        # Step 2. Gather all chunk filenames to rank 0
        all_chunks = comm.gather(my_chunks, root=0)
        
        if rank == 0:
            merge_smi_file(all_chunks, output_smi, out_dir)
            print(f"SMILES file created at {output_smi}")
        
    
    output_hdf = Path(output_hdf)
    if not output_hdf.exists():
        TMP_DIR = Path("tmp_chunks")
        if rank == 0:
            print("Creating fingerprint database...")
            output_hdf.parent.mkdir(parents=True, exist_ok=True)
            TMP_DIR.mkdir(exist_ok=True)
        
        comm.Barrier()
        
        my_chunks, num_chunks = get_chunks(rank, size, comm, output_smi)
        # ---- Only rank 0 does expensive global work ----
        completed_files = run_chunks_per_rank(rank, my_chunks, fp_type, fp_params, output_smi, TMP_DIR)
        # ---- Gather completed parts on rank 0 ----
        all_files = comm.gather(completed_files, root=0)

        if rank == 0:
            create_db_file_merged(
                all_files, num_chunks, output_hdf, TMP_DIR
            )
              
        #Path(output_smi).unlink(missing_ok=True)
        
    end = time.perf_counter()
    print(f"Completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    # Run this if this file is executed from command line but not if is imported as API
    main()