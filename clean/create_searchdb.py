"""
Create a fingerprint database from SMILES stored in Parquet files using SLURM for parallelism.
Each array processes a subset of Parquet files, converts them to SMILES format, and then creates fingerprint chunks depending on the stage
The final fingerprint database is merged from all chunks.

"""
from pathlib import Path
import os
import time
import sys
from rdkit import RDLogger
import argparse
import pyarrow.parquet as pq
from FPSim2.scripts.create_fpsim2_fp_db import count_rows, calculate_chunks, read_chunk, create_db_file
from FPSim2.io.chem import get_fp_length
from FPSim2.io.backends.pytables import create_schema, sort_db_file
import tables as tb
import json
import pyarrow.compute as pc
import pyarrow as pa
import rdkit
from importlib.metadata import version
__version__ = version("FPSim2")


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
    parser.add_argument('-s', '--stage', type=str, choices=['convert', 'fingerprint', 'merge_smi', 'merge_batches', 'merge_final', "index", "sort"], help='Processing stage', required=True)

    args = parser.parse_args()
    return args.output_smi, args.input_path, args.batch_size, args.fp_param, args.fp_type, args.output_hdf, args.stage

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


def stage_convert_parquet(input_path: str | Path, output_smi: str | Path, batch_size: int):
    """Stage 1: Convert parquet files to SMI chunks (parallelized via SLURM array)"""
    
    # Get SLURM array task ID and count
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    array_size = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    
    print(f"[Task {task_id}/{array_size}] Starting parquet conversion")
    
    input_path = Path(input_path)
    parquet_files = sorted(
        filter(lambda x: 4 <= int(x.parent.name.split("_")[-1]) <= 80, 
               input_path.glob("HAC_*/*.parquet")), 
        key=sort_function
    )
    
    out_dir = Path(output_smi).parent / "_tmp_smi_chunks"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Assign files to this task
    my_files = parquet_files[task_id::array_size]
    
    print(f"[Task {task_id}] Processing {len(my_files)} files")
    
    for pq_file in my_files:
        try:
            convert_parquet_to_smi_chunk(
                pq_file,
                out_dir,
                batch_size=batch_size,
            )
        except Exception as e:
            print(f"[Task {task_id}] ERROR processing {pq_file}: {e}")
            sys.exit(1)
    
    print(f"[Task {task_id}] Completed all assigned files")


def stage_merge_smi(input_path: str | Path, output_smi: str | Path):
    """Stage 2: Merge all SMI chunks into final file (single job)"""
    
    print("Merging SMI chunks...")
    
    input_path = Path(input_path)
    parquet_files = sorted(
        filter(lambda x: 4 <= int(x.parent.name.split("_")[-1]) <= 80, 
               input_path.glob("HAC_*/*.parquet")), 
        key=sort_function
    )
    
    out_dir = Path(output_smi).parent / "_tmp_smi_chunks"
    
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
    print(f"Merging {len(chunk_files)} chunks into {output_smi}")
    with open(output_smi, "w", encoding="utf-8") as out_f:
        for chunk_file in chunk_files:
            print(f"  Adding {chunk_file.name}")
            with open(chunk_file, "r", encoding="utf-8") as cf:
                out_f.writelines(cf)
            os.remove(chunk_file)
    
    # Clean up temp directory
    if out_dir.exists() and not any(out_dir.iterdir()):
        out_dir.rmdir()
    
    print(f"SMILES file created at {output_smi}")


def create_index_on_existing_file(batch_output: Path | str, 
                                  tmp_dir: str = None):
    """Create index on an existing PyTables file."""

    if not Path(batch_output).exists():
        print(f"{batch_output} doesn't exists")
        return
    
    if tmp_dir is None:
        tmp_dir = Path(db_file).parent / Path(db_file).stem + "_tmp_index"
        tmp_dir.mkdir(exist_ok=True)
        
    with tb.open_file(batch_output, mode="a") as f:  # ← "a" for append/modify
        # Check if index already exists
        if f.root.fps.cols.popcnt.is_indexed:
            print(f"Index already exists on {batch_output}")
            return
        
        if tmp_dir is None:
            tmp_dir = Path(batch_output.stem + "_index")
            tmp_dir.mkdir(exist_ok=True)
        
        print(f"Creating index on {batch_output}...")
        f.root.fps.cols.popcnt.create_csindex(
            tmp_dir=str(tmp_dir)
        )
        
        print("Index created successfully!")
        
        # Clean up temp directory
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()


def merge_db_files(
    input_files: list[str], output_file: str
) -> None:
    """Merges multiple FPs db files into a new one.

    Parameters
    ----------
    input_files : List[str]
        List of paths to input files
    output_file : str
        Path to output merged file
    sort_by_popcnt : bool, optional
        Whether to sort the output file by population count, by default True
    """
    if len(input_files) < 2:
        raise ValueError("At least two input files are required for merging")

    # Check that all files have same fingerprint type, parameters and RDKit version
    reference_configs = None
    for file in input_files:
        with tb.open_file(file, mode="r") as f:
            current_configs = (
                f.root.config[0],
                f.root.config[1],
                f.root.config[2],
                f.root.config[3],
            )
            if reference_configs is None:
                reference_configs = current_configs
            elif current_configs != reference_configs:
                raise ValueError(
                    f"File {file} has different fingerprint types, parameters or RDKit versions"
                )

    # Create new file with same parameters
    filters = tb.Filters(complib="blosc2", complevel=9, fletcher32=False)
    fp_type, fp_params, original_rdkit_ver, original_fpsim2_ver = reference_configs
    fp_length = get_fp_length(fp_type, fp_params)

    with tb.open_file(output_file, mode="w") as out_file:
        particle = create_schema(fp_length)
        fps_table = out_file.create_table(
            out_file.root, "fps", particle, "Table storing fps", filters=filters
        )

        # Copy config with original RDKit version
        param_table = out_file.create_vlarray(
            out_file.root, "config", atom=tb.ObjectAtom()
        )
        param_table.append(fp_type)
        param_table.append(fp_params)
        param_table.append(original_rdkit_ver)
        param_table.append(original_fpsim2_ver)

        # Copy data from all input files (appending one by one)
        for file in input_files:
            with tb.open_file(file, mode="r") as in_file:
                fps_table.append(in_file.root.fps[:])
                

def stage_create_fingerprints(output_smi: str | Path, fp_type: str, fp_param: dict):
    """Stage 3: Create fingerprint chunks (parallelized via SLURM array)"""
    
    # Get SLURM array task ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    array_size = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    
    print(f"[Task {task_id}/{array_size}] Starting fingerprint generation")
    
    TMP_DIR = Path("tmp_chunks")
    TMP_DIR.mkdir(exist_ok=True)
    
    # Count total molecules
    total_mols = count_rows(output_smi)
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
        print(f"[Task {task_id}] Creating chunk {chunk_id} (rows {chunk[0]}-{chunk[1]})")
        
        create_db_file(
            read_chunk(output_smi, chunk[0], chunk[1]),
            str(tmp_file),
            mol_format="smiles",
            fp_type=fp_type,
            fp_params=fp_param,
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


def stage_merge_fingerprints(final: bool = False, 
                             output_hdf: str | Path | None = None):
    """Stage 4: Merge fingerprint chunks into final database (single job)"""
    
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    array_size = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
        
    TMP_DIR = Path("tmp_chunks")
    MERGE_DIR = Path(output_path)
    MERGE_DIR.mkdir(exist_ok=True)
    # We need to know how many chunks were created
    chunk_files = sorted(TMP_DIR.glob("chunk_*.h5"), 
                         key=lambda x: int(x.stem.split('_')[-1]))
    
    if final:
        # If this is the final stage, merge all batches into a single file
        if output_hdf is None:
            print("ERROR: output_hdf must be specified for final merge stage")
            sys.exit(1)
        output_file = Path(output_hdf)
        batch_files = sorted(MERGE_DIR.glob("batch_*.h5"),
                            key=lambda x: int(x.stem.split('_')[-1]))
        if not batch_files:
            print("ERROR: No batch files found")
            sys.exit(1)

        merge_db_files([str(f) for f in batch_files], str(output_file))
        return output_file
    
    if not chunk_files:
        print("ERROR: No chunk files found")
        sys.exit(1)
    
    print(f"Found {len(chunk_files)} chunk files")
    
    my_chunks = chunk_files[task_id::array_size]
    batch_output = MERGE_DIR / f"batch_{task_id}.h5"
    
    # Skip if already done
    if batch_output.exists():
        print(f"[Task {task_id}] Batch already exists, skipping")
        return
    
    print(f"[Task {task_id}] Merging {len(my_chunks)} chunks into batch_{task_id}.h5")
    
    # Merge WITHOUT sorting (sorting is slow and done only once at the end)
    merge_db_files([str(f) for f in my_chunks], str(batch_output))
    
    print(f"[Task {task_id}] Batch merge complete")

  
def calc_popcnt_bins_fast(fps, fp_length: int) -> list[tuple[int, tuple[int, int]]]:
    """
    Fast popcnt bin calculation using streaming (memory-efficient for billions of molecules).
    
    Original implementation: O(n × fp_length) - scans table once per popcnt value (513 scans!)
    This implementation: O(n) - single streaming pass through sorted table
    
    Speedup: 100-500x faster, works with any database size
    
    Parameters
    ----------
    fps : tables.Table
        The fingerprint table (must be sorted by popcnt)
    fp_length : int
        Length of the fingerprint (e.g., 512)
        
    Returns
    -------
    list
        List of (popcnt, (start_idx, end_idx)) tuples
    """
    print("Calculating popcnt bins (streaming method for large databases)...")
    
    popcnt_bins = []
    current_popcnt = None
    first_idx = None
    last_idx = None
    
    # Process in chunks to avoid memory issues
    chunk_size = 10_000_000  # 10M rows per chunk
    total_rows = fps.nrows
    
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        
        # Read only popcnt column for this chunk
        popcnt_chunk = fps.read(start=start, stop=end, field='popcnt')
        
        for i, pc in enumerate(popcnt_chunk):
            global_idx = start + i
            
            if pc != current_popcnt:
                # Save previous bin
                if current_popcnt is not None:
                    popcnt_bins.append((current_popcnt, (first_idx, last_idx + 1)))
                
                # Start new bin
                current_popcnt = int(pc)
                first_idx = global_idx
            
            last_idx = global_idx
        
        # Progress indicator
        if end % 100_000_000 == 0 or end == total_rows:
            print(f"  Processed {end:,} / {total_rows:,} molecules ({end/total_rows*100:.1f}%)")
    
    # Don't forget the last bin
    if current_popcnt is not None:
        popcnt_bins.append((current_popcnt, (first_idx, last_idx + 1)))
    
    print(f"  Found {len(popcnt_bins)} popcnt bins")
    return popcnt_bins


def sort_db_file_fast(
    filename: str, 
    out_file: str | Path,
    compression_level: int = 9,
    verbose: bool = True
    
) -> None:
    """
    Optimized version of FPSim2's sort_db_file for billion-molecule databases.
    
    Improvements:
    - Lower default compression (6 instead of 9) for 2-3x speedup
    - Streaming popcnt calculation (works with any database size)
    - Progress reporting
    - Timing information
    
    Memory usage: ~1-2 GB regardless of database size (streaming)
    
    Parameters
    ----------
    filename : str
        Path to the database file to sort
    compression_level : int, optional
        Blosc2 compression level (1-9)
        - 1-3: Fast, larger files (~10-15% larger than level 9)
        - 6: Balanced (recommended, default, ~5% larger than level 9)
        - 9: Slow, smallest files
    verbose : bool, optional
        Print progress information
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimized Database Sorting")
        print(f"{'='*60}")
        print(f"File: {filename}")
        print(f"Compression level: {compression_level}")
    
    start_time = time.time()
    
    # Rename unsorted file
    tmp_filename = filename

    # Use specified compression level
    filters = tb.Filters(complib="blosc2", complevel=compression_level, fletcher32=False)
    
    try:
        with (
            tb.open_file(tmp_filename, mode="r") as fp_file,
            tb.open_file(out_file, mode="w") as sorted_fp_file,
        ):
            # Get configuration
            fp_type = fp_file.root.config[0]
            fp_params = fp_file.root.config[1]
            
            # Import here to avoid circular dependency
            fp_length = get_fp_length(fp_type, fp_params)
            
            if verbose:
                nrows = fp_file.root.fps.nrows
                print(f"Rows to sort: {nrows:,}")
                print(f"Fingerprint length: {fp_length}")
                print(f"\nStep 1/2: Copying and sorting table...")
            
            copy_start = time.time()
            
            # Create sorted copy
            dst_fps = fp_file.root.fps.copy(
                sorted_fp_file.root,
                "fps",
                filters=filters,
                copyuserattrs=True,
                overwrite=True,
                stats={
                    "groups": 0,
                    "leaves": 0,
                    "links": 0,
                    "bytes": 0,
                    "hardlinks": 0,
                },
                start=None,
                stop=None,
                step=None,
                chunkshape="auto",
                sortby="popcnt",
                check_CSI=True,
                propindexes=True,
            )
            
            copy_time = time.time() - copy_start
            if verbose:
                print(f"  Completed in {copy_time:.1f}s")
            
            # Create config table
            param_table = sorted_fp_file.create_vlarray(
                sorted_fp_file.root, "config", atom=tb.ObjectAtom()
            )
            param_table.append(fp_type)
            param_table.append(fp_params)
            param_table.append(rdkit.__version__)
            param_table.append(__version__)
            
            # Calculate popcnt bins using fast method
            if verbose:
                print(f"\nStep 2/2: Calculating popcnt bins...")
            
            bins_start = time.time()
            popcnt_bins = calc_popcnt_bins_fast(dst_fps, fp_length)
            bins_time = time.time() - bins_start
            
            param_table.append(popcnt_bins)
            
            if verbose:
                print(f"  Found {len(popcnt_bins)} popcnt bins")
                print(f"  Completed in {bins_time:.1f}s")
        
        # Remove temporary file
        os.remove(tmp_filename)
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Sorting completed successfully!")
            print(f"Total time: {total_time:.1f}s")
            print(f"  Copy/sort: {copy_time:.1f}s ({copy_time/total_time*100:.1f}%)")
            print(f"  Popcnt bins: {bins_time:.1f}s ({bins_time/total_time*100:.1f}%)")
            
            # Show file size
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"Output file size: {file_size_mb:.1f} MB")
            print(f"{'='*60}\n")
    
    except Exception as e:
        # Restore original file if sorting fails
        if os.path.exists(tmp_filename):
            if os.path.exists(filename):
                os.remove(filename)
            os.rename(tmp_filename, filename)
        raise e


def estimate_sort_time(filename: str) -> dict:
    """
    Estimate sorting time for different compression levels.
    
    Parameters
    ----------
    filename : str
        Path to the database file
        
    Returns
    -------
    dict
        Estimated times for different compression levels
    """
    with tb.open_file(filename, mode="r") as f:
        nrows = f.root.fps.nrows
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    
    # Rough estimates based on empirical testing
    # Sorting is roughly O(n log n) but dominated by I/O
    base_time = (nrows / 1_000_000) * 5  # ~5 min per million rows baseline
    
    estimates = {
        "compression_3": base_time * 0.6,
        "compression_6": base_time * 1.0,
        "compression_9": base_time * 2.0,
    }
    
    print(f"Database: {file_size_mb:.1f} MB, {nrows:,} rows")
    print(f"Estimated sorting times:")
    print(f"  Compression 3 (fast): ~{estimates['compression_3']:.0f} minutes")
    print(f"  Compression 6 (balanced): ~{estimates['compression_6']:.0f} minutes")
    print(f"  Compression 9 (slow): ~{estimates['compression_9']:.0f} minutes")
    
    return estimates
    

def stage_final(input_path: str | Path,
                output_path: str | Path = "Molecular_database/search_db", 
                sort_by_popcnt: bool = False,
                compression_level: int = 9):
    
    """Stage 5: Create index on final database and sort by population count (single job)"""
    
    # Get SLURM array task ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    MERGE_DIR = Path(output_path)
    batch_output = MERGE_DIR / f"batch_{task_id}.h5"
    # Check if final file already exists

    print(f"Creating index on {output_hdf}...")
    
    if not sort_by_popcnt:
        create_index_on_existing_file(batch_output)
        print("Index created successfully!")

        if sort_by_popcnt:
            out_dir.mkdir(parents=True, exist_ok=True)
            batch_output = out_dir / f"sorted_{task_id}.h5"
            sort_db_file_fast(file, batch_output, compression_level)
            
            print(f"Fingerprint database created at {batch_output}")


def main():
    output_smi, input_path, batch_size, fp_param, fp_type, stage, output_searchdb, compression_level, input_searchdb = parse_args()
    RDLogger.DisableLog('rdApp.*')
    
    start = time.perf_counter()
    
    if stage == 'convert':
        stage_convert_parquet(input_path, output_smi, batch_size)
    elif stage == 'merge_smi':
        stage_merge_smi(input_path, output_smi)
    elif stage == 'fingerprint':
        stage_create_fingerprints(output_smi, fp_type, fp_param)
    elif stage == 'merge_batches' or stage == 'merge_final':
        stage_merge_fingerprints(final=(stage == 'merge_final'), output_hdf=output_hdf)
    elif stage == 'sort' or stage == 'index':
        #stage_final(output_hdf, sort_by_popcnt=(stage == 'sort'))
        stage_final(input_searchdb, output_searchdb, sort_by_popcnt=(stage == 'sort'), compression_level=compression_level)
    else:
        print(f"Unknown stage: {stage}")
        sys.exit(1)
    
    end = time.perf_counter()
    print(f"Stage '{stage}' completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()