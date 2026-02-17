import dask.dataframe as dd
import pandas as pd
from pathlib import Path
import subprocess
import logging
from dask.distributed import Client
import shutil
from itertools import combinations
import heapq
import shutil
import os
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq

scheduler_address = os.environ["DASK_SCHEDULER_ADDRESS"]
client = Client(scheduler_address)    # Connect to that cluster
client.wait_for_workers(n_workers=1, timeout=180)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 helpers: export & sort
# ─────────────────────────────────────────────────────────────────────────────

def export_smiles_unsorted(
    db_path: list[str|Path],
    output_file: Path,
    smiles_col: str = "SMILES",
    block_size: str = "64MB",
) -> int:
    """
    Export ALL SMILES (including internal duplicates) from a parquet database
    to a plain-text file, one per line, unsorted.

    Deduplication is intentionally left to GNU sort -u, which handles it in a
    single pass during sorting — faster than doing it separately in Python.
    NaN values are dropped here because sort cannot handle them.

    Returns the original row count (before any deduplication).
    """
    logger.info(f"Exporting SMILES from {db_path}")
    df = dd.read_parquet(db_path, columns=[smiles_col], blocksize=block_size).dropna()

    df[[smiles_col]].to_csv(
        str(output_file),
        single_file=True,
        index=False,
        header=False,
        compute=True,
    )
    result = subprocess.run(
        ["wc", "-l", str(output_file)],
        check=True, capture_output=True, text=True,
    )
    original_count = int(result.stdout.split()[0])
    logger.info(
        f"  Written {original_count:,} SMILES (with duplicates) to {output_file.name}"
    )
    return original_count


def sort_file(
    input_file: Path,
    output_file: Path,
    temp_dir: Path,
    buffer_size: str = "4G",
    parallel: int = 4,
) -> int:
    """
    Sort input_file → output_file using GNU sort.
    -u deduplicates during sorting (no extra pass needed).
    Returns the number of unique lines in the output.
    """
    env = os.environ.copy()
    env["LC_ALL"] = "C" 
    
    logger.info(f"Sorting {input_file.name} → {output_file.name}")
    subprocess.run(
        ["LC_ALL=C sort", "-u",
            "-S", buffer_size,
            "--parallel", str(parallel),
            "-T", str(temp_dir),
            "-o", str(output_file),
            str(input_file),
        ],
        check=True,
        env=env,
    )
    result = subprocess.run(
        ["wc", "-l", str(output_file)],
        check=True, capture_output=True, text=True,
    )
    unique_count = int(result.stdout.split()[0])
    logger.info(f"  {unique_count:,} unique lines after sort -u")
    return unique_count


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: single-pass k-way merge → pairwise counts + parquet
# ─────────────────────────────────────────────────────────────────────────────

def kway_merge_to_parquet(
    db_sorted_files: dict[str, Path],
    output_parquet: Path,
    chunk_size: int = 10_000_000,
) -> tuple[dict[str, int], int]:
    """
    Single pass over all sorted-unique database files simultaneously via a
    min-heap.  Produces in one linear scan:

      • pairwise overlap counts  {"{db1}_{db2}": count}
      • redundant_smiles.parquet — one row per SMILES that appears in ≥2 DBs

    Complexity
    ──────────
    Time  : O(N log K)   N = total unique SMILES across all DBs, K = #DBs
    Memory: O(K + chunk_size rows)  — heap holds at most K entries at once

    Every sorted file is read exactly ONCE regardless of how many databases
    there are. 

    Parquet schema
    ──────────────
    SMILES       string   — the molecule
    Databases    string   — comma-separated sorted db_ids  e.g. "003,007,012"
    n_databases  int32    — number of databases containing this SMILES
    """
    logger.info(
        f"K-way merge of {len(db_sorted_files)} files → {output_parquet.name}"
    )

    schema = pa.schema([
        pa.field("SMILES",      pa.string()),
        pa.field("Databases",   pa.string()),
        pa.field("n_databases", pa.int32()),
    ])
    writer = pq.ParquetWriter(output_parquet, schema)

    def flush(rows: list[dict]) -> None:
        writer.write_table(
            pa.Table.from_pandas(
                pd.DataFrame(rows, columns=["SMILES", "Databases", "n_databases"]),
                schema=schema,
            )
        )

    # ── open files and seed the heap ─────────────────────────────────────────
    handles: dict[str, object] = {}
    heap: list[tuple[str, str]] = []

    for db_id, path in sorted(db_sorted_files.items()):
        f = open(path, "r")
        handles[db_id] = f
        line = f.readline().strip()
        if line:
            heapq.heappush(heap, (line, db_id))
            
    pairs = [sorted(x) for x in combinations(db_sorted_files.keys(), 2)]
    # initialise all pair counters at zero
    pair_counts: dict[str, int] = {
        f"{a}_{b}": 0 for a, b in pairs
    }

    # ── main merge loop ───────────────────────────────────────────────────────
    current_smiles: str | None = None
    current_dbs: set[str]      = set()
    buffer: list[dict]         = []
    total_redundant            = 0
    total_checked              = 0

    while heap:
        smiles, db_id = heapq.heappop(heap)
        total_checked += 1

        if total_checked % 10_000_000 == 0:
            logger.info(
                f"  {total_checked:,} processed, {total_redundant:,} redundant …"
            )

        if smiles != current_smiles:
            # ── emit the finished group ───────────────────────────────────
            if current_smiles is not None and len(current_dbs) >= 2:
                dbs_sorted = sorted(current_dbs)
                buffer.append({
                    "SMILES":      current_smiles,
                    "Databases":   ",".join(dbs_sorted),
                    "n_databases": len(dbs_sorted),
                })
                total_redundant += 1
                for a, b in [sorted(x) for x in combinations(dbs_sorted, 2)]:
                    pair_counts[f"{a}_{b}"] += 1
                if len(buffer) >= chunk_size:
                    logger.info(f"  Flushing {len(buffer):,} rows …")
                    flush(buffer)
                    buffer = []

            current_smiles = smiles
            current_dbs    = {db_id}
        else:
            current_dbs.add(db_id)

        nxt = handles[db_id].readline().strip()
        if nxt:
            heapq.heappush(heap, (nxt, db_id))

    # ── final group ───────────────────────────────────────────────────────────
    if current_smiles is not None and len(current_dbs) >= 2:
        dbs_sorted = sorted(current_dbs)
        buffer.append({
            "SMILES":      current_smiles,
            "Databases":   ",".join(dbs_sorted),
            "n_databases": len(dbs_sorted),
        })
        total_redundant += 1
        for a, b in [sorted(x) for x in combinations(dbs_sorted, 2)]:
            pair_counts[f"{a}_{b}"] += 1

    if buffer:
        flush(buffer)

    for f in handles.values():
        f.close()
    writer.close()

    logger.info(f"✅ {total_redundant:,} redundant SMILES → {output_parquet.name}")
    return pair_counts, total_redundant


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compute_pairwise_overlaps(
    db_paths: dict[str, list[str|Path]],
    work_dir: Path,
    output_parquet: Path,
    smiles_col: str = "SMILES",
    block_size: str = "64MB",
    sort_buffer: str = "4G",
    sort_parallel: int = 4,
    chunk_size: int = 10_000_000,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], int]:
    """
    Full pipeline:
      1. Export each DB's SMILES (with duplicates) to a text file
      2. Sort + deduplicate each file with GNU sort -u
      3. Single-pass k-way merge → pairwise counts + redundant_smiles.parquet

    Returns
    -------
    pair_counts     : {"{db1}_{db2}": overlap_count}
    original_counts : {db_id: row_count_before_any_dedup}
    unique_counts   : {db_id: unique_smiles_count_after_sort_-u}
    total_redundant : number of rows written to the parquet
    """
    work_dir = Path(work_dir)
    temp_dir = work_dir / "sort_temp"
    temp_dir.mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("STEP 1 — Export (original counts) & sort -u (unique counts)")
    logger.info("=" * 80)

    db_sorted_files: dict[str, Path] = {}
    original_counts: dict[str, int]  = {}
    unique_counts:   dict[str, int]  = {}

    for db_id, db_path in sorted(db_paths.items()):
        sorted_file   = work_dir / f"db_{db_id}_sorted.txt"
        unsorted_file = work_dir / f"db_{db_id}_unsorted.txt"

        if sorted_file.exists():
            # sorted file exists — reuse it but we still need the original count
            result = subprocess.run(
                ["wc", "-l", str(sorted_file)],
                check=True, capture_output=True, text=True,
            )
            unique_n = int(result.stdout.split()[0])
            logger.info(
                f"{db_id}: reusing sorted file ({unique_n:,} unique lines). "
                f"Re-reading parquet for original count …"
            )
            # original count: fast metadata read, no full scan
            original_n = dd.read_parquet(
                db_path, columns=[smiles_col], blocksize=block_size
            )[smiles_col].count().compute()
        else:
            original_n = export_smiles_unsorted(
                db_path, unsorted_file, smiles_col, block_size
            )
            unique_n = sort_file(
                unsorted_file, sorted_file, temp_dir, sort_buffer, sort_parallel
            )
            unsorted_file.unlink()
            logger.info(
                f"{db_id}: original={original_n:,}  unique={unique_n:,}  "
                f"internal_duplicates={original_n - unique_n:,}"
            )

        original_counts[db_id]  = original_n
        unique_counts[db_id]    = unique_n
        db_sorted_files[db_id]  = sorted_file

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    logger.info("=" * 80)
    logger.info("STEP 2 — Single-pass k-way merge")
    logger.info("=" * 80)

    pair_counts, total_redundant = kway_merge_to_parquet(
        db_sorted_files, output_parquet, chunk_size
    )

    return pair_counts, original_counts, unique_counts, db_sorted_files


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description=(
            "Stable pairwise overlap computation for billions of molecules. "
            "Uses GNU sort -u + single-pass k-way merge instead of Dask merge."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python system_sort_overlap.py Molecular_database/HAC_21
    python system_sort_overlap.py Molecular_database/HAC_21 -b 16G -p 16
    python system_sort_overlap.py Molecular_database/HAC_21 -o results/hac21.parquet
    """,
    )
    
    parser.add_argument('-dp','--database_path', type=str, 
                        help='The folder path for the database', required=False,
                        default='Molecular_database')
    parser.add_argument("-o", "--output", default="redundant_smiles.parquet",
                        help="Output parquet (default: <work_dir>/redundant_smiles.parquet)")
    parser.add_argument("-b", "--buffer-size", default="4G",
                        help="RAM for GNU sort, e.g. 4G, 8G, 16G  (default: 4G)")
    parser.add_argument("-p", "--parallel", type=int, default=None,
                        help="Sort threads (default: CPU count)")
    parser.add_argument("-s", "--smiles-col", default="SMILES",
                        help="SMILES column name (default: SMILES)")
    parser.add_argument("--block-size", default="64MB",
                        help="Dask read blocksize (default: 64MB)")
    parser.add_argument("--chunk-size", type=int, default=10_000_000,
                        help="Parquet flush interval in rows (default: 10_000_000)")

    
    args = parser.parse_args()
    
    from utils import convert_folder
    
    database_path = Path(args.database_path)
    
    if not database_path.exists():
        parser.error(f"HAC folder does not exist: {database_path}")
    
    work_dir = Path(f"{database_path}/pairwise_analysis")
    work_dir.mkdir(parents=True, exist_ok=True)
    sort_parallel = args.parallel if args.parallel else os.cpu_count()
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Data Folder: {database_path}")
    print(f"Work Directory: {work_dir}")
    print(f"Sort Settings: {args.buffer_size} buffer, {sort_parallel} parallel threads")
    
    progress = work_dir / "progress_pairwise.txt"
    progress.touch(exist_ok=True)
    hacs = sorted(database_path.glob("HAC_*"), key=lambda x: int(x.name.split("_")[-1]))
    
    for hac_folder in hacs:
        logger.info(f"\nProcessing HAC folder: {hac_folder}")
        hac = hac_folder.name.split("_")[-1]
        
        if f"HAC {hac} done" in progress.read_text():
            print(f"HAC {hac} already done, skipping.")     
            continue
        
        db_paths = convert_folder(hac_folder) 
        output_parquet = work_dir / hac / args.output
        output_parquet = Path(output_parquet)
        
        pair_counts, original_counts, unique_counts, db_sorted_files =  compute_pairwise_overlaps(
                                                                        db_paths= db_paths,
                                                                        work_dir= work_dir,
                                                                        output_parquet= output_parquet,
                                                                        smiles_col= args.smiles_col,
                                                                        block_size= args.block_size,
                                                                        sort_buffer= args.buffer_size,
                                                                        sort_parallel= sort_parallel,
                                                                        chunk_size= args.chunk_size)
        
        unique_counts_series = pd.Series(unique_counts, name="after_internal_deduplication")
        processed_counts_series = pd.Series(original_counts, name="dropna_counts")
        internal_counts = pd.concat([processed_counts_series, unique_counts_series], axis=1)
        internal_counts.to_csv(work_dir/hac/"internal_duplication.csv")
        pd.Series(pair_counts, name="pairwise_overlaps").to_csv(work_dir/hac/"pairwise_duplication.csv")
        
        with open(progress, "a") as f:
            f.write(f"HAC {hac} done\n")
        
        # Clean up sorted files if requested
        print("\n" + "="*80)
        print("CLEANING UP SORTED FILES")
        print("="*80)
        for db_id, sorted_file in db_sorted_files.items():
            if sorted_file.exists():
                sorted_file.unlink()
                print(f"  Deleted: {sorted_file.name}")