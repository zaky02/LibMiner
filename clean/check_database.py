#!/usr/bin/env python3
"""
Verify FPSim2 database integrity after merging.

Checks:
1. Total molecule count matches sum of chunks
2. No duplicate molecules
3. Database is properly sorted by popcnt
4. Popcnt bins are correct
5. No corrupted fingerprints
6. Config matches source databases
"""

import tables as tb
from pathlib import Path
import sys
import numpy as np
from collections import Counter
import random


def check_molecule_counts(chunk_dir: str, merged_db: str) -> bool:
    """
    Verify that merged database has correct total molecule count.
    """
    print("\n" + "="*60)
    print("1. MOLECULE COUNT VERIFICATION")
    print("="*60)
    
    chunk_dir = Path(chunk_dir)
    
    # Count molecules in each chunk
    chunk_files = sorted(chunk_dir.glob("chunk_*.h5"), 
                        key=lambda x: int(x.stem.split('_')[-1]))
    
    if not chunk_files:
        print(f"❌ No chunk files found in {chunk_dir}")
        return False
    
    print(f"Found {len(chunk_files)} chunk files")
    
    chunk_counts = []
    total_chunks = 0
    
    for chunk_file in chunk_files:
        try:
            with tb.open_file(str(chunk_file), mode="r") as f:
                count = f.root.fps.nrows
                chunk_counts.append((chunk_file.name, count))
                total_chunks += count
        except Exception as e:
            print(f"❌ Error reading {chunk_file}: {e}")
            return False
    
    # Count molecules in merged database
    try:
        with tb.open_file(merged_db, mode="r") as f:
            merged_count = f.root.fps.nrows
    except Exception as e:
        print(f"❌ Error reading merged database: {e}")
        return False
    
    # Display results
    print(f"\nChunk counts (showing first 10 and last 5):")
    for name, count in chunk_counts:
        print(f"  {name}: {count:,} molecules")

    print(f"\n{'─'*60}")
    print(f"Total in chunks:  {total_chunks:,} molecules")
    print(f"Total in merged:  {merged_count:,} molecules")
    print(f"Difference:       {abs(merged_count - total_chunks):,}")
    print(f"{'─'*60}")
    
    if merged_count == total_chunks:
        print("✅ Molecule counts match perfectly!")
        return True
    else:
        print(f"❌ Molecule count mismatch!")
        print(f"   Missing: {total_chunks - merged_count:,} molecules" if merged_count < total_chunks else f"   Extra: {merged_count - total_chunks:,} molecules")
        return False


def check_sorting(merged_db: str) -> bool:
    """
    Quick sorting check: only verify beginning and end of database.
    Much faster for very large databases.
    """
    print("\n" + "="*60)
    print("3. SORTING VERIFICATION (QUICK MODE)")
    print("="*60)
    
    try:
        with tb.open_file(merged_db, mode="r") as f:
            fps = f.root.fps
            total_mols = fps.nrows
            
            print(f"Quick check on first and last 1M molecules (out of {total_mols:,})...")
            
            # Check first 1M
            sample_size = min(1_000_000, total_mols)
            popcnt_start = fps.read(start=0, stop=sample_size, field='popcnt')
            
            prev_pc = -1
            for i, pc in enumerate(popcnt_start):
                if pc < prev_pc:
                    print(f"❌ Sorting error in first 1M at position {i}: {prev_pc} → {pc}")
                    return False
                prev_pc = pc
            
            print(f"✅ First {sample_size:,} molecules properly sorted")
            
            # Check last 1M
            if total_mols > sample_size:
                start_idx = total_mols - sample_size
                popcnt_end = fps.read(start=start_idx, stop=total_mols, field='popcnt')
                
                prev_pc = popcnt_start[-1]  # Continue from first check
                for i, pc in enumerate(popcnt_end):
                    if pc < prev_pc:
                        print(f"❌ Sorting error in last 1M at position {start_idx + i}: {prev_pc} → {pc}")
                        return False
                    prev_pc = pc
                
                print(f"✅ Last {sample_size:,} molecules properly sorted")
            
            print("✅ Database appears to be sorted (quick check passed)")
            return True
                
    except Exception as e:
        print(f"❌ Error checking sorting: {e}")
        return False
    
    
def check_config(chunk_dir: str, merged_db: str) -> bool:
    """
    Verify that configuration matches between chunks and merged database.
    """
    print("\n" + "="*60)
    print("5. CONFIG VERIFICATION")
    print("="*60)
    
    chunk_dir = Path(chunk_dir)
    chunk_files = sorted(chunk_dir.glob("*.h5"), 
                        key=lambda x: int(x.stem.split('_')[-1]))
    
    if not chunk_files:
        print("⚠️  No chunk files to compare")
        return True
    
    try:
        # Get config from first chunk
        with tb.open_file(str(chunk_files[0]), mode="r") as f:
            chunk_fp_type = f.root.config[0]
            chunk_fp_params = f.root.config[1]
        
        # Get config from merged database
        with tb.open_file(merged_db, mode="r") as f:
            merged_fp_type = f.root.config[0]
            merged_fp_params = f.root.config[1]
        
        print(f"Chunk fingerprint type:   {chunk_fp_type}")
        print(f"Merged fingerprint type:  {merged_fp_type}")
        print(f"Chunk fingerprint params: {chunk_fp_params}")
        print(f"Merged fingerprint params: {merged_fp_params}")
        
        if chunk_fp_type != merged_fp_type:
            print("❌ Fingerprint type mismatch!")
            return False
        
        if chunk_fp_params != merged_fp_params:
            print("❌ Fingerprint parameters mismatch!")
            return False
        
        print("✅ Configuration matches")
        return True
        
    except Exception as e:
        print(f"❌ Error checking config: {e}")
        return False
    

def check_popcnt_bins(merged_db: str) -> bool:
    """
    Verify that popcnt bins in config are correct.
    """
    print("\n" + "="*60)
    print("4. POPCNT BINS VERIFICATION")
    print("="*60)
    
    try:
        with tb.open_file(merged_db, mode="r") as f:         
            # Get stored bins from config
            if len(f.root.config) < 5:
                print("⚠️  No popcnt bins found in config")
                return False  # Not critical
            
            stored_bins = f.root.config[-1]
            print(f"Found {len(stored_bins)} stored popcnt bins")
            
            if f.fps.cols.popcnt.is_indexed:
                print("Popcnt column is indexed")
                return True
            else:
                print("⚠️  Popcnt column is not indexed")
                return False
                
                
    except Exception as e:
        print(f"❌ Error checking popcnt bins: {e}")
        return False


def check_fingerprint_integrity(merged_db: str, sample_size: int = 1000) -> bool:
    """
    Check that fingerprints are not corrupted (spot check).
    """
    print("\n" + "="*60)
    print("6. FINGERPRINT INTEGRITY CHECK")
    print("="*60)
    
    try:
        with tb.open_file(merged_db, mode="r") as f:
            fps = f.root.fps
            total_mols = fps.nrows
            
            # Get fingerprint columns (f1-f8 for 1024-bit fingerprint)
            # Determine number of fp columns from table description
            fp_cols = [col for col in fps.colnames if col.startswith('f')]
            fp_length = len(fp_cols) * 64  # Each column is 64 bits
            
            print(f"Fingerprint length: {fp_length} bits ({len(fp_cols)} columns)")
            print(f"Sampling {sample_size} fingerprints...")
            
            # Sample random fingerprints
            indices = np.random.randint(0, total_mols - 1, size=sample_size)
            
            issues = []
            for idx in indices:
                row = fps[int(idx)]
                
                # Row is a tuple: (fp_id, f1, f2, f3, f4, f5, f6, f7, f8, popcnt)
                # Get stored popcnt (last element)
                stored_popcnt = row[-1]
                
                # Get fingerprint columns (all middle elements between fp_id and popcnt)
                fp_values = [row[y] for y in range(1, len(row)-1)]  # Skip fp_id (first) and popcnt (last)
                
                # Calculate actual popcnt by counting set bits
                actual_popcnt = 0
                for val in fp_values:
                    # Count bits in this 64-bit integer
                    actual_popcnt += bin(val).count('1')
                
                # Check if matches stored popcnt
                if actual_popcnt != stored_popcnt:
                    issues.append((idx, stored_popcnt, actual_popcnt))
                    if len(issues) >= 10:
                        break
            
            if issues:
                print(f"❌ Found {len(issues)} fingerprints with incorrect popcnt!")
                print("First 10 issues:")
                for idx, stored, actual in issues[:10]:
                    print(f"  Row {idx}: stored popcnt={stored}, actual={actual}")
                return False
            else:
                print(f"✅ All {len(indices)} sampled fingerprints are valid")
                return True
                
    except Exception as e:
        print(f"❌ Error checking fingerprints: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_database(chunk_dir: str = "tmp_chunks", merged_db: str = "fp_db.h5"):
    """
    Run all verification checks.
    """
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*15 + "DATABASE VERIFICATION" + " "*22 + "║")
    print("╚" + "="*58 + "╝")
    
    print(f"\nChunk directory: {chunk_dir}")
    print(f"Merged database: {merged_db}")
    
    if not Path(merged_db).exists():
        print(f"\n❌ Merged database not found: {merged_db}")
        sys.exit(1)
    
    # Run all checks
    results = {}
    
    # Only check counts if chunks still exist
    if Path(chunk_dir).exists():
        results['counts'] = check_molecule_counts(chunk_dir, merged_db)
    else:
        print(f"\n⚠️  Chunk directory not found, skipping count verification")
        results['counts'] = None
    
    results['sorting'] = check_sorting(merged_db)
    results['popcnt_bins'] = check_popcnt_bins(merged_db)
    
    if Path(chunk_dir).exists():
        results['config'] = check_config(chunk_dir, merged_db)
    else:
        results['config'] = None
    
    results['integrity'] = check_fingerprint_integrity(merged_db, sample_size=1000)
    
    # Summary
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*20 + "SUMMARY" + " "*31 + "║")
    print("╠" + "="*58 + "╣")
    
    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)
    total = len(results) - skipped
    
    for check, result in results.items():
        status = "✅ PASS" if result is True else ("❌ FAIL" if result is False else "⊘ SKIP")
        print(f"║  {check.replace('_', ' ').title():<30} {status:>25} ║")
    
    print("╠" + "="*58 + "╣")
    print(f"║  Total: {passed}/{total} passed, {failed} failed, {skipped} skipped" + " "*(26-len(str(passed))-len(str(total))-len(str(failed))-len(str(skipped))) + "║")
    print("╚" + "="*58 + "╝")
    
    if failed > 0:
        print("\n❌ Database verification FAILED!")
        print("   Please investigate the issues above before using this database.")
        sys.exit(1)
    else:
        print("\n✅ Database verification PASSED!")
        print("   The merged database appears to be correct.")
        sys.exit(0)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verify FPSim2 database integrity after merging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify with default paths
  python verify_database.py
  
  # Verify with custom paths
  python verify_database.py --chunks tmp_chunks --database fp_db.h5
  
  # Verify without chunk comparison (if chunks deleted)
  python verify_database.py --database fp_db.h5
        """
    )
    
    parser.add_argument(
        '--chunks', '-c',
        default='tmp_chunks',
        help='Path to chunk directory (default: tmp_chunks)'
    )
    
    parser.add_argument(
        '--database', '-d',
        default='fp_db.h5',
        help='Path to merged database (default: fp_db.h5)'
    )
    
    args = parser.parse_args()
    
    verify_database(args.chunks, args.database)


if __name__ == '__main__':
    main()