import pandas as pd
import glob
import os
import random
import multiprocessing as mp

# User-defined params
directory = "/gpfs/projects/bsc72/Libraries4DSD/Enamine_Real_partitioned_2M"
output_file = "random_subset_100k.csv"
total_rows = 50000
n_workers = 80

# Find all CSV files
csv_files = glob.glob(os.path.join(directory, "*.csv"))
if not csv_files:
    print("No CSV files found.")
    exit()

def worker(file_list, result_queue, seen_ids):
    while True:
        try:
            file = random.choice(file_list)
            df = pd.read_csv(file, engine="python")

            if df.empty:
                continue

            row = df.sample(n=1).iloc[0]
            row_id = row['ID']

            # Use a shared dictionary to avoid duplicate rows by ID
            if row_id not in seen_ids:
                seen_ids[row_id] = 1
                result_queue.put(row.to_dict())

        except Exception as e:
            print(f"Worker error: {e}")
            continue

def collect_rows(csv_files, total_rows, n_workers):
    manager = mp.Manager()
    result_queue = manager.Queue()
    seen_ids = manager.dict()

    processes = []
    for _ in range(n_workers):
        p = mp.Process(target=worker, args=(csv_files, result_queue, seen_ids))
        p.start()
        processes.append(p)

    print(f"Started {n_workers} workers. Collecting {total_rows} unique rows...")

    collected = []
    while len(collected) < total_rows:
        try:
            row = result_queue.get(timeout=10000)
            collected.append(row)
            print(len(collected))

            if len(collected) % 5000 == 0:
                print(f"Collected {len(collected)}/{total_rows} rows...")

        except Exception as e:
            print(f"Timeout waiting for row: {e}")
            break

    # Stop all workers
    for p in processes:
        p.terminate()

    return pd.DataFrame(collected)

# Run collection
final_df = collect_rows(csv_files, total_rows, n_workers)

# Save output
final_df.to_csv(output_file, index=False)
print(f"Saved {len(final_df)} rows to {output_file}")
