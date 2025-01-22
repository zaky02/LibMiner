import os
import numpy as np

total_files = 884
files_per_batch = 28
full_batches = int(total_files / files_per_batch)
partial_files = total_files % files_per_batch

prefix = 'ZINC20_partition'
path = '/gpfs/projects/bsc72/Libraries4DSD/ZINC20_partitioned'
out_path = '/gpfs/projects/bsc72/Libraries4DSD/ZINC20_batches'
counter = 0

for i in range(full_batches):
    f = open(f'{out_path}/ZINC20_batch_{i+1}', 'w')
    for j in range(files_per_batch):
        counter += 1
        f.write(f'{path}/{prefix}_{counter}.csv\n')
    f.close()

if partial_files > 0:
    f = open(f'{out_path}/ZINC20_batch_{i+2}', 'w')
    for j in range(partial_files):
        counter += 1
        f.write(f'{path}/{prefix}_{counter}.csv\n')
    f.close()
