import os
import numpy as np

total_files = 13085
files_per_batch = 25
full_batches = int(total_files / files_per_batch)
partial_files = total_files % files_per_batch

path = '/gpfs/projects/bsc72/Libraries4DSD/ZINC22_partitioned_2M_HMW'
out_path = '/gpfs/projects/bsc72/Libraries4DSD/ZINC22_HMW_batches'

files = sorted(os.listdir(path))
file_index = 0


for i in range(full_batches):
    f = open(f'{out_path}/ZINC22_HMW_2M_batch_{i+1}', 'w')
    for j in range(files_per_batch):
        file_name = files[file_index] 
        file_index += 1
        f.write(f'{path}/{file_name}\n')
    f.close()

if partial_files > 0:
    f = open(f'{out_path}/ZINC22_HMW_2M_batch_{i+2}', 'w')
    for j in range(partial_files):
        file_name = files[file_index]
        file_index += 1
        f.write(f'{path}/{file_name}\n')
    f.close()
