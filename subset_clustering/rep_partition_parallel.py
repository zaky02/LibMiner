import os
import glob
import pandas as pd
import multiprocessing

def process_rep_file(file):
    df = pd.read_csv(file)
    df_rep = df[df["partition_representative"] == 1]
    df_rep['source'] = file
    print(f"Processing file: {file} \n With {len(df_rep)} representatives")
    
    return df_rep

def process_final_file(file):
    df = pd.read_csv(file)
    df_final = df[df["representative_representative"] == 1]
    df_final['source'] = file
    print(f"Processing file: {file} \n With {len(df_final)} representatives")

    return df_final

def rep_part(input_dir, output_dir, part_size, pref, cores):
    
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

    representatives = []
    part_count = 0
    current_size = 0
    
    with multiprocessing.Pool(int(cores)) as pool:
        for df in pool.imap_unordered(process_rep_file, csv_files):

            print(f"Processing batch of dfs from pool of workers")
            representatives.append(df)
            current_size += len(df)
            
            print("Checking if the combined dataframes exceed the partition size")
            print(f"Current size of dataframes: {current_size}")
            
            while current_size >= part_size:
                part_count += 1
                combined_batch = pd.concat(representatives, ignore_index=True)
                batch_to_save = combined_batch.iloc[:part_size]
                remaining_data = combined_batch.iloc[part_size:]

                current_size = len(remaining_data)
                representatives = [remaining_data]

                output_file = os.path.join(output_dir, f'{pref}_partition_{part_count}.csv')
                batch_to_save.to_csv(output_file, index=False)
                print(f'Saved {output_file}')

        if representatives:
            combined_batch = pd.concat(representatives, ignore_index=True)
            part_count += 1
            output_file = os.path.join(output_dir, f'{pref}_partition_{part_count}.csv')
            combined_batch.to_csv(output_file, index=False)
            print(f'Saved {output_file}')
    print("PARTITION DONE!!!")

def final_part(input_dir, output_dir, pref, cores):
    
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

    representatives = []
    
    with multiprocessing.Pool(int(cores)) as pool:
        for df in pool.imap_unordered(process_final_file, csv_files):

            print(f"Processing batch of dfs from pool of workers")
            representatives.append(df)
            
        output_file = os.path.join(output_dir, f'{pref}_final_clust.csv')
        representatives = pd.concat(representatives, ignore_index=True)
        representatives.to_csv(output_file, index=False)
        print(f'Saved {output_file}')
    print("PARTITION DONE!!!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Script that partitions molecular DBs into smaller subsets.')
    requiredArguments = parser.add_argument_group('Required Arguments')
    requiredArguments.add_argument('--input_dir',
                                   help='Input directory where all DB csv files are stored.',
                                   required=True)
    requiredArguments.add_argument('--output_dir',
                                   help='Output directory where all partitions will be stored.',
                                   required=True)
    parser.add_argument('--part_size',
                        type=int,
                        help='Size of the molecular DB partitions.')
    requiredArguments.add_argument('--pref',
                                   help='Prefix for the output partitioned csv file name.',
                                   required=True)
    requiredArguments.add_argument('--clustering',
                                   help='Type of clustering: representative or final',
                                   required=True)
    requiredArguments.add_argument('--cores',
                                   help='Number of cores to parallelise on',
                                   type=int,
                                   required=True)

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    part_size = args.part_size
    pref = args.pref
    clustering = args.clustering
    cores = args.cores

    if clustering == 'representative':
        rep_part(input_dir, output_dir, part_size, pref, cores)
    elif clustering == 'final':
        final_part(input_dir, output_dir, pref, cores)
