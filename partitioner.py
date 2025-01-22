import pandas as pd
import glob
import os

def partition_DB(input_dir, output_dir, part_size, pref, remove):
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    part_data = []
    part_count = 0
    current_part_size = 0

    for file in csv_files:
        df = pd.read_csv(file)
        part_data.append(df)
        current_part_size += len(df)
    
        while int(current_part_size) >= int(part_size):
            combined_batch = pd.concat(part_data, ignore_index=True)
            batch_to_save = combined_batch.iloc[:part_size]
            remaining_data = combined_batch.iloc[part_size:]
    
            part_count += 1
            output_file = os.path.join(output_dir,
                                       f'{pref}_partition_{part_count}.csv')
            print(f'Saved {output_file}')
            batch_to_save.to_csv(output_file, index=False)
    
            part_data = [remaining_data]
            current_part_size = len(remaining_data)
        
        if remove:
            os.remove(file)
    
    if current_part_size > 0:
        combined_batch = pd.concat(part_data, ignore_index=True)
        part_count += 1
        output_file = os.path.join(output_dir,
                                   f'{pref}_partition_{part_count}.csv')
        print(f'Saved {output_file}')
        combined_batch.to_csv(output_file, index=False)


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
    requiredArguments.add_argument('--part_size',
                                   type=int,
                                   help='Size of the molecular DB partitions.',
                                   required=True)
    requiredArguments.add_argument('--pref',
                                   help='Prefix for the output partitioned csv file name.',
                                   required=True)
    parser.add_argument('--remove',
                        help='Flag that when activated removes the used csv files in order to avoid duplication.',
                        action='store_true')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    part_size = args.part_size
    pref = args.pref
    remove = args.remove

    partition_DB(input_dir, output_dir, part_size, pref, remove)
