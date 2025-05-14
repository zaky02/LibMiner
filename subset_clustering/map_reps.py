import os
import glob
import pandas as pd

def process_mapping(mapping_df):
    grouped = mapping_df.groupby('source')

    for source_file, df_group in grouped:
        print(f"Processing: {source_file}")

        if not os.path.exists(source_file):
            print(f"File not found: {source_file}")
            continue

        original_df = pd.read_csv(source_file)

        merge_keys = ['ID', 'SMILES', 'dockscore']
        updated_df = pd.merge(original_df, df_group, on=merge_keys, how='left', suffixes=('', '_new'))

        for col in df_group.columns:
            if col not in merge_keys and col != 'source':
                new_col = f"{col}_new"
                if new_col in updated_df.columns:
                    updated_df[col] = updated_df[new_col].combine_first(updated_df[col])
                    updated_df.drop(columns=[new_col], inplace=True)

        updated_df.to_csv(source_file, index=False)

def parallel_mapping(input_dir):
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

    for csv_file in csv_files:
        print(f"Processing mapping file: {csv_file}")
        mapping_df = pd.read_csv(csv_file)
        process_mapping(mapping_df)

    print("All files processed!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    requiredArguments = parser.add_argument_group('Required Arguments')
    requiredArguments.add_argument('--input_dir',
                                   help='Input directory which contains the csv files to map.',
                                   required=True)

    args = parser.parse_args()
    input_dir = args.input_dir
    
    parallel_mapping(input_dir)
