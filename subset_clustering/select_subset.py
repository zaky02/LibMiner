import pandas as pd


def select_subset(csv, size, db):
    
    df = pd.read_csv(csv)
    filtered_df = df[df["final_representative"] == 1]
    subset_df = filtered_df.sample(n=size, random_state=1234)
    subset_df.to_csv(f"{db}_subset_{size}.csv", index=False)

    """
    cluster_counts = df['final_cluster'].value_counts().reset_index()
    cluster_counts.columns = ['final_cluster', 'count']
    cluster_counts = cluster_counts.sort_values(by='count', ascending=False)
    print(cluster_counts)
    
    top_clusters = cluster_counts.head(size)['final_cluster']
    
    filtered_df = df[
        (df['final_cluster'].isin(top_clusters)) & 
        (df['final_representative'] == 1)
    ]
    
    filtered_df.to_csv(f"{db}_subset_{size}.csv", index=False)
    """


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--csv',
                        help='CSV file to extract the subset of representative molecules')
    parser.add_argument('--size',
                        help='Size of the subset to extract',
                        type=int,
                        default=100000)
    parser.add_argument('--db',
                        help='Name of the database we are processing to add it to the output csv file name.')

    args = parser.parse_args()

    csv = args.csv
    size = int(args.size)
    db = args.db

    select_subset(csv, size, db)
