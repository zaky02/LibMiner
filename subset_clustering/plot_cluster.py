import pandas as pd
from MolecularAnalysis.moldb import Mol, MolDB
from MolecularAnalysis.analysis import plot

def get_moldbs(csv):
    
    mols = []
    df = pd.read_csv(csv)
    # df = df[:100000]
    print(f"Processing file: {csv}")
    for i, row in df.iterrows():
        try:
            ident = row['ID']
            smiles = row['SMILES']
            molobj = Mol(smile=smiles, name=ident)
            mols.append(molobj)
            
        except:
            print(f"Error processing row {i}: {row}")
            exit()
    
    print(f"Retrieved moldb object for file: {csv}")
    moldbobj = MolDB(molList=mols, verbose=True)

    return moldbobj

def plot_cluster(csvs, sizes, markers, alphas, colors, labels,
                 perplexity, o_tsne, o_umap, o_trimap, rand_max,
                 n_iter, min_dist, n_neighbors):
    
    print("Starting tsne plotting")
    moldbs = []
    for csv in csvs:
        print(f"Retrieving moldb object for file: {csv}")
        moldb = get_moldbs(csv)
        moldbs.append(moldb)
    
    if o_tsne:
        print("Plotting the tSNE plot")
        # Plot tSNE
        plot.plotTSNE(dbs=moldbs,
                      sizes=sizes,
                      linewidth=0,
                      markers=markers,
                      alphas=alphas,
                      names=labels,
                      colors=colors,
                      perplexity=perplexity,
                      n_iter=n_iter,
                      random_max=rand_max,
                      output=o_tsne,
                      figsize=(8, 8))
        print("tSNE plot done!!")

    if o_umap:
        print("Plotting the UMAP plot")
        # Plot UMAP
        plot.plotUMAP(dbs=moldbs,
                       names=labels,
                       sizes=sizes,
                       linewidth=0,
                       markers=markers,
                       alphas=alphas,
                       colors=colors,
                       n_epochs=n_iter,
                       random_max=rand_max,
                       output=o_umap,
                       min_dist=min_dist,
                       n_neighbors=n_neighbors)
        print("UMAP plot done!!")

    if o_trimap:
        print("Plotting the triMAP plot")
        # Plot triMAP
        plot.plotTrimap(dbs=moldbs,
                        names=labels,
                        output=o_trimap,
                        random_max=rand_max,
                        colors=colors,
                        sizes=sizes,
                        alphas=alphas,
                        markers=markers,
                        figsize=(8,8),
                        linewidth=0,
                        n_iters=n_iter)
        print("triMAP plot done!!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='This script generates tsne plots from given csv files.')
    requiredArguments = parser.add_argument_group('required arguments')
    requiredArguments.add_argument('--csvs',
                                   nargs='+',
                                   help='moldbs',
                                   required=True)
    parser.add_argument('--sizes',
                        nargs='+',
                        help='List of sizes',
                        default=None)
    parser.add_argument('--colors',
                        nargs='+',
                        help='List of colors',
                        default=None)
    parser.add_argument('--labels',
                        nargs='+',
                        help='List of labels',
                        default=None)
    parser.add_argument('--markers',
                        nargs='+',
                        help='List of markers',
                        default=None)
    parser.add_argument('--alphas',
                        nargs='+',
                        help='List of alphas',
                        default=None)
    parser.add_argument('-p', '--perplexity',
                        help='Set perplexity value',
                        default=30)
    parser.add_argument('--o_tsne',
                        help='tSNE plot output name')
    parser.add_argument('--o_umap',
                        help='UMAP plot output name')
    parser.add_argument('--o_trimap',
                        help='triMAP plot output name')
    parser.add_argument('--rand_max',
                        help='',
                        default=500)
    parser.add_argument('--n_iter',
                        help='',
                        default=1000)
    parser.add_argument('--min_dist',
                        help='',
                        default=0.1)
    parser.add_argument('--n_neighbors',
                        help='',
                        default=100)
    args = parser.parse_args()


    # Set arguments
    csvs = args.csvs
    sizes = args.sizes
    markers = args.markers
    alphas = args.alphas
    colors = args.colors
    labels = args.labels
    perplexity = int(args.perplexity)
    o_tsne = args.o_tsne
    o_umap = args.o_umap
    o_trimap = args.o_trimap
    rand_max = int(args.rand_max)
    n_iter = int(args.n_iter)
    min_dist = float(args.min_dist)
    n_neighbors = int(args.n_neighbors)

    plot_cluster(csvs, sizes, markers, alphas, colors, labels,
                 perplexity, o_tsne, o_umap, o_trimap, rand_max,
                 n_iter, min_dist, n_neighbors)
