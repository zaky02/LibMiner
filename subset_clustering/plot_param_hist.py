import pandas as pd
from MolecularAnalysis.moldb import Mol, MolDB
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from statistics import mean


def get_moldbs(csv):

    mols = []
    df = pd.read_csv(csv)
    
    print(f"Processing file: {csv}")
    for i, row in df.iterrows():
        try:
            ident = row['ID']
            smiles = row['SMILES']
            molobj = Mol(smile=smiles, name=ident)
            mols.append(molobj)

        except Exception as e:
            print(f"Error processing row {i}: {row} | Error: {e}")
            exit()

    print(f"Retrieved moldb object for file: {csv}")
    moldbobj = MolDB(molList=mols, verbose=True)

    return moldbobj

def generate_colors_from_palette(palette_name, n):
    cmap = plt.get_cmap(palette_name)
    norm = mcolors.Normalize(vmin=0, vmax=n-1)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = [scalar_map.to_rgba(i) for i in range(n)]
    return colors


def plot_param_hist(csvs, paramater, outname,
                    transparency, bins, colors, labels):

    moldbs = []
    for csv in csvs:
        print(f"Retrieving moldb object for file: {csv}")
        moldb = get_moldbs(csv)
        if not moldb.paramaters:
            moldb._getMolsParamaters()
        moldbs.append(moldb)

    plt.figure()
    for i, moldbobj in enumerate(moldbs):
        x = [getattr(mol, paramater) for mol in moldbobj.mols]
        print(i, max(x))
        print(i, min(x))
        print(i, mean(x))
        plt.hist(x, bins=bins, alpha=transparency, color=colors[i],
                 label=labels[i])
    plt.legend()
    plt.xlabel(paramater)
    plt.savefig(outname + '.pdf')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Plot superposing paramater\
                                     histograms for multiple MolDB")
    requiredArguments = parser.add_argument_group('required arguments')
    requiredArguments.add_argument('--csvs',
                                   nargs='+',
                                   help='List of csv files')
    requiredArguments.add_argument('-o',
                                   help='histogram outname')
    parser.add_argument('-p',
                        help='paramater name',
                        default=None)
    parser.add_argument('--params',
                        help='plot list of paramaters',
                        action='store_true')
    parser.add_argument('-t',
                        help='Transparency value for the hist',
                        default=0.5)
    parser.add_argument('-b',
                        help='Bins value for the hist',
                        default=50)
    parser.add_argument('-c', nargs='+', help='List of colors')
    parser.add_argument('-l', nargs='+', help='Legend Labels')

    args = parser.parse_args()    

    # Check args
    csvs = args.csvs
    moldbs = []
    for csv in csvs:
        print(f"Retrieving moldb object for file: {csv}")
        moldb = get_moldbs(csv)
        if not moldb.paramaters:
            moldb._getMolsParamaters()
        moldbs.append(moldb)
    
    attributes = dir(moldbs[0].mols[0])
    attributes = [attr for attr in attributes if not attr.startswith('__')]
    attributes = [attr for attr in attributes if not attr.startswith('get')]

    list_of_paramaters = args.params
    if list_of_paramaters:
        print(attributes)
        exit()

    paramater = args.p
    if not paramater:
        raise ValueError('a paramater must be provided')
    if paramater not in attributes:
        _attributes = ','.join(attributes)
        raise ValueError('paramater must be: %s' % _attributes)

    outname = args.o
    transparency = float(args.t)
    bins = int(args.b)

    if not args.c:
        colors = generate_colors_from_palette('tab20', len(moldbs))
    else:
        colors = args.c

    if not args.l:
        labels = [None]*len(moldbs)
    else:
        labels = args.l

    plot_param_hist(csvs, paramater, outname,
                    transparency, bins, colors, labels)
