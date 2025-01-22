import argparse
from rdkit import Chem

parser = argparse.ArgumentParser()
parser.add_argument('sdf')

args = parser.parse_args()
sdffile = args.sdf

#From sdf to csv only keeping smile and id
csvfile = sdffile.replace('.sdf','_clean.csv')
fout = open(csvfile, 'w')
fout.write('ID,SMILES,dockscore\n')

supplier = Chem.SDMolSupplier(sdffile)
for j, mol in enumerate(supplier):
    name = None
    smiles = None
    try:
        name = mol.GetProp('PUBCHEM_COMPOUND_CID')
        smiles = mol.GetProp('PUBCHEM_OPENEYE_CAN_SMILES')
    except:
        print('compound %d not Can. Smiles'%j)
    if name and smiles:
        fout.write('CID_%s,%s,0\n' % (name,smiles))
fout.close()
