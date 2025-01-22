import glob
import os

filenames = glob.glob('../SureChEMBL/*')
os.chdir('../SureChEMBL')
for filename in filenames:
    filename = os.path.basename(filename) 
    print(filename)
    filename_clean = filename.replace('.txt','_clean.csv')
    cmd1 = 'echo \"ID,SMILES,dockscore\" > %s' % filename_clean
    os.system(cmd1)
    cmd2 = 'awk -F\'\\t\' \'NR > 1 {print $1 \",\" $2 \",0\"}\' %s >> %s' % (filename, filename_clean)
    os.system(cmd2)
