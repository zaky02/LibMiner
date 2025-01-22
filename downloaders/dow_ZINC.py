import os
import numpy as np

urls = np.genfromtxt('../url/ZINC-downloader-2D-txt.uri', dtype=str)
os.chdir('../ZINC20')
for i, url in enumerate(urls):
    print(f'{i+1}/{len(urls)} {url}')
    filename = os.path.basename(url)
    filename_clean = filename.replace('.txt','_clean.txt')
    cmd0 = 'echo \"ID,SMILES,dockscore\" > %s' % filename_clean
    #print(cmd0)
    os.system(cmd0)
    cmd1 = 'wget %s' % (url)
    #print(cmd1)
    os.system(cmd1)
    cmd2 = 'awk -F\'\\t\' \'NR > 1 {print $2 \",\" $1 \",0\"}\' %s >> %s' % (filename, filename_clean)
    #print(cmd2)
    os.system(cmd2)
    cmd3 = 'rm %s' % filename
    #print(cmd3)
    os.system(cmd3)
