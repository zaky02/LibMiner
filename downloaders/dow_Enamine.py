import os
import numpy as np

# Links to Enamine partitions
urls = np.genfromtxt('../url/EnamineReal-downloader2.txt', dtype=str)

script_path = os.getcwd()
os.chdir('../Enamine_Real')
data_path = os.getcwd()
for i, url in enumerate(urls):
    # Prepare output file with only ID, SMILE and dockscore rows
    print(f'{i+1}/{len(urls)} {url}')
    filename_compress = os.path.basename(url)
    filename_clean = filename_compress.replace('.cxsmiles.bz2','_clean.csv')
    cmd0 = 'echo \"ID,SMILES,dockscore\" > %s' % filename_clean
    print(cmd0)
    os.system(cmd0)

    # Download partition
    cmd1 = 'wget %s --no-check-certificat' % url
    print(cmd1)
    os.system(cmd1)
    
    # Prepare run file from template to decompress
    # and parse to output file
    partition = filename_compress.split('.')[0]
    filename_decompress = filename_compress.replace('.bz2','')
    runinp = open('../scripts/templates/run_template_Enamine.sh',
            'r')
    runout = open('../scripts/runs/%s.sh' % partition, 'w')
    for line in runinp:
        if '$PARTITION' in line:
            line = line.replace('$PARTITION', partition)
        if '$DATAPATH' in line:
            line = line.replace('$DATAPATH', data_path)
        if '$COMPRESS' in line:
            line = line.replace('$COMPRESS', filename_compress)
        if '$DECOMPRESS' in line:
            line = line.replace('$DECOMPRESS', filename_decompress)
        if '$CLEAN' in line:
            line = line.replace('$CLEAN', filename_clean)
        runout.write(line)
    runout.close()
    runinp.close()

    # Execute run file
    cmd2 = 'sbatch ../scripts/runs/%s.sh ' % partition
    print(cmd2)
    os.system(cmd2)
