import os
import numpy as np
from rdkit import Chem

# Links to PubChem partitions
urls = np.genfromtxt('../url/PubChem-downloader.txt', dtype=str)

script_path = os.getcwd()
os.chdir('../PubChem')
data_path = os.getcwd()
for i, url in enumerate(urls):
    # Prepare output file with only ID, SMILE and dockscore rows
    print(f'{i+1}/{len(urls)} {url}')
    filename_compress = os.path.basename(url)
    filename_uncompress = filename_compress.replace('.sdf.gz','.sdf')
    
    # Download partitions
    cmd1 = 'wget %s' % url
    print(cmd1)
    os.system(cmd1)
   

    # Prepare run file from template to decompress and parse
    partition = filename_compress.split('.')[0]
    runinp = open('../scripts/templates/run_template_PubChem.sh', 'r')
    runout = open('../scripts/runs/%s.sh' % partition, 'w') 
    for line in runinp:
        if '$PARTITION' in line:
            line = line.replace('$PARTITION', partition)
        if '$DATAPATH' in line:
            line = line.replace('$DATAPATH', data_path)
        if '$COMPRESS' in line:
            line = line.replace('$COMPRESS', filename_compress)
        if '$UNCOMPRESS' in line:
            line = line.replace('$UNCOMPRESS', filename_uncompress)
        runout.write(line)
    runout.close()
    runinp.close()

    # Execute run file
    cmd2 = 'sbatch ../scripts/runs/%s.sh ' % partition
    print(cmd2)
    os.system(cmd2)
