import os
import sys
import numpy as np

urls = np.genfromtxt('/gpfs/projects/bsc72/Libraries4DSD/ZINC22/urls/clean_urls.txt', dtype=str)
print(urls)

script_path = os.getcwd()
os.chdir('/gpfs/projects/bsc72/Libraries4DSD/ZINC22/urls')
data_path =  '/gpfs/projects/bsc72/Libraries4DSD/ZINC22'
for i, url in enumerate(urls):
    print(f'{i+1}/{len(urls)} {url}')
    file_name = url.strip('/').split('/')[-1]
    new_dir = url.strip('/').split('/')[-2]
    os.system(f'wget -L {url}')

    # Prepare run file from template to decompress and parse
    partition = file_name.replace('.smi.gz','')
    runinp = open('/gpfs/projects/bsc72/Libraries4DSD/scripts/templates/run_template_ZINC22.sh', 'r')
    runout = open('/gpfs/projects/bsc72/Libraries4DSD/scripts/runs/%s.sh' % new_dir, 'w')
    for line in runinp:
        if '$PARTITION' in line:
            line = line.replace('$PARTITION', partition)
        if '$NEWDIR' in line:
            line = line.replace('$NEWDIR', new_dir)
        if '$DATAPATH' in line:
            line = line.replace('$DATAPATH', data_path)
        if '$FILE_NAME' in line:
            line = line.replace('$FILE_NAME', file_name)
        runout.write(line)
    runout.close()
    runinp.close()
    # Execute run file
    cmd2 = 'sbatch /gpfs/projects/bsc72/Libraries4DSD/scripts/runs/%s.sh' % new_dir
    print(cmd2)
    os.system(cmd2)
