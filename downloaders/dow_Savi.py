import os 
import numpy as np

urls = np.genfromtxt('../url/SAVI-downloader.txt', dtype=str)
print(urls)

script_path = os.getcwd()
os.chdir('../Savi')
data_path = os.getcwd()
for i, url in enumerate(urls):
    print(f'{i+1}/{len(urls)} {url}')
    file_name = os.path.basename(url)
    os.system(f'wget -L {url}')
   
    # Prepare run file from template to decompress and parse
    partition = file_name.replace('.tar','')
    runinp = open('../scripts/templates/run_template_Savi.sh', 'r')
    runout = open('../scripts/runs/%s.sh' % partition, 'w')
    for line in runinp:
        if '$PARTITION' in line:
            line = line.replace('$PARTITION', partition)
        if '$DATAPATH' in line:
            line = line.replace('$DATAPATH', data_path)
        if '$FILE_NAME' in line:
            line = line.replace('$FILE_NAME', file_name)
        runout.write(line)
    runout.close()
    runinp.close()
    # Execute run file
    cmd2 = 'sbatch ../scripts/runs/%s.sh ' % partition
    print(cmd2)
    os.system(cmd2)
