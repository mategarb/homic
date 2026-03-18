import numpy as np
import pandas as pd
import statistics
import seaborn as sns
import os
import subprocess
import shutil
import sys
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src') 
from homic import file_readers, kraken2, process_data

#####################################################################################################

# define paths
path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes"
db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/hs_wgs"

# samples id 
samps = ["KP005", "KP012", "KP013", "KP016", "KP024", "KP025", "KP026", "KP027", "KP029","KP033","KP035","KP038","KP040","KP046","KP047","KP048"]

path_blast ="/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/blast/" + samps[0] + "_blastn_report.txt"
df = process_data.read_n_clean_blastn(path_blast, top_hits = False, evalue = 0.05, drop_sp = False)
print(df)

counts = df["species"].value_counts()

print(counts)
#####################################################################################################
### trimming ###
if False:
    print("Trimming and filtering...")
    for idx, samp in enumerate(samps):
        file1 = path + "/016-Meta-" + str(idx+1) + "/" + samp + "_S" + str(idx+1) + "_L001_R1_001.fastq.gz"
        file2 = path + "/016-Meta-" + str(idx+1) + "/" + samp + "_S" + str(idx+1) + "_L001_R2_001.fastq.gz"
        process_data.trim_decon(db_path, file1, file2, ada_path)

#####################################################################################################
### running QC after trimming ###
#if False: