import numpy as np
import pandas as pd
import statistics
import seaborn as sns
import os
import subprocess
import shutil
import plotly.express as px
import sys
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src') 
from homic import file_readers, kraken2, process_data
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt

# define paths
path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes"
db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/hs_wgs"

# samples id 
all_samps = ["KP005", "KP012", "KP013", "KP016", "KP024", "KP025", "KP026", "KP027", "KP029","KP033","KP035","KP038","KP040","KP046","KP047","KP048"]

## loading metadata
#meta = pd.read_csv("/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/metadata_kp.csv")
#meta = meta.loc[meta['id'].isin(all_samps)] # keep only the sequenced ones

# 1. shuffling reads
reads_n = []
spec_n = []
idx=7 # sample index

#for idx, samp in enumerate(all_samps): 

samp=all_samps[idx]
reads_r1 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes/016-Meta-" + str(idx+1) + "/" + samp + "_S" + str(idx+1) + "_L001_k2_1.fastq"
reads_r2 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes/016-Meta-" + str(idx+1) + "/" + samp + "_S" + str(idx+1) + "_L001_k2_2.fastq"

tmp_r1 = file_readers.fastq(reads_r1)
all_reads = len(tmp_r1[0])
print(all_reads)

for Nreads in [10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1250000, 1500000, 1750000]:
    
    # 1. create new fastqs with N reads
    file_readers.save_pfastq_Nreads(reads_r1,reads_r2, Nreads)
    
    # 2. assembly
    path2 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes/016-Meta-" + str(idx+1)
    file1 = path2 + "/" + samp + "_S" + str(idx+1) + "_L001_k2_1_" + str(Nreads) + ".fastq"
    file2 = path2 + "/" + samp + "_S" + str(idx+1) + "_L001_k2_2_" + str(Nreads) + ".fastq"
    process_data.assemble_decon(path2, db_path, file1, file2, samp)
    
    # 3. blastn
    
    # Add BLAST+ bin directory to PATH
    os.environ["PATH"] = "/gpfs/commons/home/mgarbulowski/ncbi-blast-2.16.0+/bin:" + os.environ["PATH"]
    
    fa_path = path2 + "/metagenome_contigs_"  + samp + ".fastq"
    #bdb_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/GbBacU/all_seqs.fna"
    bdb_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/nt_blast/nt_blast"
    out_path = path2 + "/" + samp + "_" + str(Nreads) + "_blastn_report.txt"
    process_data.run_blastn(fa_path, bdb_path, out_path, nthreads=48, evalue=1e-6, max_ts=5, max_h=1)
    
    
    os.remove(file1)
    os.remove(file2)
    os.remove(fa_path)