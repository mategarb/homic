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
import glob
import gzip

#### older settings, single batched ########

# top 3 mid
#5: [10000, 50000, 100000, 250000, 500000, 750000, 1000000]
#6: [10000, 50000, 100000, 250000, 500000, 750000, 1000000]
#12: [10000, 50000, 100000, 250000, 500000, 750000, 950000]

# top 3 bottom
#1: [10000, 50000, 100000, 250000, 400000]
#10: [10000, 50000, 100000, 250000, 500000, 750000]
#15: [10000, 50000, 100000, 250000, 500000, 750000]

# top 3 based on high depth
# 2: [10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1250000]
# 7: [10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1250000, 1500000, 1750000]
# 9: [10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1200000]

############################################

#### Add BLAST+ bin directory to PATH
os.environ["PATH"] = "/gpfs/commons/home/mgarbulowski/ncbi-blast-2.16.0+/bin:" + os.environ["PATH"]

#### define paths
n = 12 ## no of file to run
path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/saturation_analysis" + str(n)
db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/human_hg38"
ref_file = "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/GRCh38.fna"

main_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/"
fasta_path = main_path + "dna_fragments.fa"
fasta_rc_path = main_path + "dna_fragments_rc.fa"

#### samples id 
all_samps = ["KP003", "KP027", "KP013", "KP026", "KP035", "KP012", "KP010", "KP052", "KP049","KP040","KP048","KP005"] # all batches x3

#all_samps = ["KP003","KP004","KP008","KP010","KP011","KP021","KP037","KP041","KP049","KP052"]# batch 2 + 4

## loading metadata
#meta = pd.read_csv("/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/metadata_kp.csv")
#meta = meta.loc[meta['id'].isin(all_samps)] # keep only the sequenced ones

reads_n = []
spec_n = []
idx = n - 1 # sample index from all_samps

#for idx, samp in enumerate(all_samps): 

## read input files
samp = all_samps[idx]

print(samp)
print(path)

## unzip files
reads_file_1 = glob.glob(path + "/" + samp + "_S" + "*" + "_L001_R1_001.fastq.gz")
reads_file_2 = glob.glob(path + "/" + samp + "_S" + "*" + "_L001_R2_001.fastq.gz")
reads_file_1 = reads_file_1[0]
reads_file_2 = reads_file_2[0]

with gzip.open(reads_file_1, "rb") as f_in, open(os.path.splitext(reads_file_1)[0], "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)

with gzip.open(reads_file_2, "rb") as f_in, open(os.path.splitext(reads_file_2)[0], "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)
    
reads_r1 = glob.glob(path + "/" + samp + "_S" + "*" + "_L001_R1_001.fastq")
reads_r2 = glob.glob(path + "/" + samp + "_S" + "*" + "_L001_R2_001.fastq")

reads_r1 = reads_r1[0]
reads_r2 = reads_r2[0]

tmp_r1 = file_readers.fastq(reads_r1)
all_reads1 = len(tmp_r1[0])
print("No. of R1 reads")
print(all_reads1)

tmp_r2 = file_readers.fastq(reads_r2)
all_reads2 = len(tmp_r2[0])
print("No. of R2 reads")
print(all_reads2)

## cleaning, done once
if True:
    print("Running atropos for:")
    print(reads_r1)
    print(reads_r2)
    process_data.run_atropos_pe(reads_r1, reads_r2, fasta_path, fasta_rc_path, nhead=20, ntail=10, minlen=50, minqual=20, error_rate=0.1, threads=32)
    print("Atropos done!")
    
file1 = reads_r1.replace(".fastq", "_atroped_1.fastq")
file2 = reads_r2.replace(".fastq", "_atroped_2.fastq")

reads_Ns = [10000, 50000, 100000, 500000, 750000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000, 5500000, 6000000]

#reads_Ns = [4500000, 5000000, 5500000, 6000000]
#reads_Ns = [750000] # remember it is x2, cuase it takes N from R1 + N from R2

for Nreads in reads_Ns:
    
    ## creating subsampled fastq
    file_readers.save_pfastq_Nreads(file1, file2, Nreads)

    ## decontaminating
    #file12 = glob.glob(path + "/" + samp + "_S" + "*" + "_L001_R1_001_atroped_1_" + str(Nreads) + ".fastq")
    #file22 = glob.glob(path + "/" + samp + "_S" + "*" + "_L001_R2_001_atroped_2_" + str(Nreads) + ".fastq")
    #file12 = file12[0]
    #file22 = file22[0]
    file11 = file1.replace(".fastq", "_" + str(Nreads) + ".fastq")
    file21 = file2.replace(".fastq", "_" + str(Nreads) + ".fastq")
    
    process_data.dehostaminate_pe(file11, file21, ref_file, bwa = True, ont_reads = False, threads=32)

    #file123 = glob.glob(path + "/" + samp + "_S" + "*" + "_bwa_unhosted.fastq")
    file123 = file11.replace(".fastq", "_bwa_unhosted.fastq")
    ## assembling
    process_data.assemble_decon_se(path, db_path, file123, samp_id=samp, threads=32, min_con_len = 350) # 350 bp > read length
    
    ## assigning
    fa_path = path + "/metagenome_contigs_" + samp + ".fastq"
    bdb_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/blastn/bac_refs_refseq/bac_refs_refseq"
    #bdb_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/nt_blast/nt_blast"
    out_path = path + "/" + samp + "_" + str(Nreads) + "_blastn_report.txt"
    process_data.run_blastn(fa_path, bdb_path, out_path, nthreads=32, evalue=1e-6, max_ts=5, max_h=1)
    
    os.remove(file11)
    os.remove(file21)
    os.remove(file123)
    os.remove(fa_path)