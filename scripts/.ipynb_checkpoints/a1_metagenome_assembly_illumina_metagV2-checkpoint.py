import numpy as np
import pandas as pd
import statistics
import seaborn as sns
import os
import subprocess
import shutil
import glob
import sys
import gzip
from pathlib import Path
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src') 
from homic import file_readers, kraken2, process_data

############################
#### script description ####
# metagenome assembly
############################

######################################################################################################
### paths

db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/human_hg38"

# define paths
#path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/batch1+3"
#path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/batch2+4"
#path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/batches_1-5"
#path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/batch5"
path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/batch6"
#path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/batches_1-4"

#path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/replicates_analysis"

fasta_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/dna_fragments.fa"
fasta_rc_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/dna_fragments_rc.fa"

# samples id 
#samps = ["KP003","KP012"]
samps = ["KP003","KP004","KP005","KP008","KP010","KP011","KP012",
         "KP013","KP016","KP021","KP024","KP025","KP026","KP027","KP029",
    "KP033","KP035","KP037","KP038","KP040","KP041","KP046","KP047","KP048","KP049","KP052"]

#samps = ["B1_KP027","B3_KP027","B5_KP027","B6_KP027"]

#samps = ["KP025","KP026","KP027","KP029",
#    "KP033","KP035","KP037","KP038","KP040","KP041","KP046","KP047","KP048","KP049","KP052"]


#samps = ["KP005", "KP012", "KP013", "KP016", "KP024", "KP025", "KP026", "KP027", "KP029","KP033","KP035","KP038","KP040","KP046","KP047","KP048"] # batch 1 + 3


#samps = ["KP003","KP004","KP008","KP010","KP011","KP021","KP037","KP041","KP049","KP052"]# batch 2 + 4 

def count_reads(fastq_file):
    """Return number of reads in a FASTQ or FASTQ.GZ file."""
    open_func = gzip.open if str(fastq_file).endswith(".gz") else open
    with open_func(fastq_file, "rt") as f:
        line_count = sum(1 for _ in f)
    return line_count // 4

#####################################################################################################
### trimming + qc + decontamination ##
if False:
    #ref_file = "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/GRCh38.mmi"
    ref_file = "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/GRCh38.fna"
    for samp in samps:  
    
            print("Trimming and filtering...")
            file1 = glob.glob(path + "/" + samp + "_S" + "*" + "_L001_R1_001.fastq.gz")
            file2 = glob.glob(path + "/" + samp + "_S" + "*" + "_L001_R2_001.fastq.gz")
            file1 = file1[0]
            file2 = file2[0]
            process_data.run_atropos_pe(file1, file2, fasta_path, fasta_rc_path, nhead=20, ntail=10, minlen=50, minqual=20, error_rate=0.1, threads=32)
            
        #####################################################################################################
        ### decontaminate SE ###
    
            print("Decontaminating...")
            file11 = glob.glob(path + "/" + samp + "_S" + "*" + "_L001_R1_001_atroped_1.fastq")
            file21 = glob.glob(path + "/" + samp + "_S" + "*" + "_L001_R2_001_atroped_2.fastq")
            file11 = file11[0]
            file21 = file21[0]
            process_data.dehostaminate_pe(file11, file21, ref_file, method = "bwa", threads=32)

            before1 = count_reads(file1)
            after1 = count_reads(file11)
    
            before2 = count_reads(file2)
            after2 = count_reads(file21)
    
            filtered1 = before1 - after1
            filtered2 = before2 - after2
    
            out_file = path + "/data_stats.txt"
            
            with open(out_file, "a") as f:
                print("Sample " + samp, file=f)
                print("R1 atropos", file=f)
                print(f"Reads raw: {before1:,}", file=f)
                print(f"Reads after atropos:  {after1:,}", file=f)
                print(f"Filtered out: {filtered1:,} ({filtered1 / before1 * 100:.2f}%)", file=f)
            
                print("R2 atropos", file=f)
                print(f"Reads raw: {before2:,}", file=f)
                print(f"Reads after atropos:  {after2:,}", file=f)
                print(f"Filtered out: {filtered2:,} ({filtered2 / before2 * 100:.2f}%)", file=f)
            
                file3 = file11.replace(".fastq", "_bwa_unhosted.fastq")
            
                after3 = count_reads(file3)
                print("Decontamination", file=f)
                print(f"Reads after atropos R1 + R2: {after1 + after2:,}", file=f)
                print(f"Reads after host decontamination:  {after3:,}", file=f)
                print("", file=f)  # blank line between samples
            
            os.remove(file11)
            os.remove(file21)
    
    
#####################################################################################################
### running QC after trimming ###
if False:
    print("QC on trimmed samples...")
    path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/data_all_raw_x3"
    output_dir = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/fastqc_all_x3"

    for samp in samps:
        file = glob.glob(path + "/" + samp + "*" + "_unhosted.fastq")
        cmd = ["fastqc",
               "--outdir",
               output_dir,
               file[0]]
                
        subprocess.call(cmd)

#####################################################################################################
### assembling metagenome ###
if False:
    path2 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/data_all_raw_x4"
    print("Assembling " + str(len(samps)) + " samples...")
    for idx, samp in enumerate(samps):  
        file = glob.glob(path2 + "/" + samp + "_S" + "*" + "_bwa_unhosted.fastq")
        file = file[0]
        process_data.assemble_decon_se(path2, db_path, file, samp_id=samp, threads=32, min_con_len = 1000)

#####################################################################################################
################################# Running binning
#if False:
#    assembly_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes/metagenome_contigs_KP005.fastq"
#    file_1_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes/016-Meta-1/KP005_S1_L001_k2_1.fastq"
#    file_2_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes/016-Meta-1/KP005_S1_L001_k2_2.fastq"
#    out_dir = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes/016-Meta-1/metabat_bins"
#    
#    process_data.bow_bat(assembly_path, file_1_path, file_2_path, out_dir, False)


#####################################################################################################
### taxa assignment with blastn ### 
if True:
    
    print("Assigning taxa...")
    
    # Add BLAST+ bin directory to PATH
    os.environ["PATH"] = "/gpfs/commons/home/mgarbulowski/ncbi-blast-2.16.0+/bin:" + os.environ["PATH"]
    path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/contigs_all_x4"
    
    #bdb_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/blastn/GbBacU/all_seqs.fna"
    #bdb_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/blastn/bac_refs_refseq/bac_refs_refseq"
    bdb_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/blastn/nt_blast/nt_blast"
    #db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/bac_refs_genban/bac_refs_genban"
    
    for idx, samp in enumerate(samps):  
        fa_path = path + "/metagenome_contigs_"  + samp + ".fastq"

        #out_path = path + "/blastn_refseq/" + samp + "_blastn_report.txt"
        out_path = path + "/blastn_nt/" + samp + "_blastn_report.txt"
        process_data.run_blastn(fa_path, bdb_path, out_path, nthreads=32, evalue=1e-6, max_ts=5, max_h=1)

    print("Done!")






