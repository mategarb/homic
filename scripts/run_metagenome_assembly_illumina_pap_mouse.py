import numpy as np
import pandas as pd
import statistics
import seaborn as sns
import os
import subprocess
import shutil
import sys
import glob
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src') 
from homic import file_readers, kraken2, process_data

############################
#### script description ####
# PAP vs NoPAP analysis
# tests with smaller dbs for mouse
############################

######################################## illumina PAP ###############################################
#main_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/time_course"
main_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/polyA_v2/"

#####################################################################################################
if False: ## add mouse genome to the standard k2 db
    #path_to_db = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/standard_db_k2"
    #path_to_ref = "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/GCF_000001635.27_GRCm39_genomic.fna"
    #kraken2.add_to_db(path_to_db, path_to_ref)

    db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/comGenOnly_45_500bp/"
    ref_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/mm_refs_for_dbs/full_genome_mm_65sp_500_comGenOnly.fasta"
    kraken2.prepare_db(db_path, ref_path)

    db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/comGenOnly_45_1000bp/"
    ref_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/mm_refs_for_dbs/full_genome_mm_65sp_1000_comGenOnly.fasta"
    kraken2.prepare_db(db_path, ref_path)

    db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/comGenOnly_45_2500bp/"
    ref_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/mm_refs_for_dbs/full_genome_mm_65sp_2500_comGenOnly.fasta"
    kraken2.prepare_db(db_path, ref_path)

    db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/comGenOnly_45_5000bp/"
    ref_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/mm_refs_for_dbs/full_genome_mm_65sp_5000_comGenOnly.fasta"
    kraken2.prepare_db(db_path, ref_path)

    db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/comGenOnly_45_full/"
    ref_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/mm_refs_for_dbs/full_genome_mm_65sp_0_comGenOnly.fasta"
    kraken2.prepare_db(db_path, ref_path)
    
### cutadapt, polyA/T, adapters
if False:
    #ada_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/"
    fasta_path = main_path + "dna_fragments.fa"
    fasta_rc_path = main_path + "dna_fragments_rc.fa"
    files = glob.glob(main_path + "/mouse/Lib-*fastq.gz")

    for file in files:
        process_data.run_atropos_se(file, fasta_path, fasta_rc_path, nhead=10, ntail=20, minlen=40, minqual=20, error_rate=0.1)

# decontaminate mouse
if False:
    ref_file = "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/GCF_000001635.27_GRCm39_genomic.fna"
    files = glob.glob(main_path + "mouse/Lib-*_atroped.fastq")
    for file in files:
        process_data.dehostaminate_se(file, ref_file, method = "bwa", threads=32)

def count_reads(fastq_file):
    """Return number of reads in a FASTQ or FASTQ.GZ file."""
    open_func = gzip.open if str(fastq_file).endswith(".gz") else open
    with open_func(fastq_file, "rt") as f:
        line_count = sum(1 for _ in f)
    return line_count // 4

# counting reads in files
if False:
    files = glob.glob(main_path + "/mouse/Lib-*_atroped.fastq")

    for file in files:
        
        # atropos
        after_atropos = count_reads(file)
        # bwa
        file_bwa = file.replace(".fastq", "_bwa_unhosted.fastq")
        after_bwa = count_reads(file_bwa)
        
        out_file = main_path + "/data_stats_mouse.txt"
                
        with open(out_file, "a") as f:
            print("File name: " + file, file=f)
            print(f"After atropos: {after_atropos:,}", file=f)
            print(f"After decontamination with bwa:  {after_bwa:,}", file=f)


#####################################################################################################
### subsample control
if False:

    file2 = main_path + "20250820-Lib15_S2_L001_R2_001_atroped_bwa_unhosted.fastq" # 
    Nreads = 115859
    #process_data.subsample_fastq(file2, Nreads, seed=123)   
    process_data.subsample_fastq(file2, Nreads, seed=234)   

#####################################################################################################
### taxa assignment with kraken ###

if True:

    files = glob.glob(main_path + "/mouse/Lib-*_atroped_bwa_unhosted.fastq")
    conf = 0.1
    mhg = 3

    #k2_db1 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/comGenOnly_45_500bp/'
    #k2_db2 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/comGenOnly_45_1000bp/'
    #k2_db3 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/comGenOnly_45_2500bp/'
    #k2_db4 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/comGenOnly_45_5000bp/'
    #k2_db5 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/comGenOnly_45_full/'
    k2_dbs = ["/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/kraken2_db_uhgg_v2.0.2"]
    #k2_dbs = [k2_db1,k2_db2,k2_db3,k2_db4,k2_db5]
    #k2_dbs_nms = ['500','1000','2500','5000','full']
    k2_dbs_nms = ['uhgg']
    for path, name in zip(k2_dbs, k2_dbs_nms):
        for file in files:
            out_k2 = kraken2.classify(db_path = path,
                        input_file = file, confidence=conf, min_hit_gr=mhg, threads=32)
            out_k2_path = file.replace("_atroped_bwa_unhosted.fastq", "_" + name + "_k2.csv")
            out_k2.to_csv(out_k2_path, index=False)


