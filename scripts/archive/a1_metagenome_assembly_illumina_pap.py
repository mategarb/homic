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

######################################## illumina PAP ###############################################
main_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/"

if False:
    db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/gutmetanet"
    ref_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/GutMetaNet.fasta"
    kraken2.prepare_db(db_path, ref_path)

#####################################################################################################
### tso filter, no needed as they are not fully sequenced
#if False:
#    file1 = main_path + "20250820-Lib15_S2_L001_R2_001.fastq"
#    file2 = main_path + "20250820-Lib9_S1_L001_R2_001.fastq"
    
#    process_data.filter_tso(file1, tso_seq="AAGCAGTGGTATCAACGCAGAGTACATGGG")
#    process_data.filter_tso(file2, tso_seq="AAGCAGTGGTATCAACGCAGAGTACATGGG")
    

#####################################################################################################
### cutadapt, polyA/T, adapters
if False:
    fasta_path = main_path + "dna_fragments.fa"
    fasta_rc_path = main_path + "dna_fragments_rc.fa"
    file1 = main_path + "20250820-Lib15_S2_L001_R2_001.fastq"
    file2 = main_path + "20250820-Lib9_S1_L001_R2_001.fastq"
    
    process_data.run_atropos_se(file1, fasta_path, fasta_rc_path, nhead=20, ntail=125, minlen=50, minqual=20, error_rate=0.1)
    process_data.run_atropos_se(file2, fasta_path, fasta_rc_path, nhead=20, ntail=125, minlen=50, minqual=20, error_rate=0.1)

#####################################################################################################
### kraken before host decontamination

if False:
    data_1 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/20250820-Lib9_S1_L001_R2_001.fastq'
    data_2 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/20250820-Lib15_S2_L001_R2_001.fastq'
    
    k2_db = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/standard_db_k2'

    conf = 0.1
    mhg = 3
    out_1 = kraken2.classify(db_path = k2_db, input_file = data_1, confidence=conf, min_hit_gr=mhg, threads=16)

    out_2 = kraken2.classify(db_path = k2_db, input_file = data_2, confidence=conf, min_hit_gr=mhg, threads=16)

    
    k2file1 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/Lib9_krk_nod_nof.csv'
    k2file2 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/Lib15_krk_nod_nof.csv'

    out_1.to_csv(k2file1, index=False)
    out_2.to_csv(k2file2, index=False)


#####################################################################################################
### decontaminate SE ###
if False:
    ref_file = "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/GRCh38.fna"
    #ref_file = "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/human_index"
    
    file1 = main_path + "20250820-Lib15_S2_L001_R2_001_atroped.fastq"
    process_data.dehostaminate_se(file1, ref_file, method = "bwa", threads=32)

    file2 = main_path + "20250820-Lib9_S1_L001_R2_001_atroped.fastq"
    process_data.dehostaminate_se(file2, ref_file, method = "bwa", threads=32)

#####################################################################################################
### decontaminate SE - mito + rRNA ### didnt change anything, so I am skipping this step
if False:
    ref_file = "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/Homo_sapiens_mito+rRNA.fasta"
    
    file1 = main_path + "20250820-Lib15_S2_L001_R2_001_atroped_bwa_unhosted.fastq"
    process_data.dehostaminate_se(file1, ref_file, method = "bwa", threads=32)

    file2 = main_path + "20250820-Lib9_S1_L001_R2_001_atroped_bwa_unhosted.fastq"
    process_data.dehostaminate_se(file2, ref_file, method = "bwa", threads=32)

#####################################################################################################
### subsample control
if False:

    file2 = main_path + "20250820-Lib15_S2_L001_R2_001_atroped_bwa_unhosted.fastq" # 
    Nreads = 115859
    #process_data.subsample_fastq(file2, Nreads, seed=123)   
    process_data.subsample_fastq(file2, Nreads, seed=234)   

#####################################################################################################
### taxa assignment with kraken ###

if False:
    data_1 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/20250820-Lib9_S1_L001_R2_001_atroped_bwa_unhosted.fastq'
    data_2 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/20250820-Lib15_S2_L001_R2_001_atroped_bwa_unhosted.fastq'
    data_3 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/20250820-Lib15_S2_L001_R2_001_atroped_bwa_unhosted_115859_subsampled_rep1.fastq'
    data_4 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/20250820-Lib15_S2_L001_R2_001_atroped_bwa_unhosted_115859_subsampled_rep2.fastq'

    
    k2_db = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/f156sp_wgs'

    conf = 0.1
    mhg = 3
    out_1 = kraken2.classify(db_path = k2_db,
                input_file = data_1, confidence=conf, min_hit_gr=mhg, threads=16)

    out_2 = kraken2.classify(db_path = k2_db,
                input_file = data_2, confidence=conf, min_hit_gr=mhg, threads=16)
    
    out_3 = kraken2.classify(db_path = k2_db,
                input_file = data_3, confidence=conf, min_hit_gr=mhg, threads=16)
    
    out_4= kraken2.classify(db_path = k2_db,
                input_file = data_4, confidence=conf, min_hit_gr=mhg, threads=16)
    
    k2file1 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/Lib9_krk.csv'
    k2file2 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/Lib15_krk.csv'
    k2file3 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/Lib15sub_rep1_krk.csv'
    k2file4 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/Lib15sub_rep2_krk.csv'
    out_1.to_csv(k2file1, index=False)
    out_2.to_csv(k2file2, index=False)
    out_3.to_csv(k2file3, index=False)
    out_4.to_csv(k2file4, index=False)


#####################################################################################################
#### assembling metagenome ### not used anymore, just kraken2 (Nov 17th)
#if False:
#    path = main_path
#    path2 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes/"
#    dbpath = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/human_hg38"
#    
#    file1 = main_path + "20250820-Lib9_S1_L001_R2_001_RC_atroped_aligned_unhosted.fastq"
#    process_data.assemble_decon_se(path, dbpath, file1, samp_id="control", threads=16, min_con_len = 500)
#    
#    file2 = main_path + "20250820-Lib15_S2_L001_R2_001_RC_atroped_aligned_unhosted.fastq"
#    file3 = main_path + "20250820-Lib15_S2_L001_R2_001_RC_atroped_aligned_unhosted_303023_subsampled.fastq"
#    process_data.assemble_decon_se(path, dbpath, file2, samp_id="polyA", threads=16, min_con_len = 500)
#    process_data.assemble_decon_se(path, dbpath, file3, samp_id="polyA_balanced", threads=16, min_con_len = 500)

#####################################################################################################
### taxa assignment with blastn ### not used anymore, just kraken2 (Nov 17th)
if True:
    print("Assigning taxa...")
    # Add BLAST+ bin directory to PATH
    os.environ["PATH"] = "/gpfs/commons/home/mgarbulowski/ncbi-blast-2.16.0+/bin:" + os.environ["PATH"]
    main_path="/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/illumina_data/"
    # inputs
    fa_path1 = main_path + "20250820-Lib9_S1_L001_R2_001_atroped_bwa_unhosted.fasta"
    fa_path2 = main_path + "20250820-Lib15_S2_L001_R2_001_atroped_bwa_unhosted.fasta"
    
    # dbs
    #db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/blastn/nt_blast/nt_blast"
    #db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/blastn/GbBacU/all_seqs.fna"
    db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/blastn/bac_refs_refseq/bac_refs_refseq"
    #db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/blastn/bac_refs_genban/bac_refs_genban"

    # run blastn
    out_path1 = main_path + "blastn_rs_reads/control_blastn_report.txt"
    process_data.run_blastn(fa_path1, db_path, out_path1, nthreads=16, evalue=1e-6, max_ts=5, max_h=1)
    out_path2 = main_path + "blastn_rs_reads/polyA_blastn_report.txt"
    process_data.run_blastn(fa_path2, db_path, out_path2, nthreads=16, evalue=1e-6, max_ts=5, max_h=1)

    print("Done!")







