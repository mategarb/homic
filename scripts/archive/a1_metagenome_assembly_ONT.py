import numpy as np
import pandas as pd
import statistics
import seaborn as sns
import os
import subprocess
import pysam
import ctypes
import shutil
import sys
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src')
from homic import file_readers, kraken2, process_data

############################
#### script description ####
# ONT data analysis
############################

main_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/ont_data/"

########################################### ONT  ####################################################
### tso filter
if False:
    file1 = main_path + "20250925_1858_P2S-02509-B_PBA50717_5e6e2eeb.fastq"
    file2 = main_path + "20251015_1800_P2S-02509-B_PBA23028_e80b6c85.fastq"
    
    process_data.filter_tso(file1, tso_seq="AAGCAGTGGTATCAACGCAGAGTACATGGG")
    process_data.filter_tso(file2, tso_seq="AAGCAGTGGTATCAACGCAGAGTACATGGG")

### from Davids pipeline (cutadapt)

#ADAPTER_COUNT=4,  # number of adapters that can be trimmed from each read
#QUALITY_MIN=5,  # Low Q score for ONT...
#MIN_R2_LENGTH=12,
#OVERLAP=5,
#HOMOPOLYMER_ERROR_RATE=0.2,  # default error rate is 0.1
#HETEROPOLYMER_ERROR_RATE=0.2,  # default error rate is 0.1

#POLYA="A" * 100,
#POLYT="T" * 100,
#TXG_TSO="AAGCAGTGGTATCAACGCAGAGTACATGGG",  # 10x TSO - remove any polyadenylated TSOs
#rcTXG_TSO="CCCATGTACTCTGCGTTGATACCACTGCTT",  # rev-comp of 10x TSO sequence
#uMRT_TSO="CCCTCTCTCTCTCTTTCCTCTCTC",  #ISOPCR sequence from uMRT protocol; does not contain the 4T overhang
#SEEKER_BB_LINKER="TCTTCAGCGTTCCCGAGA",  # Adapter between BB1 & BB2 in R1 
#rcSEEKER_BB_ADAPTER="AGAGCCCTTGCGACTTCT",  # Reverse of the adapter between BB1 & BB2 in R1
#VNP="ACTTGCCTGTCGCTCTATCTTCTTTTT",
#SSP="TTTCTGTTGGTGCTGATATTGCT",

#####################################################################################################
### atropos, polyA/T
if False:
    fasta_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/ont_data/seqs_dsm.fa"
    fasta_rc_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/ont_data/seqs_dsm_rc.fa"
    file1 = main_path + "20250925_1858_P2S-02509-B_PBA50717_5e6e2eeb_tsoFiltered.fastq"
    file2 = main_path + "20251015_1800_P2S-02509-B_PBA23028_e80b6c85_tsoFiltered.fastq"
    
    # check 1
    #process_data.run_atropos_se(file1, fasta_path, fasta_rc_path, nhead=20, ntail=10, minlen=1000, minqual=10, error_rate=0.15, threads=64)
    #process_data.run_atropos_se(file2, fasta_path, fasta_rc_path, nhead=20, ntail=10, minlen=1000, minqual=10, error_rate=0.15, threads=64)
    # check 2
    process_data.run_atropos_se(file1, fasta_path, fasta_rc_path, nhead=20, ntail=20, minlen=300, maxlen=1100, minqual=5, error_rate=0.2, threads=32)
    process_data.run_atropos_se(file2, fasta_path, fasta_rc_path, nhead=20, ntail=20, minlen=300, maxlen=1100, minqual=5, error_rate=0.2, threads=32)


#####################################################################################################
### chopper, qc + len
#if True:
#    db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/human_hg38"
#    file1 = main_path + "20250925_1858_P2S-02509-B_PBA50717_5e6e2eeb_tsoFiltered_atroped.fastq"
#    file2 = main_path + "20251015_1800_P2S-02509-B_PBA23028_e80b6c85_tsoFiltered_atroped.fastq"
#    process_data.chop_decon_se(db_path, file1, min_quality=10, min_length=300, head_crop=10, tail_crop=10)
#    process_data.chop_decon_se(db_path, file2, min_quality=10, min_length=300, head_crop=10, tail_crop=10)


#####################################################################################################
### kraken db for human ### done once
#if False:
#    db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/human_hg38"
#    ref_path =  "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/GCF_000001405.26_GRCh38_genomic.fna"  #from: https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.26_GRCh38/
    
#    kraken2.prepare_db(db_path, ref_path)

#####################################################################################################
### decontaminate SE ###
if True:
    ref_file = "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/GRCh38.fna"
    
    file1 = main_path + "20250925_1858_P2S-02509-B_PBA50717_5e6e2eeb_tsoFiltered_atroped.fastq"
    process_data.dehostaminate_se(file1, ref_file, method = "mm2-splice", threads=32)

    file2 = main_path + "20251015_1800_P2S-02509-B_PBA23028_e80b6c85_tsoFiltered_atroped.fastq"
    process_data.dehostaminate_se(file2, ref_file, method = "mm2-splice", threads=32)


#####################################################################################################
### subsample control
if False:
    file1 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/20250925_1858_P2S-02509-B_PBA50717_5e6e2eeb_trimmed_ca_chopped_aln_noHost.fastq"
    file2 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/20251015_1800_P2S-02509-B_PBA23028_e80b6c85_trimmed_ca_chopped_aln_noHost.fastq"
    Nreads = 3016385
    rocess_data.subsample_fastq(file2, Nreads, seed=123)   
    process_data.subsample_fastq(file2, Nreads, seed=234)    

#####################################################################################################
### taxa assignment with kraken ###

if False:
    data_1 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/ont_data/minlen300-2000/20250925_1858_P2S-02509-B_PBA50717_5e6e2eeb_tsoFiltered_atroped_mm2_unhosted.fastq'
    data_2 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/ont_data/minlen300-2000/20251015_1800_P2S-02509-B_PBA23028_e80b6c85_tsoFiltered_atroped_mm2_unhosted.fastq'
    k2_db = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/f156sp_wgs'

    out_1 = kraken2.classify(db_path = k2_db,
                input_file = data_1)

    out_2 = kraken2.classify(db_path = k2_db,
                input_file = data_2)

    k2file1 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/ont_data/Lib15_krk.csv'
    k2file2 = '/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/ont_data/Lib9_krk.csv'
    out_1.to_csv(k2file1, index=False)
    out_2.to_csv(k2file2, index=False)


#####################################################################################################
### assembling metagenome ###
if False:
    db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/human_hg38"
    path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison"
    file1 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/20250925_1858_P2S-02509-B_PBA50717_5e6e2eeb_trimmed_ca_chopped_aln_noHost.fastq"
    file2 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/20251015_1800_P2S-02509-B_PBA23028_e80b6c85_trimmed_ca_chopped_aln_noHost_3016385_randReads_rep1.fastq" # control 1
    file3 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/20251015_1800_P2S-02509-B_PBA23028_e80b6c85_trimmed_ca_chopped_aln_noHost_3016385_randReads_rep2.fastq" # control 2
    
    process_data.assemble_decon_ont(path, db_path, file1, samp_id="polyA", threads=20) # SE
    process_data.assemble_decon_ont(path, db_path, file2, samp_id="control_rep1", threads=20, min_con_len = 500) # SE
    process_data.assemble_decon_ont(path, db_path, file3, samp_id="control_rep2", threads=20, min_con_len = 500) # SE

#####################################################################################################
### before blastn, load lib once per session ### 
#if False:
#    cmd = "export PATH=$PATH:$HOME/ncbi-blast-2.16.0+/bin"
#    subprocess.call(cmd, shell=True, executable="/bin/bash")
    # other solution (otherwise do it manually in bash)
    #blast_path = os.path.expanduser("~/ncbi-blast-2.16.0+/bin")
    #os.environ["PATH"] += os.pathsep + blast_path

#####################################################################################################
### taxa assignment with blastn ### 
if False:
    print("Assigning taxa...")
    # Add BLAST+ bin directory to PATH
    os.environ["PATH"] = "/gpfs/commons/home/mgarbulowski/ncbi-blast-2.16.0+/bin:" + os.environ["PATH"]

    # inputs
    fa_path1 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/metagenome_contigs_control_rep1.fastq"
    fa_path2 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/metagenome_contigs_control_rep2.fastq"
    fa_path3 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/metagenome_contigs_polyA.fastq"

    # dbs
    #db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/blastn/nt_blast/nt_blast"
    db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/blastn/GbBacU/all_seqs.fna"
    #db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/blastn/bac_refs_refseq/bac_refs_refseq"
    #db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/blastn/bac_refs_genban/bac_refs_genban"

    # run blastn
    out_path1 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/blastn_gbbacu3/control_rep1_blastn_report.txt"
    process_data.run_blastn(fa_path1, db_path, out_path1, nthreads=32, evalue=1e-6, max_ts=5, max_h=1)

    out_path2 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/blastn_gbbacu3/control_rep2_blastn_report.txt"
    process_data.run_blastn(fa_path2, db_path, out_path2, nthreads=32, evalue=1e-6, max_ts=5, max_h=1)

    out_path3 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/blastn_gbbacu3/polyA_blastn_report.txt"
    process_data.run_blastn(fa_path3, db_path, out_path3, nthreads=32, evalue=1e-6, max_ts=5, max_h=1)

    print("Done!")






