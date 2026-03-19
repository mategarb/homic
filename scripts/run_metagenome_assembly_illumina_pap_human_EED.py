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
# tests with smaller dbs for human EED
############################

######################################## illumina PAP ###############################################
#main_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/time_course"
#main_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/polyA_v3/"
#main_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ntekas_et_al_papNoPap/"
main_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/EED_PNK/"

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
# step. 1
if False:
    #ada_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/"
    fasta_path = main_path + "dna_fragments.fa"
    file1 = main_path + "SRR30622113_nopap/SRR30622113_2.fastq"
    process_data.run_fastp_se(file1, fasta_path, minPolyLen=10, minlen=50, ntail=20, minqual=25)
    file2 = main_path + "SRR30622109_pap/SRR30622109_2.fastq"
    process_data.run_fastp_se(file2, fasta_path, minPolyLen=10, minlen=50, ntail=20, minqual=25)


if True:
    fasta_path = main_path + "dna_fragments2.fa"
    files = glob.glob(main_path + "/EED_gut*.fastq.gz")

    for file in files:
        process_data.run_fastp_se(file, fasta_path)


if False:
    #ada_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/"
    fasta_path = main_path + "dna_fragments.fa"
    files = glob.glob(main_path + "/Mouse*.fastq.gz")

    for file in files:
        process_data.run_fastp_se(file, fasta_path)

if False:
    #ada_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/data_polyA_comparison/"
    fasta_path = main_path + "dna_fragments.fa"
    fasta_rc_path = main_path + "dna_fragments_rc.fa"
    files = glob.glob(main_path + "/human/Lib-*fastq.gz")

    for file in files:
        process_data.run_atropos_se(file, fasta_path, fasta_rc_path, minlen=50, minqual=20, error_rate=0.08)
        #process_data.run_atropos_se(file, fasta_path, fasta_rc_path, nhead=10, ntail=20, minlen=40, minqual=20, error_rate=0.1)

# step. 2 decontamination
# decontaminate human
if True:
    ref_file = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/bwa/GRCh38.primary_assembly.genome.fa"
    files = glob.glob(main_path + "/EED_gut*_fastped.fastq")
    for file in files:
        process_data.dehostaminate_se(file, ref_file, method = "bwa", threads=32)


if False:
    ref_file = "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/GRCh38.fna"
    files = glob.glob(main_path + "human/Lib-*_atroped.fastq")
    for file in files:
        process_data.dehostaminate_se(file, ref_file, method = "bwa", threads=32)

# decontaminate mouse
if False:
    ref_file = "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/GCF_000001635.27_GRCm39_genomic.fna"
    #files = glob.glob(main_path + "mouse/Lib-*_atroped.fastq")
    files = glob.glob(main_path + "Mouse*_fastped.fastq")
    for file in files:
        process_data.dehostaminate_se(file, ref_file, method = "bwa", threads=32)


if False:
    ref_file = "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/GCF_000001635.27_GRCm39_genomic.fna"
    file1 = main_path + "SRR30622113_nopap/SRR30622113_2_fastped.fastq"
    process_data.dehostaminate_se(file1, ref_file, method = "bwa", threads=32)
    file2 = main_path + "SRR30622109_pap/SRR30622109_2_fastped.fastq"
    process_data.dehostaminate_se(file2, ref_file, method = "bwa", threads=32)



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

if False:
    files = glob.glob(main_path + "/human/Lib-*_atroped.fastq")

    for file in files:
        
        # atropos
        after_atropos = count_reads(file)
        # bwa
        file_bwa = file.replace(".fastq", "_bwa_unhosted.fastq")
        after_bwa = count_reads(file_bwa)
        
        out_file = main_path + "/data_stats_human.txt"
                
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


# ntekas et al. gold standard
if False:

    #files = glob.glob(main_path + "/mouse/Lib-*_atroped_bwa_unhosted.fastq")
    files = glob.glob(main_path + "Mouse*_fastped_bwa_unhosted.fastq")
    
    conf = 0.1
    mhg = 3
    #k2_db = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/f156sp_wgs'
    k2_db = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/standard_db_k2'
    
    file1 = main_path + "SRR30622113_nopap/SRR30622113_2_fastped_bwa_unhosted.fastq"
    file2 = main_path + "SRR30622109_pap/SRR30622109_2_fastped_bwa_unhosted.fastq"
    
    out_k2 = kraken2.classify(db_path = k2_db,
            input_file = file1, confidence=conf, min_hit_gr=mhg, threads=32)
    out_k2_path = file1.replace("_fastped_bwa_unhosted.fastq", "_k2.csv")
    out_k2.to_csv(out_k2_path, index=False)

    
    out_k2 = kraken2.classify(db_path = k2_db,
            input_file = file2, confidence=conf, min_hit_gr=mhg, threads=32)
    out_k2_path = file2.replace("_fastped_bwa_unhosted.fastq", "_k2.csv")
    out_k2.to_csv(out_k2_path, index=False)
    
    k2_db = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/kraken2_db_uhgg_v2.0.2"
    out_k2 = kraken2.classify(db_path = k2_db,
            input_file = file1, confidence=conf, min_hit_gr=mhg, threads=32)
    out_k2_path = file1.replace("_fastped_bwa_unhosted.fastq", "_uhgg.csv")
    out_k2.to_csv(out_k2_path, index=False)

    
    out_k2 = kraken2.classify(db_path = k2_db,
            input_file = file2, confidence=conf, min_hit_gr=mhg, threads=32)
    out_k2_path = file2.replace("_fastped_bwa_unhosted.fastq", "_uhgg.csv")
    out_k2.to_csv(out_k2_path, index=False)

### EED
if True:

    #files = glob.glob(main_path + "/EED_mis_gut*_fastped_bwa_unhosted.fastq")
    files = glob.glob(main_path + "/EED_gut*_fastped_bwa_unhosted.fastq")
    
    conf = 0.1
    mhg = 3
    #k2_db = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/f156sp_wgs'
    k2_db = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/standard_db_k2'
    #k2_db = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/kraken2_db_uhgg_v2.0.2"
    
    for file in files:
        out_k2 = kraken2.classify(db_path = k2_db,
                    input_file = file, confidence=conf, min_hit_gr=mhg, threads=32)
        out_k2_path = file.replace("_fastped_bwa_unhosted.fastq", "_k2.csv")
        out_k2.to_csv(out_k2_path, index=False)
        

### mouse miseq i100  / nextseq
if False:

    #files = glob.glob(main_path + "/mouse/Lib-*_atroped_bwa_unhosted.fastq")
    files = glob.glob(main_path + "Mouse*_fastped_bwa_unhosted.fastq")
    
    conf = 0.1
    mhg = 3
    #k2_db = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/f156sp_wgs'
    k2_db = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/standard_db_k2'
    #k2_db = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/kraken2_db_uhgg_v2.0.2"
    
    for file in files:
        out_k2 = kraken2.classify(db_path = k2_db,
                    input_file = file, confidence=conf, min_hit_gr=mhg, threads=32)
        out_k2_path = file.replace("_fastped_bwa_unhosted.fastq", "_k2.csv")
        out_k2.to_csv(out_k2_path, index=False)
        
### human miseq i100  / nextseq
if False:

    files = glob.glob(main_path + "/human/Lib-*_atroped_bwa_unhosted.fastq")
    conf = 0.1
    mhg = 3
    #k2_db = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/f156sp_wgs'
    k2_db = '/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/standard_db_k2'
    
    for file in files:
        out_k2 = kraken2.classify(db_path = k2_db,
                    input_file = file, confidence=conf, min_hit_gr=mhg, threads=32)
        out_k2_path = file.replace("_atroped_bwa_unhosted.fastq", "_k2.csv")
        out_k2.to_csv(out_k2_path, index=False)


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
if False:
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







