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
from itertools import zip_longest

############################
#### script description ####
# saturation analysis - all batches
############################

############################################
def merge_fastqs(files, output_file):
    with open(output_file, 'w') as out_f:
        for fname in files:
            if not fname or not os.path.exists(fname):
                continue
            if os.path.getsize(fname) == 0:
                continue

            with open(fname, 'r') as in_f:
                out_f.writelines(in_f)

############################################

#### Add BLAST+ bin directory to PATH
os.environ["PATH"] = "/gpfs/commons/home/mgarbulowski/ncbi-blast-2.16.0+/bin:" + os.environ["PATH"]

## output
main_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/saturation_analysis_x7"
#main_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/saturation_analysis_x6_v2"
#main_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/saturation_analysis_x6_v3"

#### define paths
db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/kraken/human_hg38"
#ref_file = "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/GRCh38.fna"
ref_file = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/bwa/GRCh38.primary_assembly.genome.fa"
fasta_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/dna_fragments.fa"
fasta_rc_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/dna_fragments_rc.fa"

#### samples id 
## selecting what samples to process
## because samples in particular batches may differ

#all_samps = #["KP027","KP029","KP033","KP035","KP037","KP038","KP040","KP041","KP046","KP047","KP048","KP049","KP052"] # "KP003","KP004","KP005","KP008","KP010","KP011","KP012",
        # "KP013","KP016","KP021","KP024","KP025","KP026",

## samples selected for more seq after saturation analysis
# all_samps = ["KP004"] # KP004,"KP025", "KP046", "KP049", "KP037", "KP038", "KP021", "KP012"

## remaining after last batch
all_samps = ["KP005","KP011","KP013","KP026","KP029","KP033","KP048","KP003","KP010","KP012","KP052","KP021"]

## iterate over all samples
## steps need to be performed separately (we cannot merge files at the beginning) cause we detected some batch effects
for samp in all_samps:
    #samp = all_samps[0]
    print(samp)
    out_main_path = main_path + "/" + samp
    os.makedirs(out_main_path, exist_ok=True)
    
    ## all paths
    ## batch 1-4
    path1 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/batches_1-4"
    reads_f1_b14 = glob.glob(path1 + "/" + samp + "_S" + "*" + "_L001_R1_001.fastq.gz")
    reads_f2_b14 = glob.glob(path1 + "/" + samp + "_S" + "*" + "_L001_R2_001.fastq.gz")
    reads_f1_b14 = reads_f1_b14[0]
    reads_f2_b14 = reads_f2_b14[0]
    ## batch 5
    path2 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/batch5"
    reads_f1_b5 = glob.glob(path2 + "/" + samp + "_S" + "*" + "_L001_R1_001.fastq.gz")
    reads_f2_b5 = glob.glob(path2 + "/" + samp + "_S" + "*" + "_L001_R2_001.fastq.gz")
    reads_f1_b5 = reads_f1_b5[0]
    reads_f2_b5 = reads_f2_b5[0]
    ## batch 6
    path3 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/batch6"
    reads_f1_b6 = glob.glob(path3 + "/" + samp + "_S" + "*" + "_L001_R1_001.fastq.gz")
    reads_f2_b6 = glob.glob(path3 + "/" + samp + "_S" + "*" + "_L001_R2_001.fastq.gz")
    reads_f1_b6 = reads_f1_b6[0]
    reads_f2_b6 = reads_f2_b6[0]
    ## batch 7
    path4 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/batch7"
    reads_f1_b7 = glob.glob(path4 + "/" + samp + "_S" + "*" + "_L001_R1_001.fastq.gz")
    reads_f2_b7 = glob.glob(path4 + "/" + samp + "_S" + "*" + "_L001_R2_001.fastq.gz")

    if reads_f1_b7 and reads_f2_b7:
        reads_f1_b7 = reads_f1_b7[0]
        reads_f2_b7 = reads_f2_b7[0]
    else:
        reads_f1_b7 = None
        reads_f2_b7 = None
    ## batch 8
    path5 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/batch8"
    reads_f1_b8 = glob.glob(path5 + "/" + samp + "_S" + "*" + "_L001_R1_001.fastq.gz")
    reads_f2_b8 = glob.glob(path5 + "/" + samp + "_S" + "*" + "_L001_R2_001.fastq.gz")

    if reads_f1_b8 and reads_f2_b8:
        reads_f1_b8 = reads_f1_b8[0]
        reads_f2_b8 = reads_f2_b8[0]
    else:
        reads_f1_b8 = None
        reads_f2_b8 = None
    ## batch 9
    path6 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/batch9"
    reads_f1_b9 = glob.glob(path6 + "/" + samp + "_S" + "*" + "_L001_R1_001.fastq.gz")
    reads_f2_b9 = glob.glob(path6 + "/" + samp + "_S" + "*" + "_L001_R2_001.fastq.gz")

    if reads_f1_b9 and reads_f2_b9:
        reads_f1_b9 = reads_f1_b9[0]
        reads_f2_b9 = reads_f2_b9[0]
    else:
        reads_f1_b9 = None
        reads_f2_b9 = None
    
    ### atropos cleaning
    if True:
        #b1-4
        process_data.run_atropos_pe(reads_f1_b14, reads_f2_b14, fasta_path, fasta_rc_path, nhead=20, ntail=10, minlen=50, minqual=20, error_rate=0.1, threads=16)
        #b5
        process_data.run_atropos_pe(reads_f1_b5, reads_f2_b5, fasta_path, fasta_rc_path, nhead=20, ntail=10, minlen=50, minqual=20, error_rate=0.1, threads=16)
        #b6
        process_data.run_atropos_pe(reads_f1_b6, reads_f2_b6, fasta_path, fasta_rc_path, nhead=20, ntail=10, minlen=50, minqual=20, error_rate=0.1, threads=16)
        #b7
        if reads_f1_b7 is not None:
            process_data.run_atropos_pe(reads_f1_b7, reads_f2_b7, fasta_path, fasta_rc_path, nhead=20, ntail=10, minlen=50, minqual=20, error_rate=0.1, threads=16)
        #b8
        if reads_f1_b8 is not None:
            process_data.run_atropos_pe(reads_f1_b8, reads_f2_b8, fasta_path, fasta_rc_path, nhead=20, ntail=10, minlen=50, minqual=20, error_rate=0.1, threads=16)
        #b9
        if reads_f1_b9 is not None:
            process_data.run_atropos_pe(reads_f1_b9, reads_f2_b9, fasta_path, fasta_rc_path, nhead=20, ntail=10, minlen=50, minqual=20, error_rate=0.1, threads=16)
    
    
    #b1-4
    reads_f1_b14_a = reads_f1_b14.replace(".fastq.gz", "_atroped_1.fastq")
    reads_f2_b14_a = reads_f2_b14.replace(".fastq.gz", "_atroped_2.fastq")
    Ntot_f1_b14 = file_readers.fastq(reads_f1_b14_a)
    Ntot_f2_b14 = file_readers.fastq(reads_f2_b14_a)
    print(len(Ntot_f1_b14[0]))
    print(len(Ntot_f2_b14[0]))
    #b5
    reads_f1_b5_a = reads_f1_b5.replace(".fastq.gz", "_atroped_1.fastq")
    reads_f2_b5_a = reads_f2_b5.replace(".fastq.gz", "_atroped_2.fastq")
    Ntot_f1_b5 = file_readers.fastq(reads_f1_b5_a)
    Ntot_f2_b5 = file_readers.fastq(reads_f2_b5_a)
    print(len(Ntot_f1_b5[0]))
    print(len(Ntot_f2_b5[0]))
    #b6
    reads_f1_b6_a = reads_f1_b6.replace(".fastq.gz", "_atroped_1.fastq")
    reads_f2_b6_a = reads_f2_b6.replace(".fastq.gz", "_atroped_2.fastq")
    Ntot_f1_b6 = file_readers.fastq(reads_f1_b6_a)
    Ntot_f2_b6 = file_readers.fastq(reads_f2_b6_a)
    print(len(Ntot_f1_b6[0]))
    print(len(Ntot_f2_b6[0]))
    #b7
    if reads_f1_b7 is not None:
        reads_f1_b7_a = reads_f1_b7.replace(".fastq.gz", "_atroped_1.fastq")
        reads_f2_b7_a = reads_f2_b7.replace(".fastq.gz", "_atroped_2.fastq")
        Ntot_f1_b7 = file_readers.fastq(reads_f1_b7_a)
        Ntot_f2_b7 = file_readers.fastq(reads_f2_b7_a)
        print(len(Ntot_f1_b7[0]))
        print(len(Ntot_f2_b7[0]))
    else:
        reads_f1_b7_a = []
        reads_f2_b7_a = []
        
    #b8
    if reads_f1_b8 is not None:
        reads_f1_b8_a = reads_f1_b8.replace(".fastq.gz", "_atroped_1.fastq")
        reads_f2_b8_a = reads_f2_b8.replace(".fastq.gz", "_atroped_2.fastq")
        Ntot_f1_b8 = file_readers.fastq(reads_f1_b8_a)
        Ntot_f2_b8 = file_readers.fastq(reads_f2_b8_a)
        print(len(Ntot_f1_b8[0]))
        print(len(Ntot_f2_b8[0]))
    else:
        reads_f1_b8_a = []
        reads_f2_b8_a = []
        
    #b9
    if reads_f1_b9 is not None:
        reads_f1_b9_a = reads_f1_b9.replace(".fastq.gz", "_atroped_1.fastq")
        reads_f2_b9_a = reads_f2_b9.replace(".fastq.gz", "_atroped_2.fastq")
        Ntot_f1_b9 = file_readers.fastq(reads_f1_b9_a)
        Ntot_f2_b9 = file_readers.fastq(reads_f2_b9_a)
        print(len(Ntot_f1_b9[0]))
        print(len(Ntot_f2_b9[0]))
    else:
        reads_f1_b9_a = []
        reads_f2_b9_a = []
    
    reads_Ps = [x / 100 for x in [1,5,10,20,30,40,50,55,60,70,80,90,95,99,100]] # percentage, so balanced number is taken from each batch
    #reads_Ps = [1]
    
    reads_Ns_b14 = [round(x * len(Ntot_f1_b14[0])) for x in reads_Ps]
    print(reads_Ns_b14)
    reads_Ns_b5 = [round(x * len(Ntot_f1_b5[0])) for x in reads_Ps]
    print(reads_Ns_b5)
    reads_Ns_b6 = [round(x * len(Ntot_f1_b6[0])) for x in reads_Ps]
    print(reads_Ns_b6)
    if reads_f1_b7 is not None:
        reads_Ns_b7 = [round(x * len(Ntot_f1_b7[0])) for x in reads_Ps]
        print(reads_Ns_b7)
    else:
        reads_Ns_b7 = []
        
    if reads_f1_b8 is not None:
        reads_Ns_b8 = [round(x * len(Ntot_f1_b8[0])) for x in reads_Ps]
        print(reads_Ns_b8)
    else:
        reads_Ns_b8 = []
        
    if reads_f1_b9 is not None:
        reads_Ns_b9 = [round(x * len(Ntot_f1_b9[0])) for x in reads_Ps]
        print(reads_Ns_b9)    
    else:
        reads_Ns_b9 = []
    

    reads_Ns_all = [
        (x or 0) + (y or 0) + (z or 0) + (w or 0) + (v or 0) + (u or 0)
        for x, y, z, w, v, u in zip_longest(
            reads_Ns_b14,
            reads_Ns_b5,
            reads_Ns_b6,
            reads_Ns_b7,
            reads_Ns_b8,
            reads_Ns_b9,
            fillvalue=0
        )
    ]

    print(reads_Ns_all)
    
    # subsampling and taxa assignment 
    
    if True:
        for i, (n_b14, n_b5, n_b6, n_b7, n_b8, n_b9) in enumerate(
            zip_longest(
                reads_Ns_b14,
                reads_Ns_b5,
                reads_Ns_b6,
                reads_Ns_b7 or [],
                reads_Ns_b8 or [],
                reads_Ns_b9 or [],
                fillvalue=None
            )
        ):
            
            ## creating subsampled fastq + decontamination
            # b1-4
            file_readers.save_pfastq_Nreads(reads_f1_b14_a, reads_f2_b14_a, n_b14)
            tmp_f1_b14 = reads_f1_b14_a.replace(".fastq", "_" + str(n_b14) + ".fastq")
            tmp_f2_b14 = reads_f2_b14_a.replace(".fastq", "_" + str(n_b14) + ".fastq")
            process_data.dehostaminate_pe(tmp_f1_b14, tmp_f2_b14, ref_file, method = "bwa", threads=32)
            # b5
            file_readers.save_pfastq_Nreads(reads_f1_b5_a, reads_f2_b5_a, n_b5)
            tmp_f1_b5 = reads_f1_b5_a.replace(".fastq", "_" + str(n_b5) + ".fastq")
            tmp_f2_b5 = reads_f2_b5_a.replace(".fastq", "_" + str(n_b5) + ".fastq")
            process_data.dehostaminate_pe(tmp_f1_b5, tmp_f2_b5, ref_file, method = "bwa", threads=32)
            # b6
            file_readers.save_pfastq_Nreads(reads_f1_b6_a, reads_f2_b6_a, n_b6)
            tmp_f1_b6 = reads_f1_b6_a.replace(".fastq", "_" + str(n_b6) + ".fastq")
            tmp_f2_b6 = reads_f2_b6_a.replace(".fastq", "_" + str(n_b6) + ".fastq")
            process_data.dehostaminate_pe(tmp_f1_b6, tmp_f2_b6, ref_file, method = "bwa", threads=32)
            # b7
            if reads_f1_b7_a is not None and reads_f2_b7_a is not None and n_b7 is not None:
                file_readers.save_pfastq_Nreads(reads_f1_b7_a, reads_f2_b7_a, n_b7)
                tmp_f1_b7 = reads_f1_b7_a.replace(".fastq", "_" + str(n_b7) + ".fastq")
                tmp_f2_b7 = reads_f2_b7_a.replace(".fastq", "_" + str(n_b7) + ".fastq")
                process_data.dehostaminate_pe(tmp_f1_b7, tmp_f2_b7, ref_file, method = "bwa", threads=32)
                tmp_dec_b7 = tmp_f1_b7.replace(".fastq", "_bwa_unhosted.fastq") 
            # b8
            if reads_f1_b8_a is not None and reads_f2_b8_a is not None and n_b8 is not None:
                file_readers.save_pfastq_Nreads(reads_f1_b8_a, reads_f2_b8_a, n_b8)
                tmp_f1_b8 = reads_f1_b8_a.replace(".fastq", "_" + str(n_b8) + ".fastq")
                tmp_f2_b8 = reads_f2_b8_a.replace(".fastq", "_" + str(n_b8) + ".fastq")
                process_data.dehostaminate_pe(tmp_f1_b8, tmp_f2_b8, ref_file, method = "bwa", threads=32)
                tmp_dec_b8 = tmp_f1_b8.replace(".fastq", "_bwa_unhosted.fastq")
            # b9
            if reads_f1_b9_a is not None and reads_f2_b9_a is not None and n_b9 is not None:
                file_readers.save_pfastq_Nreads(reads_f1_b9_a, reads_f2_b9_a, n_b9)
                tmp_f1_b9 = reads_f1_b9_a.replace(".fastq", "_" + str(n_b9) + ".fastq")
                tmp_f2_b9 = reads_f2_b9_a.replace(".fastq", "_" + str(n_b9) + ".fastq")
                process_data.dehostaminate_pe(tmp_f1_b9, tmp_f2_b9, ref_file, method = "bwa", threads=32)
                tmp_dec_b9 = tmp_f1_b9.replace(".fastq", "_bwa_unhosted.fastq")
            
            # assembling
            tmp_dec_b14 = tmp_f1_b14.replace(".fastq", "_bwa_unhosted.fastq")
            tmp_dec_b5 = tmp_f1_b5.replace(".fastq", "_bwa_unhosted.fastq")
            tmp_dec_b6 = tmp_f1_b6.replace(".fastq", "_bwa_unhosted.fastq")
            
            out_merged = main_path + "/merged_" + samp + ".fastq"
            #merge_fastqs(tmp_dec_b14, tmp_dec_b5, tmp_dec_b6, tmp_dec_b7, tmp_dec_b8, tmp_dec_b9, out_merged)
            merge_fastqs([tmp_dec_b14, tmp_dec_b5, tmp_dec_b6, tmp_dec_b7, tmp_dec_b8, tmp_dec_b9],out_merged)
            ## assembling
            process_data.assemble_decon_se(main_path, db_path, out_merged, samp_id=samp, threads=32, min_con_len = 500)
            
            ## assigning
            fa_path = main_path + "/metagenome_contigs_" + samp + ".fastq"
            bdb_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/blastn/bac_refs_refseq/bac_refs_refseq"
            #bdb_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/nt_blast/nt_blast"
            out_path = out_main_path + "/" + samp + "_" + str(reads_Ns_all[i]) + "_blastn_report.txt"
            process_data.run_blastn(fa_path, bdb_path, out_path, nthreads=32, evalue=1e-6, max_ts=5, max_h=1)

            os.remove(tmp_dec_b14)
            os.remove(tmp_dec_b5)
            os.remove(tmp_dec_b6)
            os.remove(tmp_dec_b7)
            os.remove(tmp_dec_b8)
            os.remove(tmp_f1_b14)
            os.remove(tmp_f2_b14)
            os.remove(tmp_f1_b5)
            os.remove(tmp_f2_b5)
            os.remove(tmp_f1_b6)
            os.remove(tmp_f2_b6)
            if os.path.exists(tmp_f1_b7):
                os.remove(tmp_f1_b7)
                os.remove(tmp_f2_b7)
            if os.path.exists(tmp_f1_b7):
                os.remove(tmp_f1_b8)
                os.remove(tmp_f2_b8)
            if os.path.exists(tmp_f1_b9):
                os.remove(tmp_f1_b9)
                os.remove(tmp_f2_b9)
            os.remove(out_merged)
            os.remove(fa_path)
    
    # remove atropos files
    os.remove(reads_f1_b14_a)
    os.remove(reads_f2_b14_a)
    os.remove(reads_f1_b5_a)
    os.remove(reads_f2_b5_a)
    os.remove(reads_f1_b6_a)
    os.remove(reads_f2_b6_a)
    if os.path.exists(tmp_f1_b7):
        os.remove(reads_f1_b7_a)
        os.remove(reads_f2_b7_a)
    if os.path.exists(tmp_f1_b8):
        os.remove(reads_f1_b8_a)
        os.remove(reads_f2_b8_a)
    if os.path.exists(tmp_f1_b9):
        os.remove(reads_f1_b9_a)
        os.remove(reads_f2_b9_a)