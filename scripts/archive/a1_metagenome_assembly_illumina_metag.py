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

######################################################################################################
### db construction for human ###

db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/hs_wgs"
#ref_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/references/Homo_sapiens.GRCh38.dna.toplevel.fa"
#kraken2.prepare_db(db_path, ref_path)

# define paths
path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes"
ada_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/adapters.fa"

# samples id 
samps = ["KP005", "KP012", "KP013", "KP016", "KP024", "KP025", "KP026", "KP027", "KP029","KP033","KP035","KP038","KP040","KP046","KP047","KP048"]

# samples id 
#samps = ["KP024", "KP025", "KP026", "KP027", "KP029","KP033"]
#samps = ["KP033"]
#samps = ["KP048"]
#bcl2fastq --runfolder-dir /gpfs/commons/instruments/miseq/M00911/251027_M00911_0311_000000000-M5VLH --output-dir proj_shm/metagenomes_lib



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
if False:
    print("QC on trimmed samples...")
    for idx, samp in enumerate(samps):  
        cmd = ["fastqc",
               "--outdir",
               "metagenomes_lib/fastqc_trimmed",
               "metagenomes_lib/Metagenomes/016-Meta-" + str(idx+1) + "/" + samp + "_S" + str(idx+1) + "_L001_k2_1.fastq",]
               
        cmd2 = ["fastqc",
               "--outdir",
               "metagenomes_lib/fastqc_trimmed",
               "metagenomes_lib/Metagenomes/016-Meta-" + str(idx+1) + "/" + samp + "_S" + str(idx+1) + "_L001_k2_2.fastq",]
                
        subprocess.call(cmd)
        subprocess.call(cmd2)

#####################################################################################################
### assembling metagenome ###
if False:
    path2 = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes/"
    print("Assembling "+str(len(samps))+" samples...")
    for idx, samp in enumerate(samps):  
        file1 = path2 + "016-Meta-" + str(idx+1) + "/" + samp + "_S" + str(idx+1) + "_L001_k2_1.fastq"
        file2 = path2 + "016-Meta-" + str(idx+1) + "/" + samp + "_S" + str(idx+1) + "_L001_k2_2.fastq"
        process_data.assemble_decon(path2, db_path, file1, file2, samps[idx])

#####################################################################################################
################################# Running binning
if False:
    assembly_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes/metagenome_contigs_KP005.fastq"
    file_1_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes/016-Meta-1/KP005_S1_L001_k2_1.fastq"
    file_2_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes/016-Meta-1/KP005_S1_L001_k2_2.fastq"
    out_dir = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes/016-Meta-1/metabat_bins"
    
    process_data.bow_bat(assembly_path, file_1_path, file_2_path, out_dir, False)


#####################################################################################################
### before blastn, load lib once per session ### 
if False:
    cmd = "export PATH=$PATH:$HOME/ncbi-blast-2.16.0+/bin"
    subprocess.call(cmd, shell=True, executable="/bin/bash")
    # other solution (otherwise do it manually in bash)
    #blast_path = os.path.expanduser("~/ncbi-blast-2.16.0+/bin")
    #os.environ["PATH"] += os.pathsep + blast_path

#####################################################################################################
### taxa assignment with blastn ### 
if True:
    print("Assigning taxa...")
    # Add BLAST+ bin directory to PATH
    os.environ["PATH"] = "/gpfs/commons/home/mgarbulowski/ncbi-blast-2.16.0+/bin:" + os.environ["PATH"]
    for idx, samp in enumerate(samps):  
        fa_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/Metagenomes/metagenome_contigs_"  + samp + ".fastq"
        #db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/nt_blast/nt_blast"
        #db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/bac_refs_refseq/bac_refs_refseq"
        db_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/ref_dbs/bac_refs_genban/bac_refs_genban"
        #out_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/blast_nt_top5/" + samp + "_blastn_report.txt"
        #out_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/blast_rs_top5/" + samp + "_blastn_report.txt"
        out_path = "/gpfs/commons/home/mgarbulowski/016_proj_shm/metagenomes_lib/blast_gb_top5/" + samp + "_blastn_report.txt"
        process_data.run_blastn(fa_path, db_path, out_path, nthreads=32, evalue=1e-6, max_ts=5, max_h=1)

    print("Done!")

#####################################################################################################
################################# Creating DB for kraken2 - done once per DB
if False:
    db_path = "/proj/berzelius-2024-407/users/x_magar/dbs/mm_wgs"
    ref_path = "/proj/berzelius-2024-407/users/x_magar/data/references/GRCm39.genome.fa"
    kraken2.prepare_db(db_path, ref_path)

#####################################################################################################
################################# Running Kraken2 classification
if False:
    db_path = "/proj/berzelius-2024-407/users/x_magar/dbs/k2_standard_25" # 
    data_path = '/proj/berzelius-2024-407/users/x_magar/data/asf_data/SRR25456916.fastq'
    startt = time.time()
    out = kraken2.classify(db_path = db_path,
                    input_file = data_path)
    endt = time.time()
    print("Time of Kraken2 classification: ")
    print((endt - startt)/60)
    
    out.to_csv('/proj/berzelius-2024-407/users/x_magar/data/asf_data/kraken2_classes_SRR25456916_mic.csv', index=False)







