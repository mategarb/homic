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


# db creation for human
db_path = "/gpfs/commons/home/mgarbulowski/dbs/hs_wgs"
#ref_path = "/gpfs/commons/home/mgarbulowski/proj_shm/references/GCF_000001635.27_GRCm39_genomic.fna"
#kraken2.prepare_db(db_path, ref_path)


# call the fun
path = "/gpfs/commons/home/mgarbulowski/data_bi_wgs/us"
#
print("Filtering...")
file1 = path + "/SRR34978769_1.fastq"
file2 = path + "/SRR34978769_2.fastq"
process_data.trim_decon(db_path, file1, file2)
#
print("Assembling...")
file1 = path + "/SRR34978769_k2_1.fastq"
file2 = path + "/SRR34978769_k2_2.fastq"
process_data.assemble_decon(path, db_path, file1, file2)
#
# before blastn, load lib using: export PATH=$PATH:$HOME/ncbi-blast-2.16.0+/bin
print("Assigning taxa...")
fa_path1 = "/gpfs/commons/home/mgarbulowski/data_bi_wgs/us/mic_contigs.fastq" # fix this, cause it takes mic_contigs.fastq
db_path1 = "/gpfs/commons/home/mgarbulowski/ref_dbs/GbBacU/all_seqs.fna"
out1 = "/gpfs/commons/home/mgarbulowski/data_bi_wgs/us/SRR34978769_blastn.txt"
process_data.run_blastn(fa_path1, db_path1, out1)

print("Done")
#os.remove(fa_path1)
os.rename(fa_path1, "/gpfs/commons/home/mgarbulowski/data_bi_wgs/us/SRR34978769_mic_contigs.fastq")

##
#
print("Filtering...")
file1 = path + "/SRR34978773_1.fastq"
file2 = path + "/SRR34978773_2.fastq"
process_data.trim_decon(db_path, file1, file2)
#
print("Assembling...")
file1 = path + "/SRR34978773_k2_1.fastq"
file2 = path + "/SRR34978773_k2_2.fastq"
process_data.assemble_decon(path, db_path, file1, file2)
#
# before blastn, load lib using: export PATH=$PATH:$HOME/ncbi-blast-2.16.0+/bin
print("Assigning taxa...")
fa_path1 = "/gpfs/commons/home/mgarbulowski/data_bi_wgs/us/mic_contigs.fastq" # fix this, cause it takes mic_contigs.fastq
db_path1 = "/gpfs/commons/home/mgarbulowski/ref_dbs/GbBacU/all_seqs.fna"
out1 = "/gpfs/commons/home/mgarbulowski/data_bi_wgs/us/SRR34978773_blastn.txt"
process_data.run_blastn(fa_path1, db_path1, out1)

print("Done")
#os.remove(fa_path1)
os.rename(fa_path1, "/gpfs/commons/home/mgarbulowski/data_bi_wgs/us/SRR34978773_mic_contigs.fastq")

##
#
print("Filtering...")
file1 = path + "/SRR34978795_1.fastq"
file2 = path + "/SRR34978795_2.fastq"
process_data.trim_decon(db_path, file1, file2)
#
print("Assembling...")
file1 = path + "/SRR34978795_k2_1.fastq"
file2 = path + "/SRR34978795_k2_2.fastq"
process_data.assemble_decon(path, db_path, file1, file2)
#
# before blastn, load lib using: export PATH=$PATH:$HOME/ncbi-blast-2.16.0+/bin
print("Assigning taxa...")
fa_path1 = "/gpfs/commons/home/mgarbulowski/data_bi_wgs/us/mic_contigs.fastq" # fix this, cause it takes mic_contigs.fastq
db_path1 = "/gpfs/commons/home/mgarbulowski/ref_dbs/GbBacU/all_seqs.fna"
out1 = "/gpfs/commons/home/mgarbulowski/data_bi_wgs/us/SRR34978795_blastn.txt"
process_data.run_blastn(fa_path1, db_path1, out1)

print("Done")
#os.remove(fa_path1)
os.rename(fa_path1, "/gpfs/commons/home/mgarbulowski/data_bi_wgs/us/SRR34978795_mic_contigs.fastq")