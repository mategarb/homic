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

# db creation for mouse
# db_path = "/proj/berzelius-2024-407/users/x_magar/dbs/mm_wgs"
# ref_path = "/proj/berzelius-2024-#407/users/x_magar/references/GCF_000001635.27_GRCm39_genomic.fna"
# kraken2.prepare_db(db_path, ref_path)

# db creation for human
db_path = "/gpfs/commons/home/mgarbulowski/dbs/mm_wgs"
#ref_path = "/gpfs/commons/home/mgarbulowski/proj_shm/references/GCF_000001635.27_GRCm39_genomic.fna"
#kraken2.prepare_db(db_path, ref_path)

#breakpoint()

# call the fun
path = "/gpfs/commons/home/mgarbulowski/data_moumic"

#file1 = path + "/all_reads_1.fastq"
#file2 = path + "/all_reads_2.fastq"
#file1 = path + "/ERR5838529_1.fastq"
#file2 = path + "/ERR5838529_2.fastq"
#process_data.trim_decon(path, db_path, file1, file2)


#file1 = path + "/all_reads_k2_1.fastq"
#file2 = path + "/all_reads_k2_2.fastq"
#file1 = path + "/ERR5838529_k2_1.fastq"
#file2 = path + "/ERR5838529_k2_2.fastq"
#process_data.assemble_decon(path, db_path, file1, file2)
print("1 start")
fa_path1 = "/gpfs/commons/home/mgarbulowski/data_moumic/ERR5838516_mic_contigs.fastq"
db_path1 = "/gpfs/commons/home/mgarbulowski/ref_dbs/GbBacU/all_seqs.fna"
out1 = "/gpfs/commons/home/mgarbulowski/data_moumic/ERR5838516_blastn.txt"
process_data.run_blastn(fa_path1, db_path1, out1)
print("1 done")

print("2 start")
fa_path2 = "/gpfs/commons/home/mgarbulowski/data_moumic/ERR5838523_mic_contigs.fastq"
db_path2 = "/gpfs/commons/home/mgarbulowski/ref_dbs/GbBacU/all_seqs.fna"
out2 = "/gpfs/commons/home/mgarbulowski/data_moumic/ERR5838523_blastn.txt"
process_data.run_blastn(fa_path2, db_path2, out2)
print("2 done")

print("3 start")
fa_path3 = "/gpfs/commons/home/mgarbulowski/data_moumic/ERR5838529_mic_contigs.fastq"
db_path3 = "/gpfs/commons/home/mgarbulowski/ref_dbs/GbBacU/all_seqs.fna"
out3 = "/gpfs/commons/home/mgarbulowski/data_moumic/ERR5838529_blastn.txt"
process_data.run_blastn(fa_path3, db_path3, out3)
print("3 done")
