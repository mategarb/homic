import numpy as np
import pandas as pd
import statistics
import seaborn as sns
import os
import subprocess
import shutil
import sys
sys.path.append('/proj/berzelius-2024-407/users/x_magar/homic/src') # to load our package
from homic import file_readers, kraken2, process_data

# db creation for mouse
# db_path = "/proj/berzelius-2024-407/users/x_magar/dbs/mm_wgs"
# ref_path = "/proj/berzelius-2024-#407/users/x_magar/references/GCF_000001635.27_GRCm39_genomic.fna"
# kraken2.prepare_db(db_path, ref_path)


# call the fun
path = "/proj/berzelius-2024-407/users/x_magar/data/spf_data"

file1 = path + "/SRR25456947_1.fastq"
file2 = path + "/SRR25456947_2.fastq"

process_data.trim_decon(path, file1, file2)

file1 = path + "/SRR25456949_1.fastq"
file2 = path + "/SRR25456949_2.fastq"

process_data.trim_decon(path, file1, file2)

file1 = path + "/SRR25456950_1.fastq"
file2 = path + "/SRR25456950_2.fastq"

process_data.trim_decon(path, file1, file2)

file1 = path + "/SRR25456957_1.fastq"
file2 = path + "/SRR25456957_2.fastq"

process_data.trim_decon(path, file1, file2)

file1 = path + "/SRR25456958_1.fastq"
file2 = path + "/SRR25456958_2.fastq"

process_data.trim_decon(path, file1, file2)


