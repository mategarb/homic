import sys
sys.path.append('/proj/berzelius-2024-407/users/x_magar/homic/src') # to load our package
from homic import file_readers, dl_evaluation, make_plots, dl_model, kraken2

import pickle
import os
import time
import io
import re
import numpy as np
import pandas as pd
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from numpy import argmax
from scipy.spatial import distance
from functools import reduce
from scipy import stats
from sklearn import metrics
#from umi_tools import UMIClusterer
from collections import Counter, defaultdict
from itertools import islice,tee

## converting to reverse complementary (if needed)
# ref_path = "/proj/berzelius-2024-407/users/x_magar/data/references/mm_16S.fasta"
# file_readers.save_fasta_as_rev_comp(ref_path)

#data_path = '/proj/berzelius-2024-407/users/x_magar/data/asf_data/SRR25456916.fastq'
#file_readers.save_fastq_as_rev_comp(data_path)
#breakpoint()
#####################################################################################################
################################# Creating DB for kraken2 - done once per DB

#db_path = "/proj/berzelius-2024-407/users/x_magar/dbs/mm_wgs"
#ref_path = "/proj/berzelius-2024-407/users/x_magar/data/references/GRCm39.genome.fa"
#kraken2.prepare_db(db_path, ref_path)

#####################################################################################################
################################# Running Kraken2 classification

db_path = "/proj/berzelius-2024-407/users/x_magar/dbs/k2_standard_25" # 
data_path = '/proj/berzelius-2024-407/users/x_magar/data/asf_data/SRR25456916.fastq'
startt = time.time()
out = kraken2.classify(db_path = db_path,
                input_file = data_path)
endt = time.time()
print("Time of Kraken2 classification: ")
print((endt - startt)/60)

out.to_csv('/proj/berzelius-2024-407/users/x_magar/data/asf_data/kraken2_classes_SRR25456916_mic.csv', index=False)

#kraken2.classify(, threads=2)