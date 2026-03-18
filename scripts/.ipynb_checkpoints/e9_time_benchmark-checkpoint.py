import time
import sys
import os
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src')
from homic import file_readers, simulate_16S, dl_model, make_plots, dl_evaluation
import numpy as np
import pandas as pd
import pickle
from collections import Counter, defaultdict
from tensorflow.keras import Model
import tensorflow.keras as keras

from functools import reduce
from scipy import stats
from sklearn import metrics
import statistics

from itertools import islice,tee
import io
import re

# static parameters
nsp = 10000 # number of reads per species

#species_tra = None # species_list based on training data to match the same species, cane be left None
er = 0.001
ew = (1, 2, 1)
tr = [0.01, 0.05] # range to truncate read from both ends (percentage), e.g. [0.05,0.1] will draw two values of percentage 


path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/references_3prime/3prime_65sp_1000_comGen.txt"
path2 = "/gpfs/commons/home/mgarbulowski/proj_shm/inputs/raw_data/SRR25456942_D2_2.fastq" # fastq 1
#path3 = "/gpfs/commons/home/mgarbulowski/proj_shm/inputs/raw_data/SRR25456944_C2_2.fastq" # fastq 2

output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/time_benchmark_data'

#####################################################################################################
#####################################################################################################
################################# 1M


# outs
#output_path = output_path_folder + "/SRR25456942_3prime_1M"
#figs_path = output_path_folder + "/figs"
#
#
#mic_refs = file_readers.fasta(path1)
#r2_header_lines, r2_read_lines, r2_qual_lines = file_readers.fastq(path2)
#
#
#startt = time.time()
#starts, quals, hds, species_list = simulate_16S.training_data_fast(1000000, # total number of reads
#                                                                      output_path+"_er",
#                                                                      mic_refs,
#                                                                      r2_header_lines,
#                                                                      r2_read_lines,
#                                                                      r2_qual_lines,
#                                                                      impute_errors = True,
#                                                                      trunc_range = [0.1,0.2],
#                                                                      print_stats = True)
#
#endt = time.time()
#print("Time of data simulation[min.]: ")
#print((endt - startt)/60)
#
#####################################################################################################
#####################################################################################################
################################# 10M


# outs
#output_path = output_path_folder + "/SRR25456942_3prime_10M"
#figs_path = output_path_folder + "/figs"
##
##
#mic_refs = file_readers.fasta(path1)
#r2_header_lines, r2_read_lines, r2_qual_lines = file_readers.fastq(path2)
##
##
#startt = time.time()
#starts, quals, hds, species_list = simulate_16S.training_data_fast(10000000, # total number of reads
#                                                                      output_path+"_er",
#                                                                      mic_refs,
#                                                                      r2_header_lines,
#                                                                      r2_read_lines,
#                                                                      r2_qual_lines,
#                                                                      impute_errors = True,
#                                                                      trunc_range = [0.1,0.2],
#                                                                      print_stats = True)
#
#endt = time.time()
#print("Time of data simulation[min.]: ")
#print((endt - startt)/60)
#breakpoint()
######################################################################################################
######################################################################################################
################################# 150M

# outs
#output_path = output_path_folder + "/SRR25456942_3prime_150M_2"
#figs_path = output_path_folder + "/figs"
#
#
#mic_refs = file_readers.fasta(path1)
#r2_header_lines, r2_read_lines, r2_qual_lines = file_readers.fastq(path2)
#
#
#startt = time.time()
#starts, quals, hds, species_list = simulate_16S.training_data_fast(150000000, # total number of reads
#                                                                      output_path+"_er",
#                                                                      mic_refs,
#                                                                      r2_header_lines,
#                                                                      r2_read_lines,
#                                                                      r2_qual_lines,
#                                                                      impute_errors = True,
#                                                                      trunc_range = [0.1,0.2],
#                                                                      print_stats = True)
#
#endt = time.time()
#print("Time of data simulation[min.]: ")
#print((endt - startt)/60)

#####################################################################################################
#####################################################################################################
################################# 300M

# outs
#output_path = output_path_folder + "/SRR25456942_3prime_300M"
#figs_path = output_path_folder + "/figs"
#
#
#mic_refs = file_readers.fasta(path1)
#r2_header_lines, r2_read_lines, r2_qual_lines = file_readers.fastq(path2)
#
#
#startt = time.time()
#starts, quals, hds, species_list = simulate_16S.training_data_fast(300000000, # total number of reads
#                                                                      output_path+"_er",
#                                                                      mic_refs,
#                                                                      r2_header_lines,
#                                                                      r2_read_lines,
#                                                                      r2_qual_lines,
#                                                                      impute_errors = True,
#                                                                      trunc_range = [0.1,0.2],
#                                                                      print_stats = True)
#
#endt = time.time()
#print("Time of data simulation[min.]: ")
#print((endt - startt)/60)



#### dl model
input_model = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/model_950701_totreads_11epochs_50batches.keras'
input_encoder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/tra_encoder_950701_totreads_11epochs_50batches.h5'

# loading model and encoder
model = keras.models.load_model(input_model)
encoder = pickle.load(open(input_encoder, 'rb'))


###### TEST ############ TEST ############ TEST ######
path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/time_benchmark_data/SRR25456942_3prime_10M_er_tra_simulated.fastq'
heads, reads, quals = file_readers.fastq(path)

del heads, quals

print("Data loaded")
print("Number of reads:")
print(len(reads))

# initialize data of lists.
data = {'read': reads[:7500000]}

# Create DataFrame
info_xy = pd.DataFrame(data)

startt = time.time()
pred_classes, freq_pred_classes = dl_model.predict_class_for_reads(info_xy, model, encoder, rank="none")
endt = time.time()
print("Regular")
print("Time of prediction [min.]: ")
print((endt - startt)/60)

#startt = time.time()
#pred_classes, freq_pred_classes = dl_model.par_predict_class_for_reads(info_xy, model, encoder, rank="none")
#endt = time.time()
#print("Parallel")
#print("Time of prediction [min.]: ")
#print((endt - startt)/60)

