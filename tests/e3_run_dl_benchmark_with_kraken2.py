import sys
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src') # to load our package
from homic import file_readers, dl_evaluation, make_plots, dl_model, kraken2

import pickle
import os
import time
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from numpy import argmax
from scipy.spatial import distance
import statistics
from functools import reduce
from scipy import stats
#from umi_tools import UMIClusterer
from collections import Counter, defaultdict
from tensorflow.keras import Model
from itertools import islice,tee
import io
import re
#####################################################################################################
################################# Kraken2

########### Creating DB for kraken2 - done once per DB

#db_path = "/gpfs/commons/home/mgarbulowski/proj_shm/references_3prime/kraken_db1000"
db_path = "/gpfs/commons/home/mgarbulowski/ref_dbs/standard_db_k2"

#ref_path =  "/gpfs/commons/home/mgarbulowski/proj_shm/references_3prime/3prime_65sp_1000_comGen.txt"
#startt0 = time.time()
#kraken2.prepare_db(db_path, ref_path)
#endt0 = time.time()
#print("Time of Kraken2 db creation: ")
#print((endt0 - startt0)/60)

########### Running Kraken2 classification

#data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_1000_cg/SRR25456942_3prime_10000ps_val_simulated.fastq'

data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/time_benchmark_data/SRR25456942_3prime_1M_er_tra_simulated.fastq'
#
startt = time.time()
out = kraken2.classify(db_path = db_path,
                input_file = data_path)
endt = time.time()
print("Time of Kraken2 classification: ")
print((endt - startt)/60)
#
##out.to_csv('/gpfs/commons/home/mgarbulowski/proj_shm/outputs/kraken_bench/kraken2_classes_3prime1000_1M.csv', index=False)
#
##2
data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/time_benchmark_data/SRR25456942_3prime_10M_er_tra_simulated.fastq'
#
#startt = time.time()
out = kraken2.classify(db_path = db_path,
                input_file = data_path)
endt = time.time()
print("Time of Kraken2 classification: ")
print((endt - startt)/60)
#
#out.to_csv('/gpfs/commons/home/mgarbulowski/proj_shm/outputs/kraken_bench/kraken2_classes_3prime1000_10M.csv', index=False)
#
##3
data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/time_benchmark_data/SRR25456942_3prime_150M_er_tra_simulated.fastq'
#
startt = time.time()
out = kraken2.classify(db_path = db_path,
                input_file = data_path)
endt = time.time()
print("Time of Kraken2 classification: ")
print((endt - startt)/60)
#
##out.to_csv('/gpfs/commons/home/mgarbulowski/proj_shm/outputs/kraken_bench/kraken2_classes_3prime1000_150M.csv', index=False)
#breakpoint()
#
#4
data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/time_benchmark_data/SRR25456942_3prime_300M_er_tra_simulated.fastq'

startt = time.time()
out = kraken2.classify(db_path = db_path,
                input_file = data_path)
endt = time.time()
print("Time of Kraken2 classification: ")
print((endt - startt)/60)

#out.to_csv('/gpfs/commons/home/mgarbulowski/proj_shm/outputs/kraken_bench/kraken2_classes_3prime1000_300M.csv', index=False)


breakpoint()

#####################################################################################################
################################# DL

#### barcodes
path_bcodes = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/10015_barcodes.txt'
bcodes = file_readers.load_barcodes(path_bcodes)

#### output from Kraken2
path_krk = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/kraken2_5000ps_val_simulated_output.txt'
#path_krk = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_small/kraken2_val_simulated_output.txt'
krk_out = file_readers.load_kraken2_output(path_krk)

#### fastq with simulated reads
path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/SRR25456942_5000ps_val_simulated.fastq'
# path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/SRR25456942_D2_20000ps_val_simulated.fastq'
#path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_small/SRR25456942_val_simulated.fastq' # small data, 
# path = '/gpfs/commons/home/mgarbulowski/proj_shm/old_files/simulated_R2_150.fastq'
# read input fastq file, ending with "_matchBarcode.fastq"
heads, reads, quals = file_readers.fastq(path)

print("number of reads:")
print(len(heads))

#### dl model
input_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs'
input_model = 'model_1000000_totreads_11epochs_50batches_LSTM.keras'
#input_model = 'model_500000_totreads_11epochs_50batches_.keras'
model = keras.models.load_model(os.path.join(input_path, input_model))

# summarize model
# model.summary()

#### encoder
input_encoder = 'tra_encoder_1000000_totreads_11epochs_50batches_LSTM.h5'
#input_encoder = 'val_encoder_500000_totreads_11epochs_50batches_.h5'
encoder = pickle.load(open(os.path.join(input_path, input_encoder), 'rb'))

#### taxa info, fastq headers with species labels
# this is some unspecified pickle file that contains all the data
path_head = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/SRR25456942_5000ps_val_genus_species.txt'
#path_head = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/SRR25456942_D2_20000ps_val_genus_species.txt'
#path_head = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_small/SRR25456942_val_genus_species.txt' # small data
#path_head = '/gpfs/commons/home/mgarbulowski/proj_shm/old_files/header_taxa.txt' # old data

krk_preds = krk_out['taxid']

fastq_spot_d, info = file_readers.make_benchmark_table(path_head, reads, krk_preds, bcodes)



## columns in info
#'fastq' - fastq full header
#'tile' - tile id (from header)
#'x' - position x  (from header)
#'y' - position y (from header)
#'taxa1' - species part I, truth
#'taxa2' - species part II, truth
#'read' - read sequence 
#'taxa_predictions' - taxid of predictions from Kraken2
#'taxa' - truth species, truth
#'taxa_order' - truth taxa information, ordered 
#'superkingdom' - taxid predictions from Kraken2 translated to taxa info via ete3
#'phylum' - taxid predictions from Kraken2 translated to taxa info via ete3
#'class' - taxid predictions from Kraken2 translated to taxa info via ete3
#'order' - taxid predictions from Kraken2 translated to taxa info via ete3
#'family' - taxid predictions from Kraken2 translated to taxa info via ete3
#'genus' - taxid predictions from Kraken2 translated to taxa info via ete3
#'species' - taxid predictions from Kraken2 translated to taxa info via ete3
#'barcode' - barcode sequence
#'Bx' - barcode position X, spot definition (for synthetic data, assigned randomly)
#'By' - barcode position Y, spot definition (for synthetic data, assigned randomly)


# kraken 2 report file, fastq header - taxa text file, fastq file
# dl_evaluation.taxa_per_spots(report_file, ft_file, fastq_file)

########### Kraken2 + DL reassignment #############

taxa_orders = ["species", "genus", "family", "order", "class", "phylum", "superkingdom"]
cluster_l, reassign_d = dl_evaluation.reassign_classes_per_spot(info, model, encoder)


pearson_d, braycurtis_d, accuracy_d, precision_d, recall_d = dl_evaluation.merge_prediction_results(info, cluster_l, fastq_spot_d, taxa_orders, reassign_d)


# Calculate average pearson per taxa level
#path_figs = '/gpfs/commons/home/mgarbulowski/proj_shm/figs/'
#make_plots.bench_barplot(pearson_d, taxa_orders, path_figs, mtype="pearson")
#make_plots.bench_barplot(accuracy_d, taxa_orders, path_figs, mtype="accuracy")
#make_plots.bench_barplot(precision_d, taxa_orders, path_figs, mtype="precision")
#make_plots.bench_barplot(recall_d, taxa_orders, path_figs, mtype="recall")

#make_plots.bray_curtis_bench(braycurtis_d, taxa_orders, path_figs)

#stats_d = dl_evaluation.per_spot_stats(info, reassign_d, fastq_spot_d, taxa_orders)
#path_csv = '/gpfs/commons/home/mgarbulowski/proj_shm/figs/report_stats_small_val_data.csv'

#stats_ddf.to_csv(path_csv)

#print(stats_d['10x10'])                    
#print(stats_d['27x8'])





