import pickle
import os
import time
import sys
import tensorflow.keras as keras
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src') # to load our package
from homic import file_readers, dl_evaluation, make_plots, simulate_16S
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
################################# Load files

#### barcodes
#path_bcodes = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/10015_barcodes.txt'
#bcodes = file_readers.load_barcodes(path_bcodes)
bcodes = simulate_16S.simulate_barcodes(100,100)
print("Number of barcodes: ")
print(len(bcodes))

#### output from Kraken2
path_krk = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_data_medium_5000ps_ln/kraken2_5000ps_val_simulated_output.txt'
krk_out = file_readers.load_kraken2_output(path_krk)
krk_preds = krk_out['taxid']
print("Number of predictions: ")
print(len(krk_preds))

#### fastq with simulated reads
path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_data_medium_5000ps_ln/SRR25456942_5000ps_val_simulated.fastq'
heads, reads, quals = file_readers.fastq(path)
print("Number of reads: ")
print(len(reads))

#### dl model
input_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs'
input_model = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/models_old/spemic62m/model_LSTM.keras'
model = keras.models.load_model(os.path.join(input_path, input_model))

# summarize model
#model.summary()

#### encoder
input_encoder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/models_old/spemic62m/train_encoder_LSTM.h5'
encoder = pickle.load(open(os.path.join(input_path, input_encoder), 'rb'))

#### taxa info, fastq headers with species labels, gold truth
path_head = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_data_medium_5000ps_ln/SRR25456942_5000ps_val_genus_species.txt'

path_krk_rep = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_data_medium_5000ps_ln/kraken2_5000ps_val_simulated__report.txt'
dl_evaluation.taxa_per_spots(path_krk_rep, path_head, path, input_model, input_encoder)

breakpoint()

#####################################################################################################
################################# Perform the analysis

startt = time.time()
fastq_spot_d, info = file_readers.make_benchmark_table(path_head, reads, krk_preds, bcodes)
endt = time.time()
print("Time of make_benchmark_table: ")
print((endt - startt)/60)

########### Kraken2 + DL reassignment #############
startt = time.time()
cluster_l, reassign_d = dl_evaluation.reassign_classes_per_spot(info, model, encoder)
endt = time.time()
print("Time of reassign_classes_per_spot: ")
print((endt - startt)/60)

taxa_orders = ["species", "genus", "family", "order", "class", "phylum", "superkingdom"]

startt = time.time()
pearson_d, braycurtis_d, accuracy_d, precision_d, recall_d = dl_evaluation.merge_prediction_results(info, cluster_l, fastq_spot_d, taxa_orders, reassign_d)
endt = time.time()
print("Time of merge_prediction_results: ")
print((endt - startt)/60)

# Calculate average pearson per taxa level
#path_figs = '/gpfs/commons/home/mgarbulowski/proj_shm/figs/'
#make_plots.bench_barplot(pearson_d, taxa_orders, path_figs, mtype="pearson")
#make_plots.bench_barplot(accuracy_d, taxa_orders, path_figs, mtype="accuracy")
#make_plots.bench_barplot(precision_d, taxa_orders, path_figs, mtype="precision")
#make_plots.bench_barplot(recall_d, taxa_orders, path_figs, mtype="recall")
#make_plots.bray_curtis_bench(braycurtis_d, taxa_orders, path_figs)

startt = time.time()
stats_d = dl_evaluation.per_spot_stats(info, reassign_d, fastq_spot_d, taxa_orders)
endt = time.time()
print("Time of per_spot_stats: ")
print((endt - startt)/60)

#path_csv = '/gpfs/commons/home/mgarbulowski/proj_shm/figs/report_stats_small_val_data.csv'
#stats_ddf.to_csv(path_csv)

#print(stats_d['10x10'])                    
#print(stats_d['27x8'])





