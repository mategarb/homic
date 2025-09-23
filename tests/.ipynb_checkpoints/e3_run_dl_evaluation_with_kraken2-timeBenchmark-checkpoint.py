import pickle
import os
import time
import sys
import tensorflow.keras as keras
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src') # to load our package
from homic import file_readers, dl_evaluation, make_plots
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
path_bcodes = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/10015_barcodes.txt'
bcodes = file_readers.load_barcodes(path_bcodes)
print(type(bcodes))

#### output from Kraken2
path_krk = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_data_medium_5000ps_ln/kraken2_5000ps_val_simulated_output.txt'
#path_krk = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_small/kraken2_val_simulated_output.txt'
krk_out = file_readers.load_kraken2_output(path_krk)
krk_preds = krk_out['taxid']
print(type(krk_preds))
print(krk_preds)
#### fastq with simulated reads
path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_data_medium_5000ps_ln/SRR25456942_5000ps_val_simulated.fastq'
# path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/SRR25456942_D2_20000ps_val_simulated.fastq'
#path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_small/SRR25456942_val_simulated.fastq' # small data, 
# path = '/gpfs/commons/home/mgarbulowski/proj_shm/old_files/simulated_R2_150.fastq'
# read input fastq file, ending with "_matchBarcode.fastq"
heads, reads, quals = file_readers.fastq(path)
print(type(reads))
print("number of reads:")
print(len(heads))

#### dl model
input_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs'
input_model = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/models_old/spemic62m/model_LSTM.keras'
#input_model = 'model_500000_totreads_11epochs_50batches_.keras'
model = keras.models.load_model(os.path.join(input_path, input_model))

# summarize model
# model.summary()

#### encoder
input_encoder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/models_old/spemic62m/train_encoder_LSTM.h5'
#input_encoder = 'val_encoder_500000_totreads_11epochs_50batches_.h5'
encoder = pickle.load(open(os.path.join(input_path, input_encoder), 'rb'))

#### taxa info, fastq headers with species labels
# this is some unspecified pickle file that contains all the data
path_head = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_data_medium_5000ps_ln/SRR25456942_5000ps_val_genus_species.txt'
#path_head = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/SRR25456942_D2_20000ps_val_genus_species.txt'
#path_head = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_small/SRR25456942_val_genus_species.txt' # small data
#path_head = '/gpfs/commons/home/mgarbulowski/proj_shm/old_files/header_taxa.txt' # old data

krk_preds = krk_out['taxid']
breakpoint()
fastq_spot_d, info = file_readers.make_benchmark_table(path_head, reads, krk_preds, bcodes)
print(fastq_spot_d)

print("info:")
print(type(info))
print(info)


# kraken 2 report file, fastq header - taxa text file, fastq file
# dl_evaluation.taxa_per_spots(report_file, ft_file, fastq_file)

########### Kraken2 + DL reassignment #############
print(type(model))
print(type(encoder))
cluster_l, reassign_d = dl_evaluation.reassign_classes_per_spot(info, model, encoder)


print(type(cluster_l))

print(type(reassign_d))

taxa_orders = ["species", "genus", "family", "order", "class", "phylum", "superkingdom"]

pearson_d, braycurtis_d, accuracy_d, precision_d, recall_d = dl_evaluation.merge_prediction_results(info, cluster_l, fastq_spot_d, taxa_orders, reassign_d)
print(type(pearson_d))

# Calculate average pearson per taxa level
#path_figs = '/gpfs/commons/home/mgarbulowski/proj_shm/figs/'
#make_plots.bench_barplot(pearson_d, taxa_orders, path_figs, mtype="pearson")
#make_plots.bench_barplot(accuracy_d, taxa_orders, path_figs, mtype="accuracy")
#make_plots.bench_barplot(precision_d, taxa_orders, path_figs, mtype="precision")
#make_plots.bench_barplot(recall_d, taxa_orders, path_figs, mtype="recall")

#make_plots.bray_curtis_bench(braycurtis_d, taxa_orders, path_figs)

stats_d = dl_evaluation.per_spot_stats(info, reassign_d, fastq_spot_d, taxa_orders)
print(type(stats_d))
#path_csv = '/gpfs/commons/home/mgarbulowski/proj_shm/figs/report_stats_small_val_data.csv'

#stats_ddf.to_csv(path_csv)

#print(stats_d['10x10'])                    
#print(stats_d['27x8'])





