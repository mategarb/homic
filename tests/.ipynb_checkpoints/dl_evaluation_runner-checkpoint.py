import pickle
import os
import time
import sys
import tensorflow.keras as keras
sys.path.append('/gpfs/commons/home/mgarbulowski/shm_package/src') # to load our package
from homics import file_readers, dl_evaluation
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

#### output from Kraken2
#path_krk = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/kraken2_5000ps_val_simulated_output.txt'
path_krk = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_small/kraken2_val_simulated_output.txt'
krk_out = file_readers.load_kraken2_output(path_krk)

#### fastq with simulated reads
#path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/SRR25456942_5000ps_val_simulated.fastq'
# path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/SRR25456942_D2_20000ps_val_simulated.fastq'
path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_small/SRR25456942_val_simulated.fastq' # small data, 
# path = '/gpfs/commons/home/mgarbulowski/proj_shm/old_files/simulated_R2_150.fastq'
# read input fastq file, ending with "_matchBarcode.fastq"
heads, reads, quals = file_readers.fastq(path)

print("number of reads:")
print(len(heads))

#### dl model
input_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs'
input_model = 'model_500000_totreads_11epochs_50batches_.keras'
#input_model = 'model_500000_totreads_11epochs_50batches_.keras'
model = keras.models.load_model(os.path.join(input_path, input_model))

# summarize model
# model.summary()

#### encoder
input_encoder = 'tra_encoder_500000_totreads_11epochs_50batches_.h5'
#input_encoder = 'val_encoder_500000_totreads_11epochs_50batches_.h5'
encoder = pickle.load(open(os.path.join(input_path, input_encoder), 'rb'))

#### taxa info, fastq headers with species labels
# this is some unspecified pickle file that contains all the data
#path_head = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/SRR25456942_5000ps_val_genus_species.txt'
#path_head = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/SRR25456942_D2_20000ps_val_genus_species.txt'
path_head = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/validation_small/SRR25456942_val_genus_species.txt' # small data
#path_head = '/gpfs/commons/home/mgarbulowski/proj_shm/old_files/header_taxa.txt' # old data

taxa_info_path = "/gpfs/commons/home/mgarbulowski/proj_shm/inputs/taxa_names_dict.pkl"
krk_preds = krk_out['taxid']

fastq_spot_d, info = file_readers.make_benchmark_table(path_head, taxa_info_path, reads, krk_preds, bcodes)

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
#'Bx' - barcode position X, spot definition 
#'By' - barcode position Y, spot definition 

#tmp_chjeck = info['taxa_order'].str.split(",", expand=True)
#print(tmp_chjeck)
#
#print(set(tmp_chjeck.iloc[:, 0]))
#print(set(info['phylum']))

#print(set(tmp_chjeck.iloc[:, 1]))
#print(set(info['class']))

#print(set(tmp_chjeck.iloc[:, 2]))
#print(set(info['order']))

#print(set(tmp_chjeck.iloc[:, 3]))
#print(set(info['family']))

#print(set(tmp_chjeck.iloc[:, 3]))
#print(set(tmp_chjeck.iloc[:, 4]))
#breakpoint()

# kraken 2 report file, fastq header - taxa text file, fastq file
# dl_evaluation.taxa_per_spots(report_file, ft_file, fastq_file)

########### Kraken2 + DL reassignment #############
info['phylum'] = info['phylum'].replace(['Bacillota'],'Firmicutes')


cluster_l, reassign_D = dl_evaluation.reassign_classes_per_spot(info, model, encoder)

taxa_orders = ["species", "genus", "family", "order", "class", "phylum", "superkingdom"]

pearson_d = dl_evaluation.merge_prediction_results(info, cluster_l, fastq_spot_d, taxa_orders)
# Calculate average pearson per taxa level

p_stats = {}
for itp, tp in enumerate(['DL model', 'Kraken2']):
    pl=[]
    cl=[]
    ol=[]
    fl=[]
    gl=[]
    l = [pl, cl, ol, fl, gl]


    for k,v in pearson_d.items():

        for i, tx in enumerate(taxa_orders[1:-1]):
            try:
                l[i].append(v[tx][itp])
            except KeyError:
                continue
    mean_l = []
    for i, il in enumerate(l):
        mean_l.append(round(sum(il) / len(il), 2))
    p_stats[tp] = mean_l

        
p_stats['taxa_order'] = taxa_orders[1:-1]   
p_df = pd.DataFrame(p_stats)
p_df= pd.melt(p_df, id_vars=['taxa_order'], value_vars=['DL model', 'Kraken2'])

   
P_stats = p_df

#P_stats = pd.concat(P_stats)
P_stats.reset_index(inplace=True)

# Plot
# for a 2x3 subplot
axes = []
for row_idx in range(0,1):
    for col_idx in range(0,1):
        ax = plt.subplot2grid((1, 1), (row_idx, col_idx))
        axes.append(ax)
ax=axes[0]
sns.lineplot(data=P_stats, x='taxa_order', y='value', hue='variable', ax=ax)

ax.set_ylabel("Average Pearson correlation coefficient")
ax.set_xlabel("")

# Legend
L=ax.legend()
L.get_texts()[0].set_text('Kraken2 + DL model')
L.get_texts()[1].set_text('Kraken2')


fname = time.strftime("%Y%m%d-%H%M%S")

plt.savefig('/gpfs/commons/home/mgarbulowski/proj_shm/figs/' + fname + '_pearsons.png')
