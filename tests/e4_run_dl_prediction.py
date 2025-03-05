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
from sklearn import metrics

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

#### fastq with simulated reads
#path = '/gpfs/commons/home/mgarbulowski/homic_package/data/SRR25456942_example_16SrRNAseq_data.fastq'
path = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/asf_data/SRR25456916.fastq'
#path = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/asf_data/SRR25456917.fastq'
#path = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/asf_data/SRR25456918.fastq'
#path = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/raw_data/SRR25456944_C2_2.fastq'
#path = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/other_data/SRR25670668.fastq'

heads, reads, quals = file_readers.fastq(path)
del heads, quals

print("Data loaded")
print("number of reads:")
print(len(reads))

#### dl model
input_path = '/gpfs/commons/home/mgarbulowski/homic_package/models/asfmic_16S'
input_model = 'model_LSTM.keras'
input_encoder = 'train_encoder_LSTM.h5'

# loading model and encoder
model = keras.models.load_model(os.path.join(input_path, input_model))
encoder = pickle.load(open(os.path.join(input_path, input_encoder), 'rb'))

#### taxa info, gold standard
#path_head = '/gpfs/commons/home/mgarbulowski/homic_package/data/SRR25456942_example_gold_truth_species.txt'
#info = pd.read_csv(path_head, sep='[ ,:,|]', header = None, usecols=[5, 6, 7, 9, 10], names = ['tile', 'x', 'y','taxa1', 'taxa2'], engine='python')
#info['truth_taxa'] = info['taxa1'] + ' ' + info['taxa2']
#info.loc[info["truth_taxa"] == "[Eubacterium] eligens", "truth_taxa"] = 'Lachnospira eligens' # cause it has different names

# initialize data of lists.
data = {'read': reads}

# Create DataFrame
info_xy = pd.DataFrame(data)


def dna_encode_embedding_table(dna_input, name="dna_encode"):
    """
    DNA embedding.
    """

    embedding_values = np.zeros([len(dna_input), 5], np.float32)
    values = ("A", "C", "G", "T", "N")
    for j, b in enumerate(dna_input):
        if b in values:
            embedding_values[j, values.index(b)] = 1
    return embedding_values
    

def stack_padding(info_xy):
    # Stack reads into one tenso
    
    info_xy['one hot tensor'] = info_xy.apply(lambda row: dna_encode_embedding_table(row['read']), axis=1)
    X = np.array(info_xy['one hot tensor'].tolist(), dtype=object)

    # Padding to the same sequence length
    masking_value = -1
    max_seq_len = max(len(x) for x in info_xy['one hot tensor'].tolist())
    N = X.shape[0]
    dimension = 5

    Xpad = np.full((N, max_seq_len, dimension), fill_value=masking_value)
    for s, x in enumerate(X):
        seq_len = x.shape[0]
        Xpad[s, 0:seq_len, :] = x
        
    return Xpad

Xpad = stack_padding(info_xy) # stacking and padding/masking of reads
    
            
predictions = model.predict(Xpad, verbose = 0) # predict assignments using the model
rv_predictions = encoder.inverse_transform(predictions.argmax(axis=1)) # predict taxa using encoder

data2 = {'y_pred': rv_predictions}

preds = pd.DataFrame(data2)
preds = preds['y_pred'].str.split(',', expand=True)


y_pred = preds.iloc[:,0]
y_pred = y_pred.tolist()

c = Counter(y_pred)

for i in c:
    c[i] = round(float(c[i]/len(y_pred)), 3) * 100

# frequency of species
print(c)
df = pd.DataFrame.from_dict(c, orient='index')
df.to_csv('/gpfs/commons/home/mgarbulowski/proj_shm/inputs/other_data/pred_species_SRR25670668.txt', sep='\t')

y_pred2 = preds.iloc[:,1]
y_pred2 = y_pred2.tolist()

c2 = Counter(y_pred2)

for i in c2:
    c2[i] = round(float(c2[i]/len(y_pred2)), 3) * 100

# frequency of species
print(c2)
df2 = pd.DataFrame.from_dict(c2, orient='index')
df2.to_csv('/gpfs/commons/home/mgarbulowski/proj_shm/inputs/other_data/pred_genus_SRR25670668.txt', sep='\t')


#y_true = info['truth_taxa'].tolist()
#print(classification_report(y_true, y_pred))

#vacc = metrics.accuracy_score(y_true, y_pred)
#print("Accuracy:")
#print(vacc)

#vmi = metrics.mutual_info_score(y_true, y_pred)
#print("MI:")
#print(round(vmi,3))

#vpvv =metrics.precision_score(y_true, y_pred, average=None)
#print("PPV:")
#print(round(np.mean(vpvv),3))




