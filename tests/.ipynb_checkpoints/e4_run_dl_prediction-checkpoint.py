import pickle
import os
import time
import sys
import tensorflow.keras as keras
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src') # to load our package
from homic import file_readers, dl_evaluation, make_plots, dl_model
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
#path = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/asf_data/SRR25456916.fastq'
#path = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/asf_data/SRR25456917.fastq'
#path = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/asf_data/SRR25456918.fastq'
#path = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/raw_data/SRR25456942_D2_2.fastq'
path = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/raw_data/SRR25456944_C2_2.fastq'
#path = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/other_data/SRR25670668.fastq'

heads, reads, quals = file_readers.fastq(path)
del heads, quals

print("Data loaded")
print("Number of reads:")
print(len(reads))

#### dl model
input_path = '/gpfs/commons/home/mgarbulowski/homic_package/models/asfmic_16S'
input_model = 'model.keras'
input_encoder = 'tra_encoder.h5'

# loading model and encoder
model = keras.models.load_model(os.path.join(input_path, input_model))#input_path + "/" + input_model#
encoder = pickle.load(open(os.path.join(input_path, input_encoder), 'rb'))


# initialize data of lists.
data = {'read': reads[:1000000]}

# Create DataFrame
info_xy = pd.DataFrame(data)


startt = time.time()
pred_classes, freq_pred_classes = dl_model.predict_class_for_reads(info_xy, model, encoder, rank="none")
endt = time.time()
print("Time of prediction[min.]: ")
print((endt - startt)/60)

# frequency of species
df = pd.DataFrame.from_dict(freq_pred_classes, orient='index')
df.to_csv('/gpfs/commons/home/mgarbulowski/proj_shm/outputs/predictions/pred_asf16S_SRR25456944.txt', sep='\t')



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




