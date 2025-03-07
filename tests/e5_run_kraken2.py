import sys
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src') # to load our package
from homic import file_readers, dl_evaluation, make_plots, dl_model, kraken2

import pickle
import os
import time

import io
import re
#import tensorflow.keras as keras
#from tensorflow.keras import Model

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


#####################################################################################################
################################# Creating DB for kraken2 - done once per DB

#db_path = "/gpfs/commons/home/mgarbulowski/custom_db_test"
#ref_path = "/gpfs/commons/home/mgarbulowski/proj_shm/outputs/references/16S_rr_63species_8ASF_mm.fa"
#kraken2.prepare_db(db_path, ref_path)

#####################################################################################################
################################# Running Kraken2 classification

db_path = '/gpfs/commons/home/mgarbulowski/custom_bacterial_db'
data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/inputs/asf_data/SRR25456916.fastq'
startt = time.time()
out = kraken2.classify(db_path = db_path,
                input_file = data_path)
endt = time.time()
print("Time of Kraken2 classification: ")
print((endt - startt)/60)

out.to_csv('/gpfs/commons/home/mgarbulowski/proj_shm/inputs/asf_data/kraken2_classes_SRR25456916.csv', index=False)

#kraken2.classify(, threads=2)