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
import ete3
from Bio.Seq import Seq
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

#data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_125_cg/SRR25456942_3prime_er_tra_simulated.fastq'
#outc_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_125_cg/SRR25456942_3prime_er_tra_genus_species.txt'
#
data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_125_cg/SRR25456942_3prime_10000ps_val_simulated.fastq'
outc_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_125_cg/SRR25456942_3prime_10000ps_val_genus_species.txt'
#

startt = time.time()
out = kraken2.classify(db_path = db_path,
                input_file = data_path)
endt = time.time()
print("Time of Kraken2 classification: ")
print((endt - startt)/60)
##
k2file = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/kraken_bench/kraken2_classes_3prime1000_125_2.csv'
out.to_csv(k2file, index=False)

acc = kraken2.evaluate_kraken(k2file, outc_path)
print("Accuracy: ")
print(acc)



##2
data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_250_cg/SRR25456942_3prime_10000ps_val_simulated.fastq'
outc_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_250_cg/SRR25456942_3prime_10000ps_val_genus_species.txt'
#
#startt = time.time()
out = kraken2.classify(db_path = db_path,
                input_file = data_path)
endt = time.time()
print("Time of Kraken2 classification: ")
print((endt - startt)/60)
#
k2file = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/kraken_bench/kraken2_classes_3prime1000_250_2.csv'
out.to_csv(k2file, index=False)

acc = kraken2.evaluate_kraken(k2file, outc_path)
print("Accuracy: ")
print(acc)


##3
data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_500_cg/SRR25456942_3prime_10000ps_val_simulated.fastq'
outc_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_500_cg/SRR25456942_3prime_10000ps_val_genus_species.txt'
#
startt = time.time()
out = kraken2.classify(db_path = db_path,
                input_file = data_path)
endt = time.time()
print("Time of Kraken2 classification: ")
print((endt - startt)/60)
#
k2file = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/kraken_bench/kraken2_classes_3prime1000_500_2.csv'
out.to_csv(k2file, index=False)

acc = kraken2.evaluate_kraken(k2file, outc_path)
print("Accuracy: ")
print(acc)


#4
data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_1000_cg/SRR25456942_3prime_10000ps_val_simulated.fastq'
outc_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_1000_cg/SRR25456942_3prime_10000ps_val_genus_species.txt'
#
startt = time.time()
out = kraken2.classify(db_path = db_path,
                input_file = data_path)
endt = time.time()
print("Time of Kraken2 classification: ")
print((endt - startt)/60)

k2file = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/kraken_bench/kraken2_classes_3prime1000_1000_2.csv'
out.to_csv(k2file, index=False)

acc = kraken2.evaluate_kraken(k2file, outc_path)
print("Accuracy: ")
print(acc)


#5
data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_1500_cg/SRR25456942_3prime_10000ps_val_simulated.fastq'
outc_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_1500_cg/SRR25456942_3prime_10000ps_val_genus_species.txt'
#
startt = time.time()
out = kraken2.classify(db_path = db_path,
                input_file = data_path)
endt = time.time()
print("Time of Kraken2 classification: ")
print((endt - startt)/60)

k2file = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/kraken_bench/kraken2_classes_3prime1000_1500_2.csv'
out.to_csv(k2file, index=False)

acc = kraken2.evaluate_kraken(k2file, outc_path)
print("Accuracy: ")
print(acc)


#6
data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_2000_cg/SRR25456942_3prime_10000ps_val_simulated.fastq'
outc_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_2000_cg/SRR25456942_3prime_10000ps_val_genus_species.txt'
#
startt = time.time()
out = kraken2.classify(db_path = db_path,
                input_file = data_path)
endt = time.time()
print("Time of Kraken2 classification: ")
print((endt - startt)/60)

k2file = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/kraken_bench/kraken2_classes_3prime1000_2000_2.csv'
out.to_csv(k2file, index=False)

acc = kraken2.evaluate_kraken(k2file, outc_path)
print("Accuracy: ")
print(acc)

#7
#data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_2500_cg/SRR25456942_3prime_er_tra_simulated.fastq'
#outc_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_2500_cg/SRR25456942_3prime_er_tra_genus_species.txt'
data_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_2500_cg/SRR25456942_3prime_10000ps_val_simulated.fastq'
outc_path = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_2500_cg/SRR25456942_3prime_10000ps_val_genus_species.txt'
#
startt = time.time()
out = kraken2.classify(db_path = db_path,
                input_file = data_path)
endt = time.time()
print("Time of Kraken2 classification: ")
print((endt - startt)/60)

k2file = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/kraken_bench/kraken2_classes_3prime1000_2500_2.csv'
out.to_csv(k2file, index=False)

acc = kraken2.evaluate_kraken(k2file, outc_path)
print("Accuracy: ")
print(acc)


