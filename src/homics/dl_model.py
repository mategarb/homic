#!/usr/bin/env python coding: utf-8

"""
This script is to train the deep learning (DL) model
Simulated reads of length 150bp since it reflects the read length in a real sample
"""

# python dl_model.py outputs/SRR25456942_D2_simulated.fastq outputs/SRR25456942_D2_genus_species.txt
# python dl_model.py outputs/SRR25456944_C2_simulated.fastq outputs/SRR25456944_C2_genus_species.txt

#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# Read all libs


import os
import numpy as np
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import json
import time
import pickle

from itertools import islice,tee
import io
from collections import defaultdict
import re

from collections import Counter
from numpy import argmax

# DL
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import Sequential, Model, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, SimpleRNN, Input, Concatenate, Bidirectional, Masking, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.optimizers import SGD

import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report
from numba import njit, prange


#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# Functions 

# sequencing error
def introducing_seq_error(row, num_errors):
    """
    Introducing a single base error (replacement). 
    """
    #inds = [i for i,_ in enumerate(row) if not row.isspace()] # array of all posible positions
    inds = list(range(len(row)))
    sam = random.sample(inds, num_errors) # sample on which position read should have an error
    lst = list(row)
    for ind in sam:
        letters_to_draw = ["A", "C", "T", "G"]
        if lst[ind] == "A" or lst[ind] == "T" or lst[ind] == "C" or lst[ind] == "G":
            letters_to_draw.remove(lst[ind]) # as we don't want to impute the same base
        letts = iter(random.sample(letters_to_draw, 1))
        lst[ind] = next(letts)
    return "".join(lst)

# DL model + plots
class FragmentClassifier(object):

    def __init__(self, input_dna, input_taxa, test_fraction,
                 epochs, batch_size, n_classes, Max_seq_len, Masking_value, output_model):
        self.input_dna = input_dna
        self.input_taxa = input_taxa
        self.test_fraction = test_fraction
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.max_seq_len = Max_seq_len
        self.masking_value = Masking_value
        self.output_model = output_model
        self.x_train, self.x_test, self.y_train, self.y_test = self.load_and_split()
        
        # Initialize architecture
        self.classifier = None
        self.embedder = None
        self.architecture()
        
    def load_and_split(self):
        dna_reads = self.input_dna
        print('total number of reads in input:', len(dna_reads))
        print('number of classes:', self.input_taxa.shape[1])

        x = np.array(dna_reads)
        y = np.array(self.input_taxa)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_fraction)

        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        print("Training classes:", y_train.sum(axis=0).astype(int))
        print("Test classes:", y_test.sum(axis=0).astype(int))
               
        return x_train, x_test, y_train, y_test

    def architecture(self):
        
        inp = Input(shape=(None , 5))        
        mask = Masking(mask_value=self.masking_value, input_shape=(self.max_seq_len, 5))(inp)
        
        conv5 = Conv1D(filters=64, kernel_size=15, activation='relu', padding="same")(mask) 
        conv7 = Conv1D(filters=64, kernel_size=17, activation='relu', padding="same")(mask) 
        conv11 = Conv1D(filters=64, kernel_size=19, activation='relu', padding="same")(mask)
        conv13 = Conv1D(filters=64, kernel_size=23, activation='relu', padding="same")(mask)
        mrg = Concatenate()([conv5, conv7, conv11, conv13])
        mrg = Dropout(rate=0.5)(mrg)

        # Add 2 bidirectional LSTMs
        x = Bidirectional(LSTM(64, return_sequences=True))(mrg)
        x = Bidirectional(LSTM(64))(x)
        dp1 = Dropout(rate=0.2)(x)
        
        emb1 = Dense(64, activation='relu')(dp1) #16
        dp2 = Dropout(rate=0.1)(emb1)
        emb2 = Dense(32, activation='relu')(dp2) #8
        decision = Dense(self.input_taxa.shape[1], activation='softmax')(emb2)
        
        model = Model(inputs=inp, outputs=decision)
        model.compile(optimizer='Adam', 
                      loss='categorical_crossentropy', 
                      metrics=['categorical_accuracy']) 
                
        ### Point class to models
        self.embedder = emb2 #embedder
        self.classifier = model
        
    def train(self):
        classifier = self.classifier
        embedder = self.embedder
        output_model = self.output_model
        x_train, y_train = self.x_train, self.y_train
        x_test, y_test = self.x_test, self.y_test
        
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []

        # Optimize number of epochs
        earlystopping = callbacks.EarlyStopping(monitor ="loss", 
                                        mode ="min", patience = 3, # 5 in v1.0.0
                                        restore_best_weights = True, verbose=1)
        
        # Define pairs in the training set
        training_set  = list(zip(x_train, y_train))
        
        num_training_loops = self.epochs
        for e in prange(num_training_loops):
            
            ### it is a good idea to shuffle the training set before each training loop
            np.random.shuffle(training_set)
            
            # Compute derivatives for training set once
            for x, y in training_set:
                xt = x.reshape((1, x.shape[0], x.shape[1]))   
                yt = y.reshape((1, y.shape[0]))

                ### Do one update step with one sequence
                ca = classifier.fit(xt, yt, batch_size=self.batch_size, 
                                    epochs=1, verbose=0, validation_data=None, callbacks =[earlystopping]) 
                    
            ### evaluate once per epoch
            ### Computes metrics in the order they are entered in Model() above
            eval_tr = classifier.evaluate(x_train, y_train, verbose=0)
            eval_te = classifier.evaluate(x_test, y_test, verbose=0)

            
            ### Compute once per epoch ####
            ### Store only the last point ####
            train_loss.append(eval_tr[0])
            train_acc.append(eval_tr[1])
            val_loss.append(eval_te[0]) 
            val_acc.append(eval_te[1]) 
            
            ### Report progress
            print("Computed epoch %d/%d, training loss: %.3f, validation loss: %.3f, training acc: %.3f, validation acc: %.3f" % 
                  (e, num_training_loops, train_loss[-1], val_loss[-1], train_acc[-1], val_acc[-1]))
        
        ########### LOSS vs EPOCHS ################
        #fig, ax = plt.subplots()
        #plt.plot(train_loss)
        #plt.plot(val_loss)
        #plt.title('model loss')
        #plt.ylabel('loss')
        #plt.xlabel('epoch')
        #plt.legend(['train', 'validation'])
        #plt.show()
        #fig.savefig(os.path.join(output_fig, 'loss_vs_epochs_'+naming+'.pdf'))
        
        ########### Acc vs EPOCHS ################
        #fig, ax = plt.subplots()
        #plt.plot(train_acc)
        #plt.plot(val_acc)
        #plt.title('model accuracy')
        #plt.ylabel('acc')
        #plt.xlabel('epoch')
        #plt.legend(['train', 'validation'])
        #plt.show()   
        #fig.savefig(os.path.join(output_fig, 'acc_vs_epochs_'+naming+'.pdf'))
        
        ########### Save model and architecture to single file  ###########
        classifier.save(output_model)
        
# Separate plots from DL model

def evaluation(model):
    classifier = model.classifier
    embedder = model.embedder
    x_test, y_test = model.x_test, model.y_test
    n_classes = model.n_classes

    y_pred = classifier.predict(x_test)

    ########### ROC ################

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC 
    fig, ax = plt.subplots()
    ax.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC-AUC')
    #ax.legend(loc="lower right")
    plt.show()
    fig.savefig(os.path.join(output_fig, 'ROC_AUC_'+naming+'.pdf'))

    ########### STATS ################

    test_F1 = sklearn.metrics.f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='micro') 

    acc = np.mean(np.argmax(y_test, axis=1) == np.argmax(y_pred, axis=1))

    with open(os.path.join(output_fig, "output_metrics_"+naming+'.txt'), "a") as f:
        print("F1: " + str(test_F1), file=f)
        print("Accuracy: " + str(acc), file=f)       
        

#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# Data prepration

# Load and prepare simulated data
# Read input fasta file
def prepare_data(input_fq, ref_d, ge_sp_dict):
    fasta_d_tmp = parser(input_fq)

    # Add taxa order for the species names in dict
    ref_df = pd.DataFrame.from_dict(ref_d, orient='index', columns=['taxa'])

    ref_df['taxa_order'] = ref_df['taxa'].map(ge_sp_dict)
    ref_df['taxa_order'] = ref_df['taxa_order'].str.join(',')

    ref_d = ref_df.to_dict('index')
    ### Prepare simulated data ###

    df = pd.DataFrame.from_dict(ref_d, orient='index', columns=['taxa', 'taxa_order'])
    df.reset_index(inplace=True)
    df_sim = df.sample(frac=1).reset_index() # Random shuffle

    df_sim['read'] = df_sim['index'].map(fasta_d_tmp)
    df_sim['read'] = list(filter(str.strip, df_sim['read']))

    merging_cols = ['index', 'taxa_order', 'read'] 
    df_merge = df_sim.loc[:,merging_cols]

    df_merge['taxa_order_GE'] = df_merge['taxa_order'].str.split(',').str[1:]
    df_merge['taxa_order_GE'] = df_merge['taxa_order_GE'].str.join(',') # GE genus?

    return df_merge

# Load and prepare simulated data
# Read input fasta file
def prepare_data_old(input_fq, ref_d, ge_sp_dict, MIN_COUNT, errorR):
    fasta_d_tmp = parser(input_fq)

    # Add taxa order for the species names in dict
    ref_df = pd.DataFrame.from_dict(ref_d, orient='index', columns=['taxa'])

    ref_df['taxa_order'] = ref_df['taxa'].map(ge_sp_dict)
    ref_df['taxa_order'] = ref_df['taxa_order'].str.join(',')

    ref_d = ref_df.to_dict('index')

    ### Prepare simulated data ###

    df = pd.DataFrame.from_dict(ref_d, orient='index', columns=['taxa', 'taxa_order'])
    df.reset_index(inplace=True)
    df = df.sample(frac=1).reset_index(drop=True) # Random shuffle

    df_keep = df.groupby('taxa').head(MIN_COUNT)
    df_keep.reset_index(inplace=True)

    tot_reads = df.shape[0] # all reads number

    # Randomly select the missing values
    num_to_add = tot_reads - df_keep.shape[0] # as some are removed
    print('Num to add:', num_to_add)

    df_add = df[~df['index'].isin(df_keep['index'].tolist())]
    df_add = df_add.sample(n = num_to_add)

    # Merge df_keep and df_add
    df_sim = pd.concat([df_keep, df_add])
    

    # Match taxa with fastq header
    df_sim['read'] = df_sim['index'].map(fasta_d_tmp)
    df_sim.fillna('Unknown', inplace=True)
    
    ### INTRODUCE RANDOM ERRORS  ###

    # Sum read length for all reads
    tot_read_length = df_sim['read'].str.len().sum()
    

    tot_num_errors = int(tot_read_length * errorR)

    if tot_num_errors > df_sim.shape[0]:
        # Randomly select reads which shall include a simulated seq error
        df_subset = df_sim.copy()
        num_errors = int(tot_num_errors/df_sim.shape[0])
    else:
        # Randomly select reads which shall include a simulated seq error, to simplify: one error per read
        df_subset = df_sim.sample(n=tot_num_errors, replace=False)
        num_errors = 1

    df_subset['read'] = df_subset.apply(lambda row: introducing_seq_error(row['read'], num_errors), axis=1)

    # Delete subset from main df
    new_sim = df_sim[~df_sim['index'].isin(df_subset['index'].tolist())]

    df_sim = pd.concat([new_sim, df_subset])

    ## SELECTINGA ##
    merging_cols = ['index', 'taxa_order', 'read'] 

    df_merge = df_sim.loc[:,merging_cols]
    df_merge['taxa_order_GE'] = df_merge['taxa_order'].str.split(',').str[1:]
    df_merge['taxa_order_GE'] = df_merge['taxa_order_GE'].str.join(',')
    
    return df_merge

def one_hot_encoder(df_merge):
    df_merge['one_hot_tensor'] = df_merge.apply(lambda row: dna_encode_embedding_table(row['read']), axis=1)
    X = np.array(df_merge['one_hot_tensor'].tolist(), dtype=object)

    # Padding to the same sequence length
    masking_value = -1
    max_seq_len = max(len(x) for x in df_merge['one_hot_tensor'].tolist())
    N = X.shape[0]
    dimension = 5

    Xpad = np.full((N, max_seq_len, dimension), fill_value=masking_value)
    for s, x in enumerate(X):
        seq_len = x.shape[0]
        Xpad[s, 0:seq_len, :] = x
    return Xpad

def one_hot_model(df_merge, Xpad, epochs, batches, output_path):
    naming = str(df_merge.shape[0])+'_totreads_'+str(epochs)+'epochs_'+str(batches)+'batches_'
    masking_value = -1
    max_seq_len = max(len(x) for x in df_merge['one_hot_tensor'].tolist())
    
    y = df_merge['taxa_order'] #df_merge['taxa_order_GE']

    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = to_categorical(encoded_y)

    # Save Encoder
    # To save DL model/embedder
    output_encoder = output_path + '/encoder_' + naming + '.h5'
    pickle.dump(encoder, open(output_encoder, 'wb'))

    assert set(dummy_y.sum(axis=1)) == {1}
    # Initialize model and printout summary
    # Number of taxa to classify
    n_classes = dummy_y.shape[1]

    model = FragmentClassifier(Xpad, dummy_y, test_fraction=0.2, 
                               epochs=epochs, batch_size=batches,
                               n_classes=n_classes, Max_seq_len=max_seq_len,
                               Masking_value=masking_value, 
                               output_model=output_path + '/model_' + naming + '.h5')
    return model


def taxa_assignment(report):
    """
    Create a dictionary with the taxa order with taxID as key and taxa order as value
    Takes the kraken2 report as input and Returns the dictionary.
    """
    tax_order_dict = {}
    sp_tax_dict = {} # This is an extra dictionary with only species and theri corresponding NCBI taxID

    for index, row in report.iterrows():
        
        # Domain
        if row['torder'] == 'D':
            domain = row['sci-name'].lstrip()
            dvalue = ','.join([domain] + ['Unassigned'] * 6)
            tax_order_dict[row['ncbi taxID']] = dvalue
        # Subdomain
        elif row['torder'] == 'D1':
            tax_order_dict[row['ncbi taxID']] = dvalue

        # Phylum
        elif row['torder'] == 'P':
            phylum = row['sci-name'].lstrip()
            pvalue = ','.join([domain] + [phylum] + ['Unassigned'] * 5)
            tax_order_dict[row['ncbi taxID']] = pvalue

        elif ((row['torder'].startswith('P')) & (len(row['torder']) > 1)): # if x1 or x2 or xn
            tax_order_dict[row['ncbi taxID']] = pvalue

        # Class
        elif row['torder'] == 'C':
            Class = row['sci-name'].lstrip()
            cvalue = ','.join([domain] + [phylum] + [Class] + ['Unassigned'] * 4)
            tax_order_dict[row['ncbi taxID']] = cvalue

        # Order
        elif ((row['torder'] == 'O') | (row['torder'] == 'C1')): 
            order = row['sci-name'].lstrip()
            ovalue = ','.join([domain] + [phylum] + [Class] + [order] + ['Unassigned'] * 3)
            tax_order_dict[row['ncbi taxID']] = ovalue            

        # Family
        elif ((row['torder'] == 'F') | (row['torder'] == 'O1') | (row['torder'] == 'C2')):
            family = row['sci-name'].lstrip()
            fvalue = ','.join([domain] + [phylum] + [Class] + [order] + [family] + ['Unassigned'] * 2)
            tax_order_dict[row['ncbi taxID']] = fvalue

        # Genus
        elif ((row['torder'] == 'G') | (row['torder'] == 'F1') | (row['torder'] == 'O2') | (row['torder'] == 'C3')):
            genus = row['sci-name'].lstrip()
            gvalue = ','.join([domain] + [phylum] + [Class] + [order] + [family] + [genus] + ['Unassigned'] * 1)
            tax_order_dict[row['ncbi taxID']] = gvalue

        # species
        elif ((row['torder'] == 'S') | (row['torder'] == 'G1') | (row['torder'] == 'F2') | (row['torder'] == 'O3') | (row['torder'] == 'C4')):
            specie = row['sci-name'].lstrip()
            svalue = ','.join([domain] + [phylum] + [Class] + [order] + [family] + [genus] + [row['sci-name'].lstrip()])
            tax_order_dict[row['ncbi taxID']] = svalue
            sp_tax_dict[specie] = row['ncbi taxID']
        
        # Subspecies    
        elif row['torder'] == 'S1':
            tax_order_dict[row['ncbi taxID']] = svalue

    return tax_order_dict, sp_tax_dict
   
def tax_order_sp(row, d, d_sp):
    """
    Based on taxID, return the taxa order found in d.
    If not taxID found in d (it is a subspecies), look for the genus+species in the d_sp.
    Returns taxa order.
    """
    if row['taxa_orderTMP'] == 10090:
        return 'Animalia,Chordata,Mammalia,Rodentia,Muridae,Mus,Mus musculus'
    
    elif row['taxa_orderTMP'] == 131567: # Cellular organism
        return ','.join(['Cellular organism'] + ['Unassigned'] * 6)
    
    else:
        try:
            taxorder = d[row['taxa_orderTMP']]
            return taxorder
        except KeyError:
            sp = ' '.join([row['sp_short'][0], row['sp_short'][1]])
            
            try:        
                new_taxid = d_sp[sp]
                return d[new_taxid]
            except KeyError:
                return ','.join(['Unassigned'] * 7)
    
           
def parser(filename):
    """
    Generate a list of tuples (header, read)
    """
    fastq_parsed = {}
    try:

        with open(filename) as fq:
            header = next(fq)
            read = next(fq)
            fastq_parsed[header[1:-1].split(' ')[0]] = read[:-1] # we don't want new-line chars or @ sign in reads
            while True:
                next(fq) # skip + line
                next(fq) # skip qscore
                header = next(fq) # store next read header
                read = next(fq) # store next read seq
                fastq_parsed[header[1:-1].split(' ')[0]] = read[:-1]
    except:
        StopIteration # when we run out of reads, stop
    return fastq_parsed
   
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
    # Stack reads into one tensor
    info_xy['one_hot_tensor'] = info_xy.apply(lambda row: dna_encode_embedding_table(row['read']), axis=1)
    X = np.array(info_xy['one_hot_tensor'].tolist())
    # Padding to the same sequence length
    masking_value = -1
    max_seq_len = max(len(x) for x in info_xy['one_hot_tensor'].tolist())
    N = X.shape[0]
    dimension = 5

    Xpad = np.full((N, max_seq_len, dimension), fill_value=masking_value)
    for s, x in enumerate(X):
        seq_len = x.shape[0]
        Xpad[s, 0:seq_len, :] = x
    return Xpad

def extract_taxa_info(info_xy, column_header, slide_type):
    y_taxaorder = info_xy[column_header].tolist()
    y_fastqH = info_xy.index.tolist()

    if slide_type == 'ST':
        y_umi = info_xy['UMI'].tolist()
        return y_taxaorder, y_fastqH, y_umi
    elif slide_type == 'QC':
        return y_taxaorder, y_fastqH

def predict_taxa_in_model(Xpad, model):
    predictions = model.predict(Xpad)
    return predictions