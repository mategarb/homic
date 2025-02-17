import time
import sys
import tensorflow as tf
sys.path.append('/gpfs/commons/home/mgarbulowski/shm_package/src')
from homics import file_readers, simulate_16S, dl_model, make_plots
import numpy as np
import pandas as pd
#################################################################################
################################# Set up GPU / CUDA

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
#
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

print("GPU set")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#################################################################################
################################# Read files & prepare names

# prepare data for DL training
# outs
output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs'
output_path = output_path_folder + "/SRR25456942"

figs_path = "/gpfs/commons/home/mgarbulowski/proj_shm/figs"

nsp = 5000
# validation
species_path_val = output_path + "_" + str(nsp) + "ps_val_genus_species.txt"
sim_data_val = output_path + "_" + str(nsp) + "ps_val_simulated.fastq"
# training 1
output_path_e = output_path_folder + "/SRR25456942_e"
species_path_tra_e = output_path_e + "_tra_genus_species.txt"
sim_data_tra_e = output_path_e + "_tra_simulated.fastq"

# training 2
output_path = output_path_folder + "/SRR25456942"
species_path_tra = output_path + "_tra_genus_species.txt"
sim_data_tra = output_path + "_tra_simulated.fastq"

#################################################################################
################################# Prepare data

# validation
species_info_val = file_readers.species_outcome(species_path_val)
prep_data_val = dl_model.prepare_data(sim_data_val, species_info_val)
oh_encoder_val = dl_model.one_hot_encoder(prep_data_val)

# deep learning model and training 1
species_info_tra = file_readers.species_outcome(species_path_tra)
prep_data_tra = dl_model.prepare_data(sim_data_tra, species_info_tra)
oh_encoder_tra = dl_model.one_hot_encoder(prep_data_tra)

# deep learning model and training 2
species_info_tra_e = file_readers.species_outcome(species_path_tra_e)
prep_data_tra_e = dl_model.prepare_data(sim_data_tra_e, species_info_tra_e)
oh_encoder_tra_e = dl_model.one_hot_encoder(prep_data_tra_e)

# merging both training data sets, no error (0.5 mln) + error (0.5 mln)
# new_keys = species_info_tra_e.keys().tolist()
# new_keys = map(( lambda x: x + 'e'), new_keys)
# species_info_tra_e = dict(zip(new_keys, list(species_info_tra_e.values())))

# species_info_tra_all = {**species_info_tra, **species_info_tra_e}
# print(species_info_tra_all)
prep_data_tra_dfs = [prep_data_tra, prep_data_tra_e]
prep_data_tra_all = pd.concat(prep_data_tra_dfs, ignore_index=True)

oh_encoder_tra_all = np.concatenate((oh_encoder_tra, oh_encoder_tra_e))

print("All files prcoessed and prepared. Initializing training...")

# DL training setup
epochs = 11
batches = 50
rank = "species"
oh_model = dl_model.one_hot_model(prep_data_tra_all, prep_data_val,
                                  oh_encoder_tra_all, oh_encoder_val,
                                  epochs, batches, rank, output_path_folder)

oh_model.classifier.summary()

startt = time.time()
history = oh_model.train2()
endt = time.time()
print("Time of training[min.]: ")
print((endt - startt)/60)


make_plots.metrics(history, figs_path)
dl_model.evaluation(oh_model, figs_path)
