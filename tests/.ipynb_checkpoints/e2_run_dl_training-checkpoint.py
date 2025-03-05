import time
import sys
import tensorflow as tf
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src')
from homic import file_readers, simulate_16S, dl_model, make_plots
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
#output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_asf_16S_400k' # training_data_preselected+noisy_1mln
#output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_spf_preselected+noisy_1mln_v2'
output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_asf+spf_16S'

#output_path = output_path_folder + "/SRR25456942_asf"
#output_path = output_path_folder + "/SRR25456942_spf"
output_path = output_path_folder + "/SRR25456942_spf_asf"

figs_path = "/gpfs/commons/home/mgarbulowski/proj_shm/figs"

nsp = 10000
# validation
species_path_val = output_path + "_" + str(nsp) + "ps_val_genus_species.txt"
sim_data_val = output_path + "_" + str(nsp) + "ps_val_simulated.fastq"

# training data 1
#output_path_e = output_path_folder + "/training_data_asf_16S/SRR25456942_asf_er"
species_path_tra_e = output_path + "_er_tra_genus_species.txt"
sim_data_tra_e = output_path + "_er_tra_simulated.fastq"

# training data 2
#output_path = output_path_folder + "/training_data_asf_16S/SRR25456942_asf"
species_path_tra = output_path + "_tra_genus_species.txt"
sim_data_tra = output_path + "_tra_simulated.fastq"

#################################################################################
################################# Prepare data

# validation
ts = False # skip taxa info, just labels
species_info_val = file_readers.species_outcome(species_path_val)
prep_data_val = dl_model.prepare_data(sim_data_val, species_info_val, ts)
oh_encoder_val = dl_model.one_hot_encoder(prep_data_val)

# deep learning model and training 1
species_info_tra = file_readers.species_outcome(species_path_tra)
species_info_tra_e = file_readers.species_outcome(species_path_tra_e)

prep_data_tra = dl_model.prepare_data(sim_data_tra, species_info_tra, ts)
prep_data_tra_e = dl_model.prepare_data(sim_data_tra_e, species_info_tra_e, ts)

prep_data_tra_dfs = [prep_data_tra, prep_data_tra_e]
prep_data_tra_all = pd.concat(prep_data_tra_dfs, ignore_index=True)
oh_encoder = dl_model.one_hot_encoder(prep_data_tra_all)
#oh_encoder_tra_e = dl_model.one_hot_encoder(prep_data_tra_e)
# deep learning model and training 2





# merging both training data sets, no error (0.5 mln) + error (0.5 mln)
# new_keys = species_info_tra_e.keys().tolist()
# new_keys = map(( lambda x: x + 'e'), new_keys)
# species_info_tra_e = dict(zip(new_keys, list(species_info_tra_e.values())))

# species_info_tra_all = {**species_info_tra, **species_info_tra_e}
# print(species_info_tra_all)




#oh_encoder_tra_all = prep_data_tra_all["one_hot_tensor"] #np.concatenate((oh_encoder_tra, oh_encoder_tra_e))

print(oh_encoder.shape)
#print(oh_encoder_tra.shape)
#print(oh_encoder_tra_e.shape)
#oh_encoder_tra_all = oh_encoder_tra + oh_encoder_tra_e #np.concatenate((oh_encoder_tra, oh_encoder_tra_e), axis=0)

print("All files processed and prepared. Initializing training...")

# DL training setup
epochs = 11
batches = 50
rank = "none"
oh_model = dl_model.one_hot_model(prep_data_tra_all, prep_data_val,
                                  oh_encoder, oh_encoder_val,
                                  epochs, batches, rank, output_path_folder)

oh_model.classifier.summary()

startt = time.time()
history = oh_model.train2()
endt = time.time()
print("Time of training[min.]: ")
print((endt - startt)/60)


make_plots.metrics(history, figs_path)
dl_model.evaluation(oh_model, figs_path)
