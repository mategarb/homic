import time
import sys
sys.path.append('/gpfs/commons/home/mgarbulowski/shm_package/src')
from homics import file_readers, simulate_16S, dl_model, make_plots
import numpy as np


#####################################################################################################
#####################################################################################################
################################# I/O paths and sequences

# inps
path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/outputs/16S_rr.fa" # fasta
path2 = "/gpfs/commons/home/mgarbulowski/proj_shm/inputs/SRR25456942_D2_2.fastq" # fastq 1
path3 = "/gpfs/commons/home/mgarbulowski/proj_shm/inputs/SRR25456944_C2_2.fastq" # fastq 2
surface_probe = "GGATTAGATACCCBDGTAGTCGT"

taxa_info_path = "/gpfs/commons/home/mgarbulowski/proj_shm/inputs/taxa_names_dict.pkl"

# outs
output_path = "/gpfs/commons/home/mgarbulowski/proj_shm/outputs/SRR25456942_D2"
output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs'
figs_path = "/gpfs/commons/home/mgarbulowski/proj_shm/figs"
#####################################################################################################
#####################################################################################################
################################# Data simulation


# read references for the microbiome
mic_refs = file_readers.fasta(path1)

[mic_refs, scores_vec] = simulate_16S.prune_references(surface_probe, mic_refs) # from 0 to the position of aligned probe


# read real SHM-seq fastq
r2_header_lines, r2_read_lines, r2_qual_lines = file_readers.fastq(path2)
print("Total organisms in ref: ", len(mic_refs))

# simulate training data for DL model

# Sequence error rate 
errorR = 0.001
species_list = list(mic_refs.keys())
species_tra = species_list
simulate_16S.validation_data(20000, output_path, mic_refs, species_tra,
                             r2_header_lines, r2_read_lines, r2_qual_lines, errorR)

# n (20000) per species (not in total) should give aoround 1,2 mln reads in total