import time
import sys
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src')
from homic import file_readers, simulate_16S, dl_model, make_plots
import numpy as np

#####################################################################################################
#####################################################################################################
################################# I/O paths and sequences

# inps
path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/outputs/references/16S_rr_62species_microbiome_only.fa" # fasta
path2 = "/gpfs/commons/home/mgarbulowski/proj_shm/inputs/raw_data/SRR25456942_D2_2.fastq" # fastq 1
path3 = "/gpfs/commons/home/mgarbulowski/proj_shm/inputs/raw_data/SRR25456944_C2_2.fastq" # fastq 2
surface_probe = "GGATTAGATACCCBDGTAGTCGT"

# outs
output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs'
output_path = output_path_folder + "/SRR25456942_example"

figs_path = "/gpfs/commons/home/mgarbulowski/proj_shm/figs"
#####################################################################################################
#####################################################################################################
################################# Data simulation


# read references for the microbiome
mic_refs = file_readers.fasta(path1)


[mic_refs, scores_vec] = simulate_16S.prune_references(surface_probe, mic_refs) # from 0 to the position of aligned probe


# read real SHM-seq fastq for data simulation
r2_header_lines, r2_read_lines, r2_qual_lines = file_readers.fastq(path2)

# simulate training data for the DL model
# startd = time.time()

#impute_errors = True
#trunc_range = [0,0]
#print_stats = True
#algn_scores, starts, quals, species_list = simulate_16S.training_data(50, # total number of reads
#                                                                      output_path,
#                                                                      1, # 0.77 est. as mean + std
#                                                                      mic_refs,
#                                                                      r2_header_lines,
#                                                                      r2_read_lines,
#                                                                      r2_qual_lines,
#                                                                      impute_errors,
#                                                                      trunc_range,
#                                                                      print_stats)

#endd = time.time()
#print("Time of data simulation[min.]: ")
#print((endd - startd)/60)


# make_plots.alignment_scores(algn_scores, figs_path)
# make_plots.data_stats(starts, quals, 151, figs_path)

# Sequence error rate 
nsp = 100 #5000 # number of reads per species

#species_tra = None # species_list based on training data to match the same species, cane be left None
er = 0.001
ew = (1, 2, 1)
tr = [0.01, 0.05] # range to truncate read from both ends (percentage), e.g. [0.05,0.1] will draw two values of percentage between range, one to truncate from left, other to truncate from right
#print_stats = True, default
#shuffle = True, default

simulate_16S.validation_data(nsp, # number of reads per genus/species, to create balanced data
                             output_path,
                             mic_refs,
                             r2_header_lines, 
                             r2_read_lines, 
                             r2_qual_lines,
                             error_rate = er,
                             error_weights = ew,
                             trunc_range = tr) 
