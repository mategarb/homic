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
output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs'
output_path = output_path_folder + "/SRR25456942_e"

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


startd = time.time()
# no error
impute_errors = True
algn_scores, starts, quals, species_list  = simulate_16S.training_data(500000, output_path, 1, # 0.77 est. as mean + std 
                           mic_refs, r2_header_lines, r2_read_lines, r2_qual_lines, impute_errors) # n (500000) in total

breakpoint()

print("Mean:")
print(np.mean(algn_scores))
print("Median:")
print(np.median(algn_scores))
print("Std:")
print(np.std(algn_scores))

endd = time.time()
print("Time of data simulation[min.]: ")
print((endd - startd)/60)
make_plots.alignment_scores(algn_scores, figs_path)
make_plots.data_stats(starts, quals, 151, figs_path)

# Sequence error rate 
errorR = 0.001

species_tra = species_list
nsp = 5000
simulate_16S.validation_data(nsp, output_path, mic_refs, species_tra,
                             r2_header_lines, r2_read_lines, r2_qual_lines, errorR) # n e.g. 2000 per species (not in total)
