import time
import sys
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src')
from homic import file_readers, simulate_16S, dl_model, make_plots
import numpy as np

#####################################################################################################
#####################################################################################################
################################# I/O paths and sequences

# fasta reference inps
#path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/outputs/references/16S_rr_63species_microbiome_only.fa"
#path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/outputs/references/16S_asf_8species-plusMouse.fasta"
#path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/outputs/references/16S_rr_63species_8ASF_mm.fa"
#path01 = "/gpfs/commons/home/mgarbulowski/proj_shm/outputs/references/16S_asf_8species.fasta"

#path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/references_3prime/3prime_65sp_1000.txt"
#path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/references_3prime/3prime_65sp_1000_comGen.txt"
path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/references_3prime/3prime_65sp_1000_comGen.txt"

path2 = "/gpfs/commons/home/mgarbulowski/proj_shm/inputs/raw_data/SRR25456942_D2_2.fastq" # fastq 1
path3 = "/gpfs/commons/home/mgarbulowski/proj_shm/inputs/raw_data/SRR25456944_C2_2.fastq" # fastq 2
#surface_probe = "GGATTAGATACCCBDGTAGTCGT"

# outs
output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_1000_cg'
output_path = output_path_folder + "/SRR25456942_3prime1000"
figs_path = output_path_folder + "/figs"

#####################################################################################################
#####################################################################################################
################################# Data simulation


# read references for the microbiome
#mic_refs1 = file_readers.fasta(path1)
#print(mic_refs1)

#np0 = mic_refs1.keys()
#npa = np.array(list(np0))

# saving species list
#spth = "/gpfs/commons/home/mgarbulowski/proj_shm/outputs/references/keys_spec63.txt"
#with open(spth, "w") as txt_file:
#    for line in npa:
#        txt_file.write("".join(line) + "\n")


#mic_refs2 = file_readers.fasta(path01)
#print(mic_refs2)
#breakpoint()

mic_refs = file_readers.fasta(path1)
#[mic_refs, scores_vec] = simulate_16S.prune_references(surface_probe, mic_refs) # from 0 to the position of aligned probe

#mic_refs = {**mic_refs1, **mic_refs2}

# read real SHM-seq fastq for data simulation
r2_header_lines, r2_read_lines, r2_qual_lines = file_readers.fastq(path2)


# Sequence error rate 
nsp = 10000 # number of reads per species

#species_tra = None # species_list based on training data to match the same species, cane be left None
er = 0.001
ew = (1, 2, 1)
tr = [0.01, 0.05] # range to truncate read from both ends (percentage), e.g. [0.05,0.1] will draw two values of percentage between range, one to truncate from left, other to truncate from right
#print_stats = True, default
#shuffle = True, default


# simulate training data for the DL model
# startd = time.time()

starts, quals, species_list = simulate_16S.training_data_fast(500000, # total number of reads
                                                                      output_path+"_er",
                                                                      mic_refs,
                                                                      r2_header_lines,
                                                                      r2_read_lines,
                                                                      r2_qual_lines,
                                                                      impute_errors = True,
                                                                      trunc_range = [0.1,0.2],
                                                                      print_stats = True)


simulate_16S.validation_data(nsp, # number of reads per genus/species, to create balanced data
                             output_path,
                             mic_refs,
                             r2_header_lines, 
                             r2_read_lines, 
                             r2_qual_lines,
                             error_rate = er,
                             error_weights = ew,
                             trunc_range = tr) 


#algn_scores, starts, quals, species_list = simulate_16S.training_data(400000, # total number of reads
#                                                                      output_path+"_er",
#                                                                      1, # full data, without preselection
#                                                                      mic_refs,
#                                                                      r2_header_lines,
#                                                                      r2_read_lines,
#                                                                      r2_qual_lines,
#                                                                      impute_errors = True,
#                                                                      trunc_range = [0.1,0.2],
#                                                                      print_stats = True)
#athr = round(np.mean(algn_scores), 4) + round(np.std(algn_scores), 4)
#print(athr)#
#
#algn_scores2, starts, quals, species_list = simulate_16S.training_data(100000, # total number of reads
#                                                                      output_path,
#                                                                      athr, # preselection, est. as mean + std
#                                                                      mic_refs,
#                                                                      r2_header_lines,
#                                                                      r2_read_lines,
#                                                                      r2_qual_lines,
#                                                                      impute_errors = False,
#                                                                      trunc_range = [0,0],
#                                                                      print_stats = True)#
#
#athr2 = round(np.mean(algn_scores2), 4) + round(np.std(algn_scores2), 4)
#print(athr2)

#endd = time.time()
#print("Time of data simulation[min.]: ")
#print((endd - startd)/60)


#make_plots.alignment_scores(algn_scores, figs_path)
# make_plots.data_stats(starts, quals, 151, figs_path)

