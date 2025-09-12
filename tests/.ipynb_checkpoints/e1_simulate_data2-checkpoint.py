import time
import sys
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src')
from homic import file_readers, simulate_16S, dl_model, make_plots
import numpy as np

# static parameters
nsp = 10000 # number of reads per species

#species_tra = None # species_list based on training data to match the same species, cane be left None
er = 0.001
ew = (1, 2, 1)
tr = [0.01, 0.05] # range to truncate read from both ends (percentage), e.g. [0.05,0.1] will draw two values of percentage 

vec_nr = [35000, 70000, 140000, 280000, 420000, 550000, 700000] # cov around 25000, for 125, take 35000

path2 = "/gpfs/commons/home/mgarbulowski/proj_shm/inputs/raw_data/SRR25456942_D2_2.fastq" # fastq 1
path3 = "/gpfs/commons/home/mgarbulowski/proj_shm/inputs/raw_data/SRR25456944_C2_2.fastq" # fastq 2

#####################################################################################################
#####################################################################################################
################################# 125

path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/references_3prime/3prime_65sp_125_comGen.txt"
# outs
output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_125_cg'
output_path = output_path_folder + "/SRR25456942_3prime"
figs_path = output_path_folder + "/figs"


mic_refs = file_readers.fasta(path1)

r2_header_lines, r2_read_lines, r2_qual_lines = file_readers.fastq(path2)



starts, quals, species_list = simulate_16S.training_data_fast(vec_nr[0], # total number of reads
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
breakpoint()

#####################################################################################################
#####################################################################################################
################################# 250

path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/references_3prime/3prime_65sp_250_comGen.txt"
# outs
output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_250_cg'
output_path = output_path_folder + "/SRR25456942_3prime"
figs_path = output_path_folder + "/figs"


mic_refs = file_readers.fasta(path1)

r2_header_lines, r2_read_lines, r2_qual_lines = file_readers.fastq(path2)



starts, quals, species_list = simulate_16S.training_data_fast(vec_nr[1], # total number of reads
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

#####################################################################################################
#####################################################################################################
################################# 500

path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/references_3prime/3prime_65sp_500_comGen.txt"
# outs
output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_500_cg'
output_path = output_path_folder + "/SRR25456942_3prime"
figs_path = output_path_folder + "/figs"


mic_refs = file_readers.fasta(path1)

r2_header_lines, r2_read_lines, r2_qual_lines = file_readers.fastq(path2)



starts, quals, species_list = simulate_16S.training_data_fast(vec_nr[2], # total number of reads
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

#####################################################################################################
#####################################################################################################
################################# 1000

path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/references_3prime/3prime_65sp_1000_comGen.txt"
# outs
output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_1000_cg'
output_path = output_path_folder + "/SRR25456942_3prime"
figs_path = output_path_folder + "/figs"


mic_refs = file_readers.fasta(path1)

r2_header_lines, r2_read_lines, r2_qual_lines = file_readers.fastq(path2)


starts, quals, species_list = simulate_16S.training_data_fast(vec_nr[3], # total number of reads
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

#####################################################################################################
#####################################################################################################
################################# 1500

path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/references_3prime/3prime_65sp_1500_comGen.txt"
# outs
output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_1500_cg'
output_path = output_path_folder + "/SRR25456942_3prime"
figs_path = output_path_folder + "/figs"


mic_refs = file_readers.fasta(path1)

r2_header_lines, r2_read_lines, r2_qual_lines = file_readers.fastq(path2)



starts, quals, species_list = simulate_16S.training_data_fast(vec_nr[4], # total number of reads
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

#####################################################################################################
#####################################################################################################
################################# 2000

path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/references_3prime/3prime_65sp_2000_comGen.txt"
# outs
output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_2000_cg'
output_path = output_path_folder + "/SRR25456942_3prime"
figs_path = output_path_folder + "/figs"


mic_refs = file_readers.fasta(path1)

r2_header_lines, r2_read_lines, r2_qual_lines = file_readers.fastq(path2)


starts, quals, species_list = simulate_16S.training_data_fast(vec_nr[5], # total number of reads
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

#####################################################################################################
#####################################################################################################
################################# 2500

path1 = "/gpfs/commons/home/mgarbulowski/proj_shm/references_3prime/3prime_65sp_2500_comGen.txt"
# outs
output_path_folder = '/gpfs/commons/home/mgarbulowski/proj_shm/outputs/training_data_3prime_2500_cg'
output_path = output_path_folder + "/SRR25456942_3prime"
figs_path = output_path_folder + "/figs"


mic_refs = file_readers.fasta(path1)

r2_header_lines, r2_read_lines, r2_qual_lines = file_readers.fastq(path2)

starts, quals, species_list = simulate_16S.training_data_fast(vec_nr[6], # total number of reads
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

