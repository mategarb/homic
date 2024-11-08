#!/usr/bin/env python

"""
Script to simulate R2 (reads) by taking FASTQ headers, strand and quality score from a real R2 but replacing the sequence with a part of a 16S gene (?).
This simulated R2 (16S reads) is then used as a (simulated) FASTQ file. 
Beyond simulated R2 FASTQ file, it outputs a .txt file with the names (genus + species) collected from 16S gene. This txt is created to have the truth of what species are in the simulated FASTQ file.

From which species the 16S gene is taken from is chosen randomly based on a list of species for which there are 16S gene reference fasta available. The following has to be performed before running this script:
1. Select 16S gene fasta sequences (here: 16SMicrobial.fasta from NCBI) based on a list with interesting species. 
	--> use script prepare_custom_16S_ref_from_species_list.py
2. Run blastn on command line to align the 16S surface probe to these fasta references --> get a list of the E-values of the best alignments. 
3. Run this script to get a simluated R2 with 'fake' read 2 based on where the 16S surface probe aligned in the gene.  
	The newly created 'fake' R2 will contain the same number of FASTQ header and quality score as the original R2 which it is made from. 

Run script as:
	python prepare_sim_16S_read2.py outputs/16S_rr.fa outputs/alignment.txt inputs/SRR25456944_C2_2.fastq outputs SRR25456944_C2
    
    python prepare_sim_16S_read2.py outputs/16S_rr.fa outputs/alignment.txt inputs/SRR25456942_D2_2.fastq outputs SRR25456942_D2

Arguments:
1. FASTA with all 16S references
2. alignment result from blastn
"""
import os
import pandas as pd
import sys
import random
import numpy as np
import gzip
import shutil
import time
#import multiprocessing
#import joblib
#import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from colour import Color
import statistics as stats

#fragment_length = 500 # Estimated fragment size of the 16S gene captured on the slide surface
#R2_length = 300 # Max R2 length
n_reads = 10000

#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# Data loading
# Read the fasta file with all the species
# create dictionary with species name and two elements: sequence header and sequence

startt = time.time()
print("Loading FASTA with species")

fasta_file = sys.argv[1] #'outputs/16S_rr.fa'
fasta_dict = {}

with open(fasta_file) as fp:
    fasta = fp.readlines()
fp.close()

species = 'tmp'

for line in fasta:
    # Get species name
    if line.startswith('>'): # find sequence header
        if len(species.split(' ')) == 2: # after 1st species
            fasta_dict[species] = [whole_header, ''.join(new_seq)] # this is main output from this code
            whole_header = line.rstrip()
            species = ' '.join([line.split(' ')[1], line.split(' ')[2]]) # species name only
            new_seq = []
        else:
            whole_header = line.rstrip() # 1st species
            species = ' '.join([line.split(' ')[1], line.split(' ')[2]]) # species name only
            new_seq = []            
    # Get sequence
    else: # find sequence
        new_seq.append(line.rstrip()) # whitspace removal and append of all lines with sequences

# fasta_dict is a dictionary of 2 element lists (header + sequence)

#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# Alignment statistics
# Snippet to find the best query (16S_surface_probe) alignment in fasta reference and collect the fasta seq upstream of that alignment.

# The e-value alignment file was created using blastn command line. 
# (q) = query
# (s) = subject, ie fasta reference
print("Building dictionaries with sequences for species")
# read e-value alignment file from blastn command line
eval_file = sys.argv[2]

evals = pd.read_csv(eval_file, sep = '\t', comment='#', header=None, names = ['qacc', 'sacc', 'evalue', 'qstart', 'qend', 'sstart', 'send'])

# Get the sacc in a list 
sacc_list = list(set(evals['sacc'].tolist()))

# Get the start position in the fasta ref of the query alignment for each sacc
sacc_startpos_dict = {}

for sacc in sacc_list:
    # Sort to get the lowest evalue per sacc (subject accession) - but why?
    for index, row in evals.sort_values(['evalue']).groupby('sacc').head().iterrows():
        if sacc == row['sacc']:
            sacc_startpos_dict[sacc] = int(row['sstart']) 
            # To include the 16S probe sequence which is on the surface

# below is the old part of code that was sorting by evalues, but now all evalues are the same
# sort by evalues to get the lowest per sacc
# evals = evals.sort_values(by="evalue")
# sacc_startpos_dict = evals.set_index('sacc')["sstart"].to_dict()

# Create a dictionary with species ID as key and position in the fasta reference where the surface probes align
sp_read_length_dict = {}

for seq_acc,start_pos in sacc_startpos_dict.items():   
    for species_name,header_sequence in fasta_dict.items():
        # find where is sacc and select it
        if seq_acc == header_sequence[0][header_sequence[0].find('NR_'):(header_sequence[0].find('NR_') + 11)]: 
            # select sequence for a defined length based on the position where probe aligned 
            # they are of equal length defined in fragment_length
            sp_read_length_dict[species_name] = header_sequence[1][0:(start_pos-1)] # [start_pos-1-fragment_length:start_pos-1] entire genome now 

#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# Reading real data
# Replace the sequence in an R2 fastq file from a real run with these sequences to create a fake fastq file
print("Loading real data")


real_sample_R2_file = sys.argv[3]
exten = real_sample_R2_file[-3:]
# Gunzip input files
if exten == ".gz":
    decompressed_real_sample_R2_file = real_sample_R2_file[:-3]
    # Create decompressed file
    with gzip.open(real_sample_R2_file, 'rb') as f_in:
        with open(decompressed_real_sample_R2_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
else:
    decompressed_real_sample_R2_file = real_sample_R2_file

# Open and read R2 file 
with open(decompressed_real_sample_R2_file) as r2:
    r2_lines = r2.readlines()

# Select all FASTQ headers
r2_header_lines = r2_lines[0::4]
# Select all reads
r2_read_lines = r2_lines[1::4]
# Select all quality lines
r2_qual_lines = r2_lines[3::4]


if exten == ".gz":
    os.remove(decompressed_real_sample_R2_file)

#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# Creating simulated data
# Building a simulated R2
## For each FASTQ header, replace the sequence with randomly picked 16S gene (sequence?)
## R2 on this sequence also starts randomly on the sequence (this is dependent on the fragment length)
print("Creating simulated data")
random_species_list = [] # To store what genus+species got picked

output_folder = sys.argv[4] # collect name of the folder
name = sys.argv[5] # collect name of the file

r2_output_file_name = os.path.join(output_folder, name + '_simulated.fastq')
start_vec = []
qualities_150 = [] 
#def create_simulated_data(i):
with open(r2_output_file_name, 'a+') as f:
    for i, header in enumerate(r2_header_lines):
        # Randomly pick (species) sequence from the 16S fragment from the dictionary
        random_key = random.choice(list(sp_read_length_dict)) # key = species name
        random_sequence = sp_read_length_dict[random_key] # select nuc. sequence
        # Store the genus+species which was randomly selected
        random_species_list.append(header.rstrip()+"|"+random_key) # it will be used as gold standard reference
        R2_length = len(r2_read_lines[i])
        # randomly select where the R2 read is gonna start from a normal distibuiton
        start = len(random_sequence) - int(np.random.randint(R2_length, len(random_sequence) - R2_length)) # total length - random value
        start_vec.append(start/(len(random_sequence) - R2_length))
        if R2_length == 151: # collect statistics for reads of length 150, just for plot
            qualities_150.append(list(map(ord, list(r2_qual_lines[i]))))
        random_sequence_chopped = random_sequence[start:start + R2_length]
        qual_seq = r2_qual_lines[i].rstrip()

        
        # save to output file
        print(header.rstrip(), file = f) # fastq header
        print(random_sequence_chopped, file = f) # randomly picked sequence from part of the 16S gene
        print('+', file = f) # strand
        print(qual_seq, file = f) # quality sequence
        if i == n_reads: # stop if N reads are reached
            break

# Print the genus+species list to output file

gsp_output_file_name = os.path.join(output_folder, name + '_genus_species.txt')

with open(gsp_output_file_name, 'a+') as f:
    for item in random_species_list:
        print(item, file = f)

#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# Plotting the statistics for reads

lns = [len(sub) if type(sub) != int else 1 for sub in r2_read_lines]
lns_all = [len(sub) if type(sub) != int else 1 for sub in sp_read_length_dict.values()]
qualities_150_ar = np.array(qualities_150)

fig = plt.figure(figsize=(15, 10))

gs = gridspec.GridSpec(2,2)

ax = plt.subplot(gs[0, 0])
plt.hist(lns, color = "skyblue", ec="deepskyblue", bins = 20)
plt.text(-0.05, 1.1, "A", fontsize=16, fontweight="bold", transform=ax.transAxes)
plt.ylabel("Frequency")
plt.xlabel("Read length [bp]")

ax = plt.subplot(gs[0, 1])
plt.hist(start_vec, color = "mediumseagreen", ec="seagreen", bins = 100)
plt.text(-0.05, 1.1, "B", fontsize=16, fontweight="bold", transform=ax.transAxes)
plt.ylabel("Frequency")
plt.xlabel("Normalized read start pos. [bp]")

ax = plt.subplot(gs[1, :])
bplot = plt.boxplot(qualities_150_ar, patch_artist=True)
plt.text(-0.05, 1.1, "C", fontsize=16, fontweight="bold", transform=ax.transAxes)
plt.ylabel("Quality score")
plt.xlabel("Read pos. for 150bp")
plt.xticks(color='w')

colors = sns.color_palette("cubehelix", 151)

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

    
fig.tight_layout()

# Display the plot
plt.savefig('figs/simulated_D2_overview.png')
        
print("Done!")
print("Average length: ")
print(stats.mean(lns))
#print(n_reads)
#print(lns_all)
coverage_data = (stats.mean(lns)*n_reads)/sum(lns_all) # Lander-Waterman L*N/G
print("Simulated data coverage: ")
print(coverage_data)

endt = time.time()
print("Time: ")
print(endt - startt)
