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
	python prepare_sim_16S_read2.py outputs/16S_rr.fa outputs/alignment.txt inputs/SRR25456944_C2_2.fastq outputs/SRR25456944_C2
    python prepare_sim_16S_read2.py outputs/16S_rr.fa outputs/alignment.txt inputs/SRR25456942_D2_2.fastq outputs/SRR25456942_D2
    
    python simulate_16S_training.py /gpfs/commons/home/mgarbulowski/proj_shm/outputs/16S_rr.fa /gpfs/commons/home/mgarbulowski/proj_shm/inputs/SRR25456944_C2_2.fastq /gpfs/commons/home/mgarbulowski/proj_shm/outputs/SRR25456944_C2
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
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from colour import Color
import statistics as stats

from Bio.Seq import Seq 
from Bio import Align
import collections
#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# Alignment statistics
# Snippet to find the best query (16S_surface_probe) alignment in fasta reference and collect the fasta seq upstream of that alignment.

# The e-value alignment file was created using blastn command line. 
# (q) = query
# (s) = subject, ie fasta reference
#print("Building dictionaries with sequences for species")
# read e-value alignment file from blastn command line
#eval_file = sys.argv[2]

#evals = pd.read_csv(eval_file, sep = '\t', comment='#', header=None, names = ['qacc', 'sacc', 'evalue', 'qstart', 'qend', 'sstart', #'send'])

# Get the sacc in a list 
#sacc_list = list(set(evals['sacc'].tolist()))

# Get the start position in the fasta ref of the query alignment for each sacc
#sacc_startpos_dict = {}

#for sacc in sacc_list:
    # Sort to get the lowest evalue per sacc (subject accession) - but why?
#    for index, row in evals.sort_values(['evalue']).groupby('sacc').head().iterrows():
 #       if sacc == row['sacc']:
 #           sacc_startpos_dict[sacc] = int(row['sstart']) 
            # To include the 16S probe sequence which is on the surface

# below is the old part of code that was sorting by evalues, but now all evalues are the same
# sort by evalues to get the lowest per sacc
# evals = evals.sort_values(by="evalue")
# sacc_startpos_dict = evals.set_index('sacc')["sstart"].to_dict()

# Create a dictionary with species ID as key and position in the fasta reference where the surface probes align


#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# Creating simulated data for training with reads selection
# Building a simulated R2
## For each FASTQ header, replace the sequence with randomly picked 16S gene (sequence?)
## R2 on this sequence also starts randomly on the sequence (this is dependent on the fragment length)

#fragment_length = 500 # Estimated fragment size of the 16S gene captured on the slide surface
#R2_length = 300 # Max R2 length

def training_data(n_reads, output_path, score_thr, mic_refs, r2_header_lines, r2_read_lines, r2_qual_lines):
    
    random_species_list = [] # To store what genus+species got picked
    random_only_species_list = [] # To store what genus+species got picked
    
    output_folder = output_path # collect name of the folder
    fasta_dict = mic_refs
    r2_output_file_name = output_folder + '_tra_simulated.fastq'
    start_vec = []
    all_scores = []
    qualities_150 = [] 

    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1
    aligner.gap_score = -1.5
    
    ii = 0 # so it stops when it reaches n reads with the alignment score < thr
    #def create_simulated_data(i):
    with open(r2_output_file_name, 'a+') as f:
        for key, value in fasta_dict.items():            
            random_key = key #random.choice(list(fasta_dict)) # key = species name
            random_sequence = value[1] # select nuc. sequence for a given species
            
            for i, header in enumerate(r2_header_lines):
                # Randomly pick (species) sequence from the 16S fragment from the dictionary
                # Store the genus+species which was randomly selected
                R2_length = len(r2_read_lines[i]) # define read length based on real data
                # randomly select where the R2 read is gonna start from a normal distibuiton
                start = len(random_sequence) - int(np.random.randint(R2_length, len(random_sequence) - R2_length)) 
                # total length - random value
                start_vec.append(start/(len(random_sequence) - R2_length))

                #if R2_length == 151: # collect statistics for reads of length 150, just for plot
                #    qualities_150.append(list(map(ord, list(r2_qual_lines[i]))))
                random_sequence_chopped = random_sequence[start:start + R2_length]
                qual_seq = r2_qual_lines[i].rstrip()
                seq1 = Seq(random_sequence_chopped)
                for key0, value0 in fasta_dict.items():
                    seq2 = Seq(value0[1])
                    scores_vec = []
                    if key0 not in random_key:
                        tmp_aligns = aligner.score(seq2, seq1)

                        scores_vec.append(tmp_aligns)

                if not not scores_vec:
                    if np.nanmean(scores_vec)/R2_length < score_thr:
                        random_species_list.append(header.rstrip()+"|"+random_key) 
                        # it will be used as gold standard reference
                        random_only_species_list.append(random_key)

                        # save to output file
                        print(header.rstrip(), file = f) # fastq header
                        print(random_sequence_chopped, file = f) # randomly picked sequence from part of the 16S gene
                        print('+', file = f) # strand
                        print(qual_seq, file = f) # quality sequence
                        ii = ii + 1
                        all_scores.append(np.mean(scores_vec)/R2_length)
                        if ii == n_reads: # move to next iteration if N reads are reached
                            break

    gsp_output_file_name = os.path.join(output_folder + '_tra_genus_species.txt')
    with open(gsp_output_file_name, 'a+') as f:
        for item in random_species_list:
            print(item, file = f)
            
    # using Counter to find frequency of elements
    frequency = collections.Counter(random_only_species_list)

    # printing the frequency
    print("Data generated with ",len(random_species_list)," reads")
    print("Following number of species is included in reads: ")
    print(len(dict(frequency)))

        
        
#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# Creating simulated data for validation

# Building a simulated R2
## For each FASTQ header, replace the sequence with randomly picked 16S gene (sequence?)
## R2 on this sequence also starts randomly on the sequence (this is dependent on the fragment length)

def validation_data(n_reads, output_path, mic_refs, r2_header_lines, r2_read_lines, r2_qual_lines):
    random_species_list = [] # To store what genus+species got picked

    r2_output_file_name = os.path.join(output_path + '_val_simulated.fastq')
    start_vec = []

    #def create_simulated_data(i):
    with open(r2_output_file_name, 'a+') as f:
        for key, value in mic_refs.items():            
            random_key = key #random.choice(list(fasta_dict)) # key = species name
            random_sequence = value[1] # select nuc. sequence for a given species
            
            for i, header in enumerate(r2_header_lines):
                # Randomly pick (species) sequence from the 16S fragment from the dictionary
                #random_key = random.choice(list(mic_refs)) # key = species name
                #random_sequence = mic_refs[random_key] # select nuc. sequence
                # Store the genus+species which was randomly selected
                random_species_list.append(header.rstrip()+"|"+random_key) # it will be used as gold standard reference
                R2_length = len(r2_read_lines[i])
                # randomly select where the R2 read is gonna start from a normal distibuiton
                start = len(random_sequence) - int(np.random.randint(R2_length, len(random_sequence) - R2_length)) 
                # total length - random value
                start_vec.append(start/(len(random_sequence) - R2_length))

                random_sequence_chopped = random_sequence[start:start + R2_length]
                qual_seq = r2_qual_lines[i].rstrip()

                # save to output file
                print(header.rstrip(), file = f) # fastq header
                print(random_sequence_chopped, file = f) # randomly picked sequence from part of the 16S gene
                print('+', file = f) # strand
                print(qual_seq, file = f) # quality sequence
                
                r2_header_lines.remove(header)
                if i == n_reads: # stop if N reads are reached
                    break

    # Print the genus+species list to output file

    gsp_output_file_name = os.path.join(output_path + '_val_genus_species.txt')

    with open(gsp_output_file_name, 'a+') as f:
        for item in random_species_list:
            print(item, file = f)

