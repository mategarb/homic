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

def prune_references(probe_seq, mic_ref_seqs, thr=0.25):
    
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1
    aligner.gap_score = -1
    
    scores_vec = []
    seq1 = Seq(probe_seq)
    for key, value in mic_ref_seqs.items():
        seq2 = Seq(value[1])
        tmp_aligns = aligner.align(seq2, seq1)
        alignment = tmp_aligns[0]
        scores_vec.append(alignment.score)
        coords_df = pd.DataFrame(np.concatenate(alignment.coordinates))
        mic_ref_seqs[key] = value[1][0:coords_df.iloc[0,0]]
        if len(mic_ref_seqs[key])/len(value[1]) < thr: # when most of the sequence is removed 
            # there is not enough length to draw a read
            mic_ref_seqs[key] = value[1] # in such case a full reference is taken
    return mic_ref_seqs, scores_vec

def training_data(n_reads, output_path, score_thr, mic_refs, r2_header_lines, r2_read_lines, r2_qual_lines, impute_errors, trunc_range, print_stats):
    
    random_species_list = [] # To store what genus+species got picked
    random_only_species_list = [] # To store what genus+species got picked
    
    output_folder = output_path # collect name of the folder
    fasta_dict = mic_refs
    r2_output_file_name = output_folder + '_tra_simulated.fastq'
    start_vec = []
    all_scores = []
    qual_vec = [] 
    seq_lns = []
    
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1
    aligner.gap_score = -1.5
    
    ii = 0 # so it stops when it reaches n reads with the alignment score < thr
    #def create_simulated_data(i):
    with open(r2_output_file_name, 'a+') as f:
        #for key, value in fasta_dict.items():            
            #random_key = key #random.choice(list(fasta_dict)) # key = species name
            #random_sequence = value # select nuc. sequence for a given species
            
            for i, header in enumerate(r2_header_lines):
                # Randomly pick (species) sequence from the 16S fragment from the dictionary
                random_key = random.choice(list(fasta_dict.keys())) # key = species name
                random_sequence = fasta_dict[random_key] # select nuc. sequence for a given species
                # Store the genus+species which was randomly selected
                R2_length = len(r2_read_lines[i]) # define read length based on real data
                # randomly select where the R2 read is gonna start from a normal distibuiton
                start = len(random_sequence) - int(np.random.randint(R2_length, len(random_sequence) - R2_length)) 
                # total length - random value
                start_vec.append(start/(len(random_sequence) - R2_length))

                random_sequence_read = random_sequence[start:start + R2_length]
                qual_seq = r2_qual_lines[i].rstrip()
                qual_vec.append(list(map(ord, list(r2_qual_lines[i]))))
                seq1 = Seq(random_sequence_read)
                
                scores_vec = []
                
                for key0, value0 in fasta_dict.items():
                    seq2 = Seq(value0)
                    
                    if key0 not in random_key:
                        tmp_aligns = aligner.score(seq2, seq1)
                        scores_vec.append(tmp_aligns)
 
                
                if not not scores_vec:
                    all_scores.append(np.nanmean(scores_vec)/R2_length)
                    if np.nanmean(scores_vec)/R2_length <= score_thr:
                        random_species_list.append(header.rstrip()+"|"+random_key) 
                        # it will be used as gold standard reference
                        random_only_species_list.append(random_key)
    
                            
                        if not (trunc_range[0] == 0 and trunc_range[1] == 0): # truncating the read sequence
                            truncate_left = round(random.uniform(trunc_range[0],trunc_range[1])*len(random_sequence_read)) # from the left side
                            truncate_right = round(random.uniform(trunc_range[0],trunc_range[1])*len(random_sequence_read)) # from the right side

                            random_sequence_read_trun = random_sequence_read[truncate_left:(len(random_sequence_read) - truncate_right)]
                            if len(random_sequence_read_trun) == 0: # in case everything is truncated and empty sequence is left
                                random_sequence_read_trun = random_sequence_read # take a full sequence without truncation
                        else:
                            random_sequence_read_trun = random_sequence_read # do nothing, just pass the read forward
                            
                        if impute_errors:
                            errors_no = [0, 1, 2] # no error, 1 error
                            num_errors = random.choices(errors_no, weights=(1, 2, 0), k=1)
                            random_sequence_read_trun = impute_seq_error(random_sequence_read_trun, num_errors)
                            
                        # save to output file
                        seq_lns.append(len(random_sequence_read_trun))
                        
                        print(header.rstrip(), file = f) # fastq header
                        print(random_sequence_read_trun, file = f) # randomly picked sequence from part of the 16S gene
                        print('+', file = f) # strand
                        print(qual_seq, file = f) # quality sequence
                        ii = ii + 1
                        
                        if ii == n_reads: # move to next iteration if N reads are reached
                            break

    gsp_output_file_name = os.path.join(output_folder + '_tra_genus_species.txt')
    with open(gsp_output_file_name, 'a+') as f:
        for item in random_species_list:
            print(item, file = f)
            
    # using Counter to find frequency of elements

    frequency = collections.Counter(random_only_species_list)

    if print_stats:
        # printing the frequency
        print("Data generated with ", len(random_species_list), " reads")
        print("Number of species included in the data: ", len(dict(frequency)))
    
        # print alignment statistics
        print("Average alignment scores:")
        print(round(np.mean(all_scores), 4))
        print("Median alignment scores:")
        print(round(np.median(all_scores), 4))
        print("Standard deviation alignment scores:")
        print(round(np.std(all_scores), 4))

        # print average length of reads
        print("Average reads length:")
        print(round(np.mean(seq_lns), 4))
    
    species_list = list(map(lambda x: x, set(random_only_species_list)))
    return all_scores, start_vec, qual_vec, species_list

        
        
#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# Creating simulated data for validation

# Building a simulated R2
## For each FASTQ header, replace the sequence with randomly picked 16S gene (sequence?)
## R2 on this sequence also starts randomly on the sequence (this is dependent on the fragment length)

def validation_data(n_reads, output_path, mic_refs, r2_header_lines, r2_read_lines, r2_qual_lines, species_tra = None, error_rate = 0.001, error_weights = (1, 2, 0), trunc_range = [0,0], print_stats = True, shuffle = True):
    
    random_species_list = [] # To store what genus+species got picked (header)
    random_only_species_list = [] # To store what genus+species got picked

    if shuffle: # shuffle input reads
        tmp_lst = list(zip(r2_header_lines, r2_read_lines, r2_qual_lines))
        random.shuffle(tmp_lst)
        res1, res2, res3 = zip(*tmp_lst)
        r2_header_lines, r2_read_lines, r2_qual_lines = list(res1), list(res2), list(res3)

    if species_tra is not None: # get rid of species that do not match with training data
        for key in list(mic_refs.keys()):
            if key not in species_tra:
                mic_refs.pop(key)
    
    r2_output_file_name = os.path.join(output_path + '_' + str(n_reads) + 'ps_val_simulated.fastq')
    start_vec = []
    seq_lns = []
    avg_read_len = sum(map(len, r2_read_lines))/len(r2_read_lines) # average read length in the data
    #def create_simulated_data(i):
    with open(r2_output_file_name, 'a+') as f:
        for key, value in mic_refs.items():            
            random_key = key #random.choice(list(fasta_dict)) # key = species name
            random_only_species_list.append(random_key)
            random_sequence = value # select nuc. sequence for a given species
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

                random_sequence_read = random_sequence[start:start + R2_length]
      
                if not (trunc_range[0] == 0 and trunc_range[1] == 0): # truncating the read sequence
                    truncate_left = round(random.uniform(trunc_range[0],trunc_range[1])*len(random_sequence_read)) # from the left side
                    truncate_right = round(random.uniform(trunc_range[0],trunc_range[1])*len(random_sequence_read)) # from the right side

                    random_sequence_read_trun = random_sequence_read[truncate_left:(len(random_sequence_read) - truncate_right)]
                    if len(random_sequence_read_trun) == 0: # in case everything is truncated and empty sequence is left
                                random_sequence_read_trun = random_sequence_read # take a full sequence without truncation
                else:
                    random_sequence_read_trun = random_sequence_read # do nothing, just pass the read forward
                            
                tot_num_errors = int(avg_read_len * error_rate)
                
                if tot_num_errors > len(r2_read_lines):
                    num_errors = int(tot_num_errors/len(r2_read_lines))
                    random_sequence_read_trun = impute_seq_error(random_sequence_read_trun, num_errors)
                else:
                    # Randomly select reads which shall include a simulated seq error
                    # zero, one or two errors per read
                    errors_no = [0, 1, 2] # no error, 1 error
                    num_errors = random.choices(errors_no, weights=error_weights, k=1)
                    random_sequence_read_trun = impute_seq_error(random_sequence_read_trun, num_errors)
                    
                qual_seq = r2_qual_lines[i].rstrip()
                
                seq_lns.append(len(random_sequence_read_trun))
                
                # save to output file
                print(header.rstrip(), file = f) # fastq header
                print(random_sequence_read_trun, file = f) # randomly picked sequence from part of the 16S gene
                print('+', file = f) # strand
                print(qual_seq, file = f) # quality sequence
                
                r2_header_lines.remove(header)
                if i == (n_reads-1): # stop if N reads are reached, -1 as starts from 0
                    break

    # Print the genus+species list to output file
    # gold truth genus/species list
    gsp_output_file_name = os.path.join(output_path + '_' + str(n_reads) + 'ps_val_genus_species.txt') # ps - per species

    with open(gsp_output_file_name, 'a+') as f:
        for item in random_species_list:
            print(item, file = f)
            
    # using Counter to find frequency of elements
    frequency = collections.Counter(random_only_species_list)

    if print_stats:
        # printing the frequency
        print("Data generated with ", len(random_species_list), " reads")
        print("Number of species included in the data: ", len(dict(frequency)))

        # print average length of reads
        print("Average reads length:")
        print(round(np.mean(seq_lns), 4))
    
    species_list = list(map(lambda x: x, set(random_only_species_list)))
    return seq_lns, start_vec, species_list, frequency

# sequencing error
def impute_seq_error(row, num_errors):
    """
    Introducing a single base error (replacement). 
    """
    if num_errors[0] == 0: # no error, return it back
        return row
    else:
        inds = list(range(len(row)))
        sam = random.sample(inds, num_errors[0]) # sample on which position read should have an error
        lst = list(row)
        for ind in sam:
            letters_to_draw = ["A", "C", "T", "G"]
            if lst[ind] == "A" or lst[ind] == "T" or lst[ind] == "C" or lst[ind] == "G":
                letters_to_draw.remove(lst[ind]) # as we don't want to impute the same nucleotide
            letts = iter(random.sample(letters_to_draw, 1))
            lst[ind] = next(letts)
        return "".join(lst)

#another attempt to make data estimation for unique regions
## sliding window
#window_size = 150

#for key, value in mic_refs.items():

#    print(key)
#    val = value #list(mic_refs.values())[0]
#    mic_refs.pop(key) # leave one out
#    print(mic_refs_loo)
#    all_kmers = []
#    for i in range(0, len(val) - window_size):
#        all_kmers.append(val[i:(i+window_size)])
#        for species, sequence in mic_refs_loo.items():
#            print(sequence)

#print(len(all_kmers))
