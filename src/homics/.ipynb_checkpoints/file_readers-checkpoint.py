"""
All functions for reading the files
"""
import pickle
#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# FASTA READER

def fasta(path):

    fasta_file = path # path to fasta with multiple references
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
            
    return fasta_dict # fasta_dict is a dictionary of 2 element lists (header + sequence)


#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# FASTQ READER

def fastq(path):

    real_sample_R2_file = path
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
    # Select all reads sequences
    r2_read_lines = r2_lines[1::4]
    # Select all quality lines
    r2_qual_lines = r2_lines[3::4]


    if exten == ".gz":
        os.remove(decompressed_real_sample_R2_file)
        
    return r2_header_lines, r2_read_lines, r2_qual_lines

#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# GOLD STANDARD SPECIES NAMES

def species_outcome(path):
    ref_d = {}
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            header = line.split(' ')[0][1:]
            sp = line.split('|')[1].rstrip()
            ref_d[header] = sp
            print(header)
            print(sp)
    return ref_d

#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# TAXA INFO 

def load_pickle(path):
    with open(path, 'rb') as f:
        ge_sp_dict = pickle.load(f)
    return ge_sp_dict

# Read in header ref file, ie fastq header is paired with taxa the sequence was made from

#ref_d = {}
#i = 0
#with open(ref_input) as file:
#    for line in file:
#        ref_d[i] = line
#        i = i + 1