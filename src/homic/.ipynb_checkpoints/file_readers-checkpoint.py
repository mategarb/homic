"""
All functions for reading the files
"""
import pickle
import os
import pandas as pd
import re
import ete3
from Bio.Seq import Seq

#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# FASTA READER

def fasta(path):

    """Reads fasta file.

        No default parameters. All must be specified.

        Parameters
        ----------
        path : string,
            path to the .fasta file.

        Returns
        -------
        fastq_spot_d
            a dict indicating which coordinates (spots) belong to what read. Keys are spot IDs, values are reads IDs.
        info
            pandas DataFrame with following columns:
            'fastq' - fastq full header
            'tile' - tile id (from header)
            'x' - position x  (from header)
            'y' - position y (from header)
            'taxa1' - species part I, truth
            'taxa2' - species part II, truth
            'read' - read sequence 
            'taxa_predictions' - taxid of predictions from Kraken2
            'taxa' - truth species, truth
            'taxa_order' - truth taxa information, ordered 
            'superkingdom' - taxid predictions from Kraken2 translated to taxa info via ete3
            'phylum' - taxid predictions from Kraken2 translated to taxa info via ete3
            'class' - taxid predictions from Kraken2 translated to taxa info via ete3
            'order' - taxid predictions from Kraken2 translated to taxa info via ete3
            'family' - taxid predictions from Kraken2 translated to taxa info via ete3
            'genus' - taxid predictions from Kraken2 translated to taxa info via ete3
            'species' - taxid predictions from Kraken2 translated to taxa info via ete3
            'barcode' - barcode sequence
            'Bx' - barcode position X, spot definition (for synthetic data, assigned randomly)
            'By' - barcode position Y, spot definition (for synthetic data, assigned randomly)
        """

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
            
    fasta_dict[species] = [whole_header, ''.join(new_seq)] # last one
    print("Number of organisms in reference: " + str(len(fasta_dict)))
    
    return fasta_dict # fasta_dict is a dictionary of 2 element lists (header + sequence)


def save_fasta_as_rev_comp(path):
    f_dict = fasta(path)
    path2 = path.replace(".fasta", "")
    path2 = path2 + "_rc.fasta"
    with open(path2, 'a+') as f:
        for spec, info in f_dict.items():
            print(info[0].rstrip(), file = f) # fasta header
            tmp_seq = info[1]
            tmp_seq = tmp_seq.replace("A", "t").replace("C", "g").replace("T", "a").replace("G", "c")
            tmp_seq = tmp_seq.upper()
         
            # reverse strand
            tmp_seq = tmp_seq[::-1]
        
            print(tmp_seq, file = f) # reversing complementary the sequence


#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# FASTQ READER, DL LEARNING SCRIPT

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


def save_fastq_as_rev_comp(path):
    header_lines, read_lines, qual_lines = fastq(path)
    path2 = path.replace(".fastq", "")
    path2 = path2 + "_rc.fastq"
    n = len(read_lines)
    
    with open(path2, 'a+') as f:
        for i in range(n):
            print(header_lines[i].strip(), file = f) # fastq header
            tmp_seq = read_lines[i]
            tmp_seq = tmp_seq.replace("A", "t").replace("C", "g").replace("T", "a").replace("G", "c")
            tmp_seq = tmp_seq.upper()
            tmp_seq = tmp_seq[::-1]
            print(tmp_seq.strip(), file = f) # rev comp seq
            print('+', file = f) # strand
            print(qual_lines[i].strip(), file = f) # quality sequence



#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# FASTQ READER, DL EVALUATION SCRIPT

def fastq2(path):
    fq_d = {}
    for file in os.listdir(path):
        if file.endswith(".fastq"): # _matchBarcode
            name = file.split('_R2')[0]
            
            fastq = parser(os.path.join(path, file))
    
            fq_d[name] = fastq
    return fq_d
    
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
    return ref_d

#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# TAXA INFO, DL MODEL TRAINING

def load_pickle(path):
    with open(path, 'rb') as f:
        ge_sp_dict = pickle.load(f)
    return ge_sp_dict

#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# TAXA INFO, DL MODEL EVALUATION

# fastq header
# @<instrument>:<run number>:<flowcell ID>:<lane>:<tile>:<x-pos>:<y-pos> <read>:<is filtered>:<control number>:<sample number>

# from repo: @M01581:1513:000000000-G54WL:1:1102:17323:1440 2:N:0:TTAGGC
# from SRA: @SRR25456942.1 M01581:1638:000000000-DCD22:1:1102:14231:2672 length=150 Blautia argi

# Load pickle file
def make_benchmark_table(path, reads, krk_preds, bcodes):

    """Creates a table for benchmarking spots.

        No default parameters. All must be specified.

        Parameters
        ----------
        path : string,
            path to the gold truth species / genus list.
        reads : list,
            "reads" list read with the file_readers.fastq function
        krk_preds : pandas Series,
            taken as a column "taxa" from output of file_readers.load_kraken2_output(path)
        bcodes : pandas DataFrame,
            the output from file_readers.load_barcodes(path)
            

        Returns
        -------
        fastq_spot_d
            a dict indicating which coordinates (spots) belong to what read. Keys are spot IDs, values are reads IDs.
        info
            pandas DataFrame with following columns:
            'fastq' - fastq full header
            'tile' - tile id (from header)
            'x' - position x  (from header)
            'y' - position y (from header)
            'taxa1' - species part I, truth
            'taxa2' - species part II, truth
            'read' - read sequence 
            'taxa_predictions' - taxid of predictions from Kraken2
            'taxa' - truth species, truth
            'taxa_order' - truth taxa information, ordered 
            'superkingdom' - taxid predictions from Kraken2 translated to taxa info via ete3
            'phylum' - taxid predictions from Kraken2 translated to taxa info via ete3
            'class' - taxid predictions from Kraken2 translated to taxa info via ete3
            'order' - taxid predictions from Kraken2 translated to taxa info via ete3
            'family' - taxid predictions from Kraken2 translated to taxa info via ete3
            'genus' - taxid predictions from Kraken2 translated to taxa info via ete3
            'species' - taxid predictions from Kraken2 translated to taxa info via ete3
            'barcode' - barcode sequence
            'Bx' - barcode position X, spot definition (for synthetic data, assigned randomly)
            'By' - barcode position Y, spot definition (for synthetic data, assigned randomly)
        """
    
    rows_nams = pd.read_csv(path, sep=' ', header = None,usecols=[1], engine='python', names = ['fastq']) # full header only
    info = pd.read_csv(path, sep='[ ,:,|]', header = None, usecols=[5, 6, 7, 9, 10], names = ['tile', 'x', 'y','taxa1', 'taxa2'], engine='python') # new files, sra-based
    # info = pd.read_csv(path, sep='[ ,:,|]', header = None, usecols=[4, 5, 6, 11, 12], names = ['tile', 'x', 'y','taxa1', 'taxa2'], engine='python') # old files
    
    info['read'] = reads
    info['kraken_preds'] = krk_preds
    #info.index = rows_nams.iloc[:, 0] # rename rows so they include fastq header
    info = pd.concat([rows_nams, info], axis=1)
    
    info['truth_taxa'] = info['taxa1'] + ' ' + info['taxa2']
    info.drop(columns=['taxa1', 'taxa2'])

    info.loc[info["kraken_preds"] == "[Eubacterium] eligens", "truth_taxa"] = 'Lachnospira eligens'
    info.loc[info["truth_taxa"] == "[Eubacterium] eligens", "truth_taxa"] = 'Lachnospira eligens' # cause it has different names
    #info['read'] = info['fastq'].map(fq_d[name])
    #ref_df = pd.DataFrame.from_dict(ref_d, orient='index', columns=['taxa'])
    #ge_sp_dict = load_pickle(taxa_info_path)
    ###### for gold truth from synthetic ###### ###### ###### ######
    ncbi = ete3.NCBITaxa()
    taxids = ncbi.get_name_translator(info["truth_taxa"])
    taxidsf = sum(list(taxids.values()), []) # flatten the list
    taxids_dict = dict(zip(taxids.keys(), taxidsf)) ## swapping values with keys
    taxon_id = set(taxidsf)

    #print(taxon_id)
    #lineage = ncbi.get_lineage(taxon_id)

    lineage_df = {} #pd.DataFrame()
    for tmp_taxid in taxon_id:
        if tmp_taxid != 0:
            tmp_lineage = pd.Series({rank : taxon
                                     for taxon, rank in ncbi.get_rank(
                                         ncbi.get_lineage(tmp_taxid)).items()
                                    })
            tmp_lineage = pd.Series(index=tmp_lineage.index,
                                    data =ncbi.translate_to_names(tmp_lineage))
        
            tmp_lineage.name = tmp_taxid
            tmp_lineage.fillna(value='unassigned')
            lineage_df[tmp_taxid] = tmp_lineage#pd.concat([lineage_df, tmp_lineage], axis=1)

        else:
            nms = ['no rank', 'superkingdom', 'phylum', 'class', 'family', 'genus', 'kingdom', 'species', 'order']
            lineage_df[tmp_taxid] = pd.DataFrame(['unassigned'] * len(nms), index=nms)
    
    taxids_full = [*map(taxids_dict.get, info["truth_taxa"].tolist())] # mapping between species and tax ids, to get tax ids only
    tmp_res = list(map(lineage_df.get, taxids_full))

    lineage_df_all = pd.concat(tmp_res, axis=1, ignore_index=True)
    lineage_df_all = lineage_df_all.loc[['superkingdom', 'phylum', 'class', 'order', 'family', 'genus','species']]
    lineage_df_all = lineage_df_all.T
    lineage_df_all = lineage_df_all.fillna('unassigned')
    lineage_df_all = lineage_df_all.iloc[:, ::-1] # reversing the order
    
    #ref_df['taxa_order'] = ref_df['taxa'].map(ge_sp_dict)
    info['truth_taxa_order'] = lineage_df_all.apply(','.join, axis=1).tolist()
    info['truth_taxa_order'] = info.apply(lambda row: change_order(row['truth_taxa_order']), axis=1)
    info['truth_taxa_order'] = info.apply(lambda row: rm_species(row['truth_taxa_order']), axis=1)

    ###### for kraken2 ###### ###### ###### ######
    taxids = info['kraken_preds']
    taxon_id = set(taxids.to_list())

    #print(taxon_id)
    #lineage = ncbi.get_lineage(taxon_id)

    lineage_df = {} #pd.DataFrame()
    for tmp_taxid in taxon_id:
        if str(tmp_taxid.strip()) != '0':
            tmp_lineage = pd.Series({rank : taxon
                                     for taxon, rank in ncbi.get_rank(
                                         ncbi.get_lineage(tmp_taxid)).items()
                                    })
            tmp_lineage = pd.Series(index=tmp_lineage.index,
                                    data =ncbi.translate_to_names(tmp_lineage))
        
            tmp_lineage.name = tmp_taxid
            tmp_lineage.fillna(value='unassigned')
            lineage_df[tmp_taxid] = tmp_lineage#pd.concat([lineage_df, tmp_lineage], axis=1)

        else:
            nms = ['no rank', 'superkingdom', 'phylum', 'class', 'family', 'genus', 'kingdom', 'species', 'order']
            lineage_df[tmp_taxid] = pd.DataFrame(['unassigned'] * len(nms), index=nms)
            #lineage_df = pd.concat([lineage_df, una_df], axis=1)

    tmp_res = list(map(lineage_df.get, taxids.to_list()))
    lineage_df_all = pd.concat(tmp_res, axis=1, ignore_index=True)
    lineage_df_all = lineage_df_all.loc[['superkingdom', 'phylum', 'class', 'order', 'family', 'genus','species']]
    lineage_df_all = lineage_df_all.T
    lineage_df_all = lineage_df_all.fillna('unassigned')
    info = pd.concat([info, lineage_df_all], axis=1)


    bcodes_rand = bcodes.sample(n=info.shape[0], replace=True)
    bcodes_rand = bcodes_rand.reset_index(drop=True)
    
    info = pd.concat([info, bcodes_rand], axis=1)
    
    # For truth - get fastq headers per spot
    fastq_spot_d = {}
    info_grouped = info.groupby(['Bx', 'By'])
    
    for label, group in info_grouped:
        spot = str(label[0]) + 'x' + str(label[1])
        fastq_spot_d[spot] = group['fastq'].tolist()

    # randomly add barcodes in order to mimic spots, normally it would be done with taggd

    info['Bx'] = info['Bx'].map(str)
    info['By'] = info['By'].map(str)

    return fastq_spot_d, info

def load_barcodes(path):
    bcodes = pd.read_csv(path, delimiter="\t", header = None)
    bcodes.columns = ['barcode', 'Bx', 'By']
    return bcodes

def load_kraken2_output(path):
    info = pd.read_csv(path, delimiter="\t", header = None) # , usecols=[4, 5, 6, 11, 12], names = ['tile', 'x', 'y','taxa1', 'taxa2'],
    info.columns = ['class', 'read_id', 'species', 'read_length','kmers']

    species_split = info['species'].str.split('taxid', expand=True)

    info['species'] = species_split[0]
    info['taxid'] = species_split[1]

    #species_split2 = info['species'].str.split(r'[ (]', expand=True)
    #info['species'] = species_split2[0]
    info['species'] = info['species'].str.replace(' (', '')
    info['taxid'] = info['taxid'].str.replace(')', '')
    return info

def change_order(row):
    rv_row = row.split(',')
    rv_row.reverse()
    return ','.join(rv_row) # return species level

def rm_species(row):
    rv_row = row.split(',')
    if len(rv_row) == 7:
        return ','.join(rv_row[1:]) # return genus level
    else:
        return ','.join(rv_row) # return genus level

#####################################################################################################
#####################################################################################################
#####################################################################################################
################################# PARSER

def parser(filename):
    """
    Generate a list of tuples (header, read)
    """
    fastq_parsed = {}
    try:

        with open(filename) as fq:
            header = next(fq)
            read = next(fq)
            fastq_parsed[header[1:-1].split(' ')[0]] = read[:-1] 
            while True:
                next(fq) # skip + line
                next(fq) # skip qscore
                header = next(fq) # store next read header
                read = next(fq) # store next read seq
                fastq_parsed[header[1:-1].split(' ')[0]] = read[:-1]
    except:
        StopIteration 
    return fastq_parsed
    
# Read in header ref file, ie fastq header is paired with taxa the sequence was made from

#ref_d = {}
#i = 0
#with open(ref_input) as file:
#    for line in file:
#        ref_d[i] = line
#        i = i + 1