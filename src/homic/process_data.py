#!/usr/bin/env python
import os
import re
import pysam
import ctypes
import shutil
import numpy as np
import pandas as pd
import statistics as st
import seaborn as sns
import os
from os import path
import subprocess
from Bio import Blast
from Bio import SeqIO
from Bio import Entrez
import time
import re
from .kraken2 import decontaminate_paired, decontaminate_single
from .file_readers import fastq
import glob
from Bio import SeqIO
from collections import Counter

def trim_decon(dbpath, file1, file2, adapt_seq_path, head_crop=19, crop=260, threads=32):
    
    """Trimming with trimmomatic and decontaminatig reads with kraken2.

        Parameters
        ----------
        dbpath : string,
            a path to the kraken2 db
        file1 : string,
            a path #1.fastq file
        file2 : string,
            a path #2.fastq file
        threads : string,
            number of threads
            
        Returns
        -------
        no output, files are saved
    """
    
# 1. trimming
    if file1[-2:] == "gz" and file2[-2:] == "gz":
        file11 = file1.replace(".fastq.gz", "_trimmed.fastq")
        file12 = file1.replace(".fastq.gz", "_un_trimmed.fastq")
        file21 = file2.replace(".fastq.gz", "_trimmed.fastq")
        file22 = file2.replace(".fastq.gz", "_un_trimmed.fastq")
    else:
        file11 = file1.replace(".fastq", "_trimmed.fastq")
        file12 = file1.replace(".fastq", "_un_trimmed.fastq")
        file21 = file2.replace(".fastq", "_trimmed.fastq")
        file22 = file2.replace(".fastq", "_un_trimmed.fastq")

    print("Trimming and filtering")
    cmd = ["trimmomatic",
           "PE",
            "-threads",
            str(threads),
            file1,
            file2,
            file11,
            file12,
            file21,
            file22,
            "ILLUMINACLIP:" + adapt_seq_path + ":2:30:10",
            "HEADCROP:" + str(head_crop),
            "CROP:" + str(crop),
            "LEADING:3",
            "TRAILING:3",
            "SLIDINGWINDOW:4:15",
            "MINLEN:36",] ## settings from here http://www.usadellab.org/cms/uploads/supplementary/Trimmomatic/TrimmomaticManual_V0.32.pdf
    subprocess.call(cmd)

    os.remove(file12)
    os.remove(file22)


# 2. decontaminating
    print("Decontaminating reads")

    output = file1.replace("_R1_001.fastq.gz", "_k2") # the same folder where original files are
    
    decontaminate_paired(db_path = dbpath, 
                                input_file1 = file11,
                                input_file2 = file21,
                                output = output,
                                threads = threads)
    os.remove(file11)
    os.remove(file21)

def assemble_decon(path, dbpath, file1, file2, samp_id="no_id", threads=16, min_con_len = 500):

    """Assembling with megahit and decontaminatig contigs with kraken2.

        Parameters
        ----------
        path : string,
            a path to the folder with files
        dbpath : string,
            a path to the kraken2 db
        file1 : string,
            a path #1.fastq file
        file2 : string,
            a path #2.fastq file
        threads : string,
            number of threads
            
        Returns
        -------
        no output, files are saved under path
    """
    
# 3. assembling
#    print("Assembling")
#    cmd2 = ["megahit",
#            "--12",
#            output,
#            "-o",
#            path,]
#    subprocess.call(cmd2)

    paths_r1 = file1
    paths_r2 = file2
    cmd2 = ["megahit",
            "-1",
            paths_r1,
            "-2",
            paths_r2,
            "-o",
            path + "/out_assembly",
            "--min-contig-len",
            str(min_con_len), # default 200
            "-t",
            str(threads),]  # generic metagenomes settings, default

    shutil.rmtree(path + '/out_assembly', ignore_errors=True)
    
    subprocess.call(cmd2)
    
    #os.remove(paths_r1)
    #os.remove(paths_r2)
    
    
# 4. decontaminating II 
    print("Decontaminating contigs")
    file = path + "/out_assembly/final.contigs.fa"

    output = file.replace("out_assembly/final.contigs.fa", "metagenome_contigs_" + samp_id)
    decontaminate_single(db_path = dbpath, 
                      input_file = file,
                      output = output,
                      threads = threads)
    
    shutil.rmtree(path + '/out_assembly', ignore_errors=True)
    print("Done!")




def bow_bat(assembly_path, file_1, file_2, out_dir, only_metabat=False, min_contig = 1500): # bowtie + metabat

    """Assembling with megahit and decontaminatig contigs with kraken2.

        Parameters
        ----------
        path : string,
            a path to the folder with files
        dbpath : string,
            a path to the kraken2 db
        file1 : string,
            a path #1.fastq file
        file2 : string,
            a path #2.fastq file
        threads : string,
            number of threads
            
        Returns
        -------
        no output, files are saved under path
    """
    
    # Create output directory if it doesn't exist
    if not only_metabat:

        lengths = [len(rec.seq) for rec in SeqIO.parse(assembly_path, "fasta")]
        print("Number of contigs:", len(lengths))
        print("Longest contig:", max(lengths))
        print("Shortest contig:", min(lengths))
        print("Average contig length:", st.mean(lengths))

        os.makedirs(out_dir, exist_ok=True)
        
        # Build Bowtie2 index from contigs
        print("Building Bowtie2 index...")
        ass_ind = os.path.join(out_dir, "assembly_index")
        subprocess.run([
            "bowtie2-build",
            assembly_path,
            ass_ind
        ])
        
        # Map paired reads to contigs
        print("Mapping reads to contigs with Bowtie2...")
        bam_file = os.path.join(out_dir, "reads_mapped.bam")
        sam_file = os.path.join(out_dir, "reads_mapped.sam")
    
        
        subprocess.run([
            "bowtie2",
            "-x", ass_ind,
            "-1", file_1,
            "-2", file_2,
            "-S", sam_file
        ])
        
        # Convert SAM to BAM and sort
        print("Converting SAM to sorted BAM...")
        subprocess.run([
            "samtools", "view", "-bS", sam_file, "-o", bam_file
        ])
        
        sorted_bam = os.path.join(out_dir, "reads_mapped_sorted.bam")
        subprocess.run([
            "samtools", "sort", "-o", sorted_bam, bam_file
        ])
        
        # Index BAM
        subprocess.run([
            "samtools", "index", sorted_bam
        ])

        # Remove redundant files
        os.remove(bam_file)
        os.remove(sam_file)    
        #start_char = "assembly_index"  # files starting with 'a'

        #pattern = os.path.join(out_dir, f"{start_char}*")
        #for file_path in glob.glob(pattern):
        #    if os.path.isfile(file_path):  # make sure it's a file
        #        os.remove(file_path)
        
        # Run MetaBAT2
        subprocess.run([
            "metabat2",
            "-i", assembly_path,
            "-a", sorted_bam,
            "-o", os.path.join(out_dir, "bin"),
            "--minContig", str(min_contig)
        ])
    else:

        lengths = [len(rec.seq) for rec in SeqIO.parse(assembly_path, "fasta")]
        print("Number of contigs:", len(lengths))
        print("Longest contig:", max(lengths))
        print("Shortest contig:", min(lengths))
        print("Average contig length:", st.mean(lengths))
        
        # Run MetaBAT2
        os.makedirs(out_dir, exist_ok=True)
        sorted_bam = os.path.join(out_dir, "reads_mapped_sorted.bam")
        subprocess.run([
            "metabat2",
            "-i", assembly_path,
            "-a", sorted_bam,
            "-o", os.path.join(out_dir, "bin"),
            "--minContig", str(min_contig)
        ])


def run_blastn(path_fa, path_db, path_out, nthreads=16, evalue=1e-6, max_ts=1, max_h=1, ofmt="6 qseqid sseqid pident length evalue bitscore score stitle"):

    """Runs blastn from the python level.

        Parameters
        ----------
        path_fa : string,
            a path to the fasta reference/s
        path_db : string,
            a path to the database
        path_out : string,
            an output path
        nthreads : string,
            number of threads
        evalue : string,
            evalue threshold
        ofmt : string,
            format of output result from blast
            
        Returns
        -------
        no output, files are saved under path_out
    """

    #cmd = ["export PATH=$HOME/ncbi-blast-2.16.0+/bin:$PATH",]  # Generic metagenomes settings, default
    #print(cmd)
    #subprocess.call(cmd)
    
    #makeblastdb -in ./GbBacU/all_seqs.fna -dbtype nucl
    
    cmd2 = ["blastn",
            "-query",
            path_fa,
            "-db",
            path_db,
            "-num_threads",
            str(nthreads),
            "-evalue",
            str(evalue),
            "-max_target_seqs",
            str(max_ts),
            "-max_hsps",
            str(max_h),
            "-outfmt",
            ofmt,
            "-out",
            path_out]  # Generic metagenomes settings, default
    
    subprocess.call(cmd2)


def clean_word(word):
    word = word.split(' ', 1)[1]
    word = word.replace('MAG: ', '')
    word = word.replace('uncultured ', '')
    return word

def select_species(record):
    record = record.split()[:2]
    record = ' '.join(record)
    return record
    
def select_genus(record):
    record = record.split()[:1]
    record = ' '.join(record)
    return record


def perc_contigs_assigned(path_blast, path_ctgs):

    data = pd.read_csv(path_blast, header = None, delimiter = "\t")
    data = data.rename({0: "contig_id", 1: "subject_id", 2: "pident", 3: "length", 
                            4: "evalue", 5: "bitscore", 6: "score", 7: "subject_title"}, axis='columns')
    
    blast_ctgs = set(data["contig_id"].to_list())
    all_ctgs = fastq(path_ctgs)
    
    
    all_ctgs_heads = np.array(all_ctgs[0]).flatten().tolist()
    
    all_ctgs_ids = []
    for s in all_ctgs_heads:
        idx = s.find(">") + 1  # find position of starting character
        if idx != -1:
            # take substring from idx until first space after idx
            end_idx = s.find(" ", idx)
            if end_idx == -1:  # no space found, take till end
                all_ctgs_ids.append(s[idx:])
            else:
                all_ctgs_ids.append(s[idx:end_idx])
        else:
            all_ctgs_ids.append("")
    
    inter_ctgs = list(blast_ctgs & set(all_ctgs_ids))
    
    return len(inter_ctgs)/len(all_ctgs_ids)

    
    
def read_n_clean_blastn(path_blast, top_hits = True, evalue = 1e-100, pident=0.98, drop_sp = False, drop_uncultured = True, drop_bacterium=True, drop_virus=True, best_unique = False):

    """Reads and cleans output from blastn.

        Parameters
        ----------
        path_blast : string,
            a path to the blast result
        top_hits : boolean,
            tophits only
        evalue : float,
            minimum evalue to keep a record
        drop_sp : boolean,
            drop species marked with suffix .sp
            
        Returns
        -------
        data
            pandas data frame with formatted blastn results
    """
    
    data = pd.read_csv(path_blast, header = None, delimiter = "\t")
    data = data.rename({0: "contig_id", 1: "subject_id", 2: "pident", 3: "length", 
                        4: "evalue", 5: "bitscore", 6: "score", 7: "subject_title"}, axis='columns')

    ### filtering
    # e-value
    data = data[data['evalue'] < evalue]
    # pident    
    data = data[data['pident'] > pident]
    
    ### clean MAGs
    all_specs = data["subject_title"]
    mag_rem = []
    for string in all_specs:
        new_str = re.sub(r'MAG(?:[\s_]*TPA_asm)?:?\s*', '', string)
        new_str = re.sub(r'\s*\d+$', '', new_str)
        mag_rem.append(new_str)
        
    mag_rem = [s.replace('TPA_exp:','') for s in mag_rem]
    data["subject_title"] = mag_rem

    ### clean brackets
    all_specs = data["subject_title"]
    data_nobr = [s.replace('[','').replace(']','') for s in all_specs]
    data["subject_title"] = data_nobr

    if drop_virus:
        rem_und = ["virus" not in spec.lower() for spec in data["subject_title"]]
        data = data[rem_und]
            
    ### removing reduntant words from titles to get species
    words = data["subject_title"].to_list()
    recs = list(map(clean_word, words))
    all_species = list(map(select_species, recs))
    all_genus = list(map(select_genus, recs))
    
    data["species"] = all_species
    data["genus"] = all_genus

    # other things to clean
    #CrAss-like
    #16S
    
    ### droping undefined species
    if drop_sp:
        rem_und = ["sp." not in spec for spec in data["species"]]
        data = data[rem_und]

    if drop_uncultured:
        rem_und = ["uncultured" not in spec.lower() for spec in data["species"]]
        data = data[rem_und]

    if drop_bacterium:
        rem_und = ["bacterium" not in spec.lower() for spec in data["species"]]
        data = data[rem_und]

    if top_hits:
        data = (
            data.sort_values(by=["contig_id", "evalue", "bitscore"], ascending=[True, True, False])
          .drop_duplicates(subset="contig_id", keep="first")
          .reset_index(drop=True))
        
    if best_unique:
        data = (
            data.sort_values(by=["species", "evalue", "bitscore"], ascending=[True, True, False])
          .drop_duplicates(subset="species", keep="first")
          .reset_index(drop=True))

    tmp_rem = ["homo" not in spec.lower() for spec in data["genus"]]
    data = data[tmp_rem]

    tmp_rem = ["16S" not in spec.lower() for spec in data["genus"]]
    data = data[tmp_rem]

    tmp_rem = ["human" not in spec.lower() for spec in data["genus"]]
    data = data[tmp_rem]

    tmp_rem = ["phage" not in spec.lower() for spec in data["genus"]]
    data = data[tmp_rem]

    tmp_rem = ["crassphage" not in spec.lower() for spec in data["genus"]]
    data = data[tmp_rem]

    ### finally, sorting by two columns: evalue & bitscore
    data = data.sort_values(by=['evalue'], ascending=True)
    data = data.sort_values(by=['bitscore'], ascending=False)

    return data

def chao_idx(taxa_list): #a non-parametric method for estimating the total number of species
    cnt = Counter(taxa_list)  # source: https://palaeo-electronica.org/2011_1/238/estimate.htm
    S_obs = len(cnt)
    F1 = sum(1 for v in cnt.values() if v == 1)
    F2 = sum(1 for v in cnt.values() if v == 2)
    
    if F2 == 0:
        chao1i = None
    else:
        chao1i = S_obs + (F1 * F1) / (2 * F2)

    return chao1i, S_obs, F1, F2

def jknife1(taxa_list, m): # source: https://palaeo-electronica.org/2011_1/238/estimate.htm
    cnt = Counter(taxa_list)  
    S_obs = len(cnt)
    Q1 = sum(1 for v in cnt.values() if v == 1)
    Q2 = sum(1 for v in cnt.values() if v == 2)
    
    if Q2 == 0:
        jk1 = None
    else:
        jk1 = S_obs + (Q1 * (m - 1))/m

    return jk1, S_obs, Q1, Q2

def jknife2(taxa_list, m): # source: https://palaeo-electronica.org/2011_1/238/estimate.htm
    cnt = Counter(taxa_list)  
    S_obs = len(cnt)
    Q1 = sum(1 for v in cnt.values() if v == 1)
    Q2 = sum(1 for v in cnt.values() if v == 2)
    
    if Q2 == 0:
        jk2 = None
    else:
        jk2 = S_obs + (((Q1 * (2*m - 3))/m) - ((Q2 * (m - 2) * (m - 2))/(m * (m - 1))))

    return jk2, S_obs, Q1, Q2

    
## below is snippet from the Brittas repo

# next steps are outside in bash, to map assemblies to reads and bin (optionall), finally perofrm blastn
#bam_header = {
#        'HD': {'VN': '1.5', 'SO':'unsorted'},
#        'RG': [{'ID': '0', 'SM' : 'unknown_sample', 'PL' : 'ILLUMINA' }]
#    }

#self_fastq_fw = sys.argv[1] # forward read
#self_fastq_rv =	sys.argv[2] # reverse read
#output = sys.argv[3] #output file
#self_min_length_trimming = int(sys.argv[4])

# Get the directory to put the output file in and the file name
#output_path = os.path.dirname(output)
#output_name = os.path.basename(output)

#output_file = os.path.join(output_path, output_name+"_R2_quality_trimmed.bam")
#out_log = os.path.join(output_path, output_name+"_R2_quality_trimmed.log")
#
#FILENAMESquality_trimmed_R2 = output_file
#FILENAMES_DISCARDEDquality_trimmed_discarded = "R2_quality_trimmed_discarded.fastq"
#self_keep_discarded_files = None # Don't save reads which did not go through quality filteirng
#barcode_length = 18
#self_barcode_start = 0
#self_filter_AT_content = 90
#self_filter_GC_content = 90
#self_umi_start_position = 18
#self_umi_end_position = 25
#self_min_quality_trimming = 16
##self_min_length_trimming = 20
#self_remove_polyA_distance = 10
#self_remove_polyT_distance = 10
#self_remove_polyG_distance = 10
#self_remove_polyC_distance = 10
#self_remove_polyN_distance = 10
#self_qual64 = False
#self_umi_filter = False
#self_umi_filter_template = "NNNNNNN"
#self_umi_quality_bases = 5
#self_adaptor_missmatches = 0
#self_overhang = 2
#self_disable_umi = False
#self_disable_barcode = False
#
        
def InputReadsFilter(fw,
                     rv,
                     out_rv,
                     out_rv_discarded,
                     barcode_length,
                     start_position,
                     filter_AT_content,
                     filter_GC_content,
                     umi_start,
                     umi_end,
                     min_qual,
                     min_length,
                     polyA_min_distance,
                     polyT_min_distance,
                     polyG_min_distance,
                     polyC_min_distance,
                     polyN_min_distance,
                     qual64,
                     umi_filter,
                     umi_filter_template,
                     umi_quality_bases,
                     adaptor_missmatches,
                     overhang,
                     disable_umi,
                     disable_barcode):
    """
    This class handles the input read filtering and quality trimming
      - It performs a sanity check (forward and reverse reads same length and order)
      - It performs a BWA-based quality trimming discarding very short reads
      - It removes adaptors from the reads (optional)
      - It checks for AT and GC content (optional)
      - It performs a sanity check on the UMI (optional)
    Reads that do not pass the filters are discarded (both R1 and R2)
    Reads that pass the filter are written as BAM (R2)
    :param fw: the FASTQ file with the forward reads (R1)
    :param rv: the FASTQ file with the reverse reads (R2)
    :param out_rv: the name of the output file for reverse (BAM)
    :param out_rv_discarded: the name of the output file for discarded reverse reads (FASTQ)
    :param barcode_length: length of the barcode sequence (integer)
    :param start_position: the start position of the barcode
    :param filter_AT_content: % of A and T bases a read2 must have to be discarded
    :param filter_GC_content: % of G and C bases a read2 must have to be discarded
    :param umi_start: the start position of the UMI
    :param umi_end: the end position of the UMI
    :param min_qual: the min quality value to use in the trimming
    :param min_length: the min valid length for a read after trimming
    :param polyA_min_distance: if >5 remove PolyA adaptors from the reads
    :param polyT_min_distance: if >5 remove PolyT adaptors from the reads
    :param polyG_min_distance: if >5 remove PolyG adaptors from the reads
    :param polyC_min_distance: if >5 remove PolyC adaptors from the reads
    :param polyN_min_distance: if >5 remove PolyN adaptors from the reads
    :param qual64: true of qualities are in phred64 format
    :param umi_filter: performs a UMI quality template filter when True
    :param umi_filter_template: the template to use for the UMI filter
    :param umi_quality_bases: the number of low quality bases allowed in an UMI
    :param adaptor_missmatches: number of miss-matches allowed when removing adaptors
    :param overhang: overhang to be used for the barcodes (integer)
    :param disable_umi: true if the reads do not contain UMIs
    :param disable_barcode: true if the reads do not contain barcodes
    """
    #logger = logging.getLogger("STPipeline")
    #if not (os.path.isfile(fw) or is_fifo(fw)) or not (os.path.isfile(rv) or is_fifo(rv)):
    #    error = "Error doing quality trimming, input file/s not present {}\n{}\n".format(fw,rv)
    #    logger.error(error)
    #    raise RuntimeError(error)

    # Check if discarded files must be written out
    self_keep_discarded_files = out_rv_discarded is not None

    # Build fake sequence adaptors with the parameters given
    adaptorA = "".join("A" for k in range(polyA_min_distance))
    adaptorT = "".join("T" for k in range(polyT_min_distance))
    adaptorG = "".join("G" for k in range(polyG_min_distance))
    adaptorC = "".join("C" for k in range(polyC_min_distance))
    adaptorN = "".join("N" for k in range(polyN_min_distance))
    print(polyA_min_distance, polyT_min_distance)
    # Not recommended to do adaptor trimming for adaptors smaller than 5
    do_adaptorA = polyA_min_distance >= 5
    do_adaptorT = polyT_min_distance >= 5
    do_adaptorG = polyG_min_distance >= 5
    do_adaptorC = polyC_min_distance >= 5
    do_adaptorN = polyN_min_distance >= 5
    do_AT_filter = filter_AT_content > 0
    do_GC_filter = filter_GC_content > 0

    # Quality format
    phred = 64 if qual64 else 33
      
    # Some counters
    total_reads = 0
    dropped_umi = 0
    dropped_umi_template = 0
    dropped_AT = 0
    dropped_GC = 0
    dropped_adaptor = 0
    too_short_after_trimming = 0
          
    # Some variables to avoid overhead in the loop
    #header_fw
    #sequence_fw
    #quality_fw
    #header_rv
    #sequence_rv
    #quality_rv
    #orig_sequence_rv
    #orig_quality_rv
    #discard_read
    
    # Create output file writers
    bam_file = pysam.AlignmentFile(out_rv, "wbu", header=bam_header) # Want to remove this
    fw_file = safeOpenFile(fw, "rU")
    rv_file = safeOpenFile(rv, "rU")
    if self_keep_discarded_files:
        out_rv_handle_discarded = safeOpenFile(out_rv_discarded, 'w')
        out_rv_writer_discarded = writefq(out_rv_handle_discarded)
        
    for (header_fw, sequence_fw, quality_fw), \
    (header_rv, sequence_rv, quality_rv) in zip(readfq(fw_file), readfq(rv_file)):
        
        discard_read = False
        orig_sequence_rv, orig_quality_rv = sequence_rv, quality_rv
        total_reads += 1
        
        if not sequence_fw or not sequence_rv:
            error = "Error doing quality trimming.\n" \
            "The input files are not of the same length"
            # logger.error(error)
            break

        if header_fw.split()[0] != header_rv.split()[0]:
            print("Pair reads found with different names {} and {}".format(header_fw,header_rv))
            #logger.warning("Pair reads found with different " "names {} and {}".format(header_fw,header_rv))

        # get the barcode sequence
        if disable_barcode:
            barcode = None
        else:
            barcode = sequence_fw[max(0,start_position-overhang):(start_position+barcode_length+overhang)]

        if not disable_umi:
            # If we want to check for UMI quality and the UMI is incorrect
            # then we discard the reads
            umi_seq = sequence_fw[umi_start:umi_end]
            if umi_filter \
            and not check_umi_template(umi_seq, umi_filter_template):
                dropped_umi_template += 1
                discard_read = True

            # Check if the UMI has many low quality bases
            umi_qual = quality_fw[umi_start:umi_end]
            if not discard_read and (umi_end - umi_start) >= umi_quality_bases and \
            len([b for b in umi_qual if (ord(b) - phred) < min_qual]) > umi_quality_bases:
                dropped_umi += 1
                discard_read = True
        else:
            umi_seq = None

        # If reverse read has a high AT content discard...
        if not discard_read and do_AT_filter and \
        ((sequence_rv.count("A") + sequence_rv.count("T")) / len(sequence_rv)) * 100 >= filter_AT_content:
            dropped_AT += 1
            discard_read = True

        # If reverse read has a high GC content discard...
        if not discard_read and do_GC_filter and \
        ((sequence_rv.count("G") + sequence_rv.count("C")) / len(sequence_rv)) * 100 >= filter_GC_content:
            dropped_GC += 1
            discard_read = True

        if not discard_read:
            # Perform adaptor/homopolymer filters
            if do_adaptorA and len(sequence_rv) > min_length:
                sequence_rv, quality_rv = removeAdaptor(
                    sequence_rv, quality_rv, adaptorA, adaptor_missmatches)
            if do_adaptorT and len(sequence_rv) > min_length:
                sequence_rv, quality_rv = removeAdaptor(
                    sequence_rv, quality_rv, adaptorT, adaptor_missmatches)
            if do_adaptorG and len(sequence_rv) > min_length:
                sequence_rv, quality_rv = removeAdaptor(
                    sequence_rv, quality_rv, adaptorG, adaptor_missmatches)
            if do_adaptorC and len(sequence_rv) > min_length:
                sequence_rv, quality_rv = removeAdaptor(
                    sequence_rv, quality_rv, adaptorC, adaptor_missmatches)
            if do_adaptorN and len(sequence_rv) > min_length:
                sequence_rv, quality_rv = removeAdaptor(
                    sequence_rv, quality_rv, adaptorN, adaptor_missmatches)

            # Check if the read is smaller than the minimum after removing artifacts
            if len(sequence_rv) < min_length:
                dropped_adaptor += 1
                discard_read = True

        if not discard_read:
            # Trim reverse read (will return None if length of trimmed sequence is less than min_length)
            sequence_rv, quality_rv = trim_quality(
                sequence_rv,
                quality_rv,
                min_qual,
                min_length,
                phred)
            if not sequence_rv or not quality_rv:
                too_short_after_trimming += 1
                discard_read = True

        if not discard_read:
            bam_file.write(
                convert_to_AlignedSegment(
                    header_rv,
                    sequence_rv,
                    quality_rv,
                    barcode,
                    umi_seq))
        elif self_keep_discarded_files:
            out_rv_writer_discarded.send((header_rv, orig_sequence_rv, orig_quality_rv))

    bam_file.close()                    
    fw_file.close()
    rv_file.close()
    if self_keep_discarded_files:
        out_rv_handle_discarded.flush()
        out_rv_handle_discarded.close()
        out_rv_writer_discarded.close()
                
    # Write info to the log
    dropped_rv = dropped_umi + dropped_umi_template + \
                          dropped_AT + dropped_GC + dropped_adaptor + \
                          too_short_after_trimming
    # Open output file to write to
    with open(out_log, 'w+') as f:
        print("Trimming stats total reads (pair): {}".format(total_reads), file = f)
        print("Trimming stats {} reads have been dropped!".format(dropped_rv), file = f) 
        print( float(dropped_rv), float(total_reads))
        perc2 = '{percent:.2%}'.format(percent= float(dropped_rv) / float(total_reads) )
        print("Trimming stats you just lost about {} of your data".format(perc2), file = f)
        print("Trimming stats reads remaining: {}".format(total_reads - dropped_rv), file = f)
        print("Trimming stats dropped pairs due to incorrect UMI: {}".format(dropped_umi_template), file = f)
        print("Trimming stats dropped pairs due to low quality UMI: {}".format(dropped_umi), file = f)
        print("Trimming stats dropped pairs due to high AT content: {}".format(dropped_AT), file = f)
        print("Trimming stats dropped pairs due to high GC content: {}".format(dropped_GC), file = f)
        print("Trimming stats dropped pairs due to presence of artifacts: {}".format(dropped_adaptor), file = f)
        print("Trimming stats dropped pairs due to being too short: {}".format(too_short_after_trimming), file = f)
    
    f.close()
    # Check that output file was written ok
    #cdef int remaining_reads = (total_reads - dropped_rv)
    #if not fileOk(out_rv) or remaining_reads == 0:
    #    error = "Error doing quality trimming checks of raw reads." \
    #    "\nOutput file not present {}\n".format(out_rw)
    #    logger.error(error)
    #    raise RuntimeError(error)
    
    # Adding stats to QA Stats object
    #qa_stats.input_reads_forward = total_reads
    #qa_stats.input_reads_reverse = total_reads
    #qa_stats.reads_after_trimming_forward = remaining_reads
    #qa_stats.reads_after_trimming_reverse = remaining_reads
