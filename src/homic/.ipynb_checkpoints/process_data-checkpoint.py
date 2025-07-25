#!/usr/bin/env python

import sys
sys.path.append('/proj/berzelius-2024-407/users/x_magar/homic/src') # to load our package
from homic import file_readers, kraken2, process_data

import os
import re
import pysam
import ctypes
import shutil
import numpy as np
import pandas as pd
import statistics
import seaborn as sns
import os
import subprocess
from Bio import Blast
from Bio import SeqIO
from Bio import Entrez
import time
import re

    
def trim_decon(path, dbpath, file1, file2, threads="32", option="paired"):

# 1. trimming
    file11 = file1.replace(".fastq", "_trimmed.fastq")
    file12 = file1.replace(".fastq", "_un_trimmed.fastq")
    file21 = file2.replace(".fastq", "_trimmed.fastq")
    file22 = file2.replace(".fastq", "_un_trimmed.fastq")

    print("Trimming and filtering")
    cmd = ["trimmomatic",
           "PE",
            "-threads",
            threads,
            file1,
            file2,
            file11,
            file12,
            file21,
            file22,
            "LEADING:3",
            "TRAILING:3",
            "SLIDINGWINDOW:4:15",
            "MINLEN:15",]
    subprocess.call(cmd)

    os.remove(file12)
    os.remove(file22)


# 2. decontaminating
    print("Decontaminating reads")

    output = file1.replace("_1.fastq", "_k2")
    
    kraken2.decontaminate_paired(db_path = dbpath, 
                                input_file1 = file11,
                                input_file2 = file21,
                                output = output,
                                threads = threads)
    os.remove(file11)
    os.remove(file21)

def assemble_decon(path, dbpath, file1, file2, threads="16", option="paired"):
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
            "-t",
            threads,]  # Generic metagenomes settings, default

    shutil.rmtree(path + '/out_assembly', ignore_errors=True)
    
    subprocess.call(cmd2)
    
    #os.remove(paths_r1)
    #os.remove(paths_r2)
    
    
# 4. decontaminating II 
    print("Decontaminating contigs")
    file = path + "/out_assembly/final.contigs.fa"

    output = file.replace("out_assembly/final.contigs.fa", "mic_contigs")
    kraken2.decontaminate_single(db_path = dbpath, 
                      input_file = file,
                      output = output,
                      threads = threads)
    
    #shutil.rmtree(path + '/out_assembly', ignore_errors=True)
    print("Done!")


def run_blastn(path_fa, path_db, path_out, nthreads="16", evalue = "1e-6", ofmt="6 qseqid sseqid pident length evalue bitscore score stitle"):

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
           nthreads,
            "-evalue",
            evalue,
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

def read_n_clean_blastn(path_blast, topn = 1, evalue = 0.05, drop_sp = True):
    data = pd.read_csv(path_blast, header = None, delimiter = "\t")
    data = data.rename({0: "contig_id", 1: "subject_id", 2: "pident", 3: "length", 4: "evalue", 5: "bitscore", 6: "score", 7: "subject_title"}, axis='columns')

    # removing reduntant words from titles to get species
    words = data["subject_title"].to_list()
    recs = list(map(clean_word, words))
    all_species = list(map(select_species, recs))
    all_genus = list(map(select_genus, recs))
    
    data["species"] = all_species
    data["genus"] = all_genus
    
    data = data[data['evalue'] < evalue]

    # sorting by two columns: evalue & bitscore
    data = data.sort_values(by=['evalue'], ascending=True)
    data = data.sort_values(by=['bitscore'], ascending=False)

    # remove duplicated hits  
    data = data.drop_duplicates(subset=['contig_id','species'])
    
    # droping undefined species
    if drop_sp:
        rem_und = ["sp." not in spec for spec in data["species"]]
        data = data[rem_und]

    return data
    

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
