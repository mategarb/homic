#!/usr/bin/env python3

import argparse
import bz2
import concurrent.futures
import collections
import ftplib
import glob
import gzip
import hashlib
import http.client
import inspect
import io
import itertools
import json
import logging
import lzma
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import ete3
from Bio.Seq import Seq
from os import path
from subprocess import Popen, PIPE
import pandas as pd
from io import StringIO
import numpy as np
LOG = None
SCRIPT_PATHNAME = None



def wrapper_args_to_binary_args(opts, argv, binary_args):
    for k, v in vars(opts).items():
        if k not in WRAPPER_ARGS_TO_BIN_ARGS:
            continue
        if WRAPPER_ARGS_TO_BIN_ARGS[k] not in binary_args:
            continue
        if v is False:
            continue
        if v is None:
            continue
        if v is True:
            argv.append(WRAPPER_ARGS_TO_BIN_ARGS[k])
        else:
            argv.extend([WRAPPER_ARGS_TO_BIN_ARGS[k], str(v)])


def find_kraken2_binary(name):
    # search the OS PATH
    if "PATH" in os.environ:
        for dir in os.environ["PATH"].split(":"):
            if os.path.exists(os.path.join(dir, name)):
                return os.path.join(dir, name)
    # search for binary in the same directory as wrapper
    script_parent_directory = get_parent_directory(SCRIPT_PATHNAME)
    if os.path.exists(os.path.join(script_parent_directory, name)):
        return os.path.join(script_parent_directory, name)
    # if called from within kraken2 project root, search the src dir
    project_root = get_parent_directory(script_parent_directory)
    if "src" in os.listdir(project_root) and name in os.listdir(
        os.path.join(project_root, "src")
    ):
        return os.path.join(project_root, os.path.join("src", name))
    # not found in these likely places, exit
    LOG.error("Unable to find {:s}, exiting\n".format(name))
    sys.exit(1)


def get_parent_directory(pathname):
    if len(pathname) == 0:
        return None
    pathname = os.path.abspath(pathname)
    if len(pathname) > 1 and pathname[-1] == os.path.sep:
        return os.path.dirname(pathname[:-1])
    return os.path.dirname(pathname)


def find_database(database_name):
    database_path = None
    if database_name.find(os.path.sep) < 0:
        if "KRAKEN2_DB_PATH" in os.environ:
            for directory in os.environ["KRAKEN2_DB_PATH"].split(":"):
                if os.path.exists(os.path.join(directory, database_name)):
                    database_path = os.path.join(directory, database_name)
                    break
        else:
            if database_name in os.listdir(os.getcwd()):
                database_path = database_name
    elif os.path.exists(database_name):
        database_path = database_name
    if database_path:
        for db_file in ["taxo.k2d", "hash.k2d", "opts.k2d"]:
            if not os.path.exists(os.path.join(database_path, db_file)):
                return None
    return database_path


def remove_files(filepaths, forked=False):
    total_size = 0

    for fname in filepaths:
        if not os.path.exists(fname):
            continue
        elif os.path.isdir(fname):
            with os.scandir(fname) as iter:
                directories = []
                for entry in iter:
                    if entry.is_dir():
                        directories.append(entry.path)
                    else:
                        filepaths.append(entry.path)
                if not forked and len(directories) >= 4:
                    total_size += remove_files_parallel(directories)
                else:
                    filepaths.extend(directories)
        else:
            LOG.info("Removing {}\n".format(fname))
            total_size += os.path.getsize(fname)
            os.remove(fname)

    if not forked:
        for fname in filepaths:
            if os.path.isdir(fname):
                shutil.rmtree(fname)

    return total_size


def remove_files_parallel(filepaths):
    total_size = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = []
        for fname in filepaths:
            if not os.path.exists(fname):
                continue
            future = pool.submit(remove_files, [fname], True)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            total_size += future.result()

    return total_size


def check_seqid(seqid):
    taxid = None
    match = re.match(r"(?:^|\|)kraken:taxid\|(\d+)", seqid)
    if match:
        taxid = match.group(1)
    elif re.match(r"^\d+$", seqid):
        taxid = seqid
    if not taxid:
        match = re.match(r"(?:^|\|)([A-Z]+_?[A-Z0-9]+)(?:\||\b|\.)", seqid)
        if match:
            taxid = match.group(1)
    return taxid


def hash_string(string):
    md5 = hashlib.md5()
    md5.update(string.encode())
    return md5.hexdigest()


def hash_file(filename, buf_size=8192):
    LOG.info("Calculating MD5 sum for {}\n".format(filename))
    md5 = hashlib.md5()
    with open(filename, "rb") as in_file:
        while True:
            data = in_file.read(buf_size)
            if not data:
                break
            md5.update(data)
    digest = md5.hexdigest()
    LOG.info("MD5 sum of {} is {}\n".format(filename, digest))
    return digest


### classification function modified to run in homic ###
def classify(db_path, input_file, confidence=0.01, threads=8, min_hit_gr=2):
    """Classifies reads to genus / species according to db.

        Parameters
        ----------
        db_path : string,
            a path to kraken db
        input_file : string,
            a path to input .fastq file
        confidence : float,
            kraken2 parameter - confidence (-T)
        threads : intiger,
            kraken2 parameter - number of threads (-p)
        min_hit_gr : intiger,
            kraken2 parameter - minimum hitting group (-g)

        Returns
        -------
        output
            a data frame with following columns 'outcome', 'seqid', 'taxid', 'seqlen' and 'kmers'
    """

    db_path = os.path.realpath(db_path)
    current_dir = path.join(path.dirname(__file__))
    classify_bin = os.path.join(current_dir + "/kraken2_src", "classify") # localize bins
    database_path = find_database(db_path)
    
    if database_path is None:
        LOG.error("This is not a valid database... exiting".format(db_path))
        sys.exit(1)
    if confidence < 0 or confidence > 1:
        LOG.error("Confidence must be between 0 and 1 inclusive".format(confidence))
        sys.exit(1)
        
    argv = [classify_bin,
        "-H",
        os.path.join(database_path, "hash.k2d"),
        "-t",
        os.path.join(database_path, "taxo.k2d"),
        "-o",
        os.path.join(database_path, "opts.k2d"),
        "-T",
        str(confidence),
        "-p",
        str(threads),
        "-g",
        str(min_hit_gr)]

    argv.append(input_file) # add input file at the end

    p = Popen(argv, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    rc = p.returncode

    print(err.decode())
    info = err.decode()
    
    output = output.decode()
    output = pd.read_csv(StringIO(output), delimiter="\t", header=None)
    output.columns = ['outcome', 'seqid', 'taxid', 'seqlen', 'kmers']

    return output
    
def prepare_db(db_path, ref_path):
    """Builds db for kraken2.

        Parameters
        ----------
        db_path : string,
            a path to the folder where kraken db will be created
        ref_path : string,
            a path to the input .fasta file with reference sequences
            
        Returns
        -------
        no output
            files are saved to the folder under "db_path"
    """
    
    current_dir = path.join(path.dirname(__file__))
    classify_bin = os.path.join(current_dir + "/kraken2_install", "kraken2-build")

    if not os.path.isdir(os.path.join(db_path + "/taxonomy")):    
        # 1. Downloady taxa library
        cmd = [classify_bin,
            "--download-taxonomy",
            "--db",
            db_path,]
        subprocess.call(cmd)
    if not os.path.isdir(os.path.join(db_path + "/library")):     
        # 2. Add fasta
        cmd2 = [classify_bin,
            "--add-to-library",
            ref_path,
            "--db",
            db_path,]
        subprocess.call(cmd2)
    # 3. Build db
    cmd3 = [classify_bin,
        "--build",
        "--db",
        db_path,]
    subprocess.call(cmd3)

def decontaminate_paired(db_path, input_file1, input_file2, output, confidence=0.5, threads=12, min_base_qual=22):
    """Decontamination with kraken2 for paired .fastq files.

        Parameters
        ----------
        db_path : string,
            a path to kraken db
        input_file1 : string,
            a path to the first .fastq file
        input_file2 : string,
            a path to the second .fastq file
        output : string,
            a path to the output .fastq file where host reads are removed
        confidence : float,
            kraken2 parameter - confidence (--confidence)
        threads : intiger,
            kraken2 parameter - number of threads (--threads)
        min_base_qual : intiger,
            kraken2 parameter - minimum base quality (--minimum-base-quality)
            
        Returns
        -------
        no output, files are saved under "output"
    """
    db_path = os.path.realpath(db_path)
    current_dir = path.join(path.dirname(__file__))
    classify_bin = os.path.join(current_dir + "/kraken2_install", "kraken2")
    
    database_path = find_database(db_path)
    
    if database_path is None:
        LOG.error("This is not a valid database... exiting".format(db_path))
        sys.exit(1)
    if confidence < 0 or confidence > 1:
        LOG.error("Confidence must be between 0 and 1 inclusive".format(confidence))
        sys.exit(1)


    cmd = [classify_bin,
        "--db",
        db_path,
        "--threads",
        str(threads),
        "--confidence",
        str(confidence),
        "--minimum-base-quality",
        str(min_base_qual),
        "--unclassified-out",
        output + "#.fastq",
        "--paired",
        input_file1,
        input_file2,
        ]

    subprocess.call(cmd, stdout=subprocess.DEVNULL)

def decontaminate_single(db_path, input_file, output, confidence=0.5, threads=8, min_base_qual=22):
    """Decontamination with kraken2 for single .fastq (unpaired).

        Parameters
        ----------
        db_path : string,
            a path to kraken db
        input_file : string,
            a path to the .fastq file
        output : string,
            a path to the output .fastq file where host reads are removed
        confidence : float,
            kraken2 parameter - confidence (--confidence)
        threads : intiger,
            kraken2 parameter - number of threads (--threads)
        min_base_qual : intiger,
            kraken2 parameter - minimum base quality (--minimum-base-quality)
            
        Returns
        -------
        no output, files are saved under "output"
    """

    db_path = os.path.realpath(db_path)
    current_dir = path.join(path.dirname(__file__))
    classify_bin = os.path.join(current_dir + "/kraken2_install", "kraken2")
    
    database_path = find_database(db_path)
    
    if database_path is None:
        LOG.error("This is not a valid database... exiting".format(db_path))
        sys.exit(1)
    if confidence < 0 or confidence > 1:
        LOG.error("Confidence must be between 0 and 1 inclusive".format(confidence))
        sys.exit(1)
        
    cmd = [classify_bin,
        "--db",
        db_path,
        "--threads",
        str(threads),
        "--confidence",
        str(confidence),
        "--minimum-base-quality",
        str(min_base_qual),
        "--unclassified-out",
        output + ".fastq",
        "--single",
        input_file,
        ]
    
    subprocess.call(cmd, stdout=subprocess.DEVNULL)

def evaluate_kraken(krk_path, gs_path):

    """Evaluates kraken2 prediction with gold standard.

        Parameters
        ----------
        krk_path : string,
            a path to kraken2 result (.csv)
        gs_path : string,
            a path to the gold stanard file (.txt)

        Returns
        -------
            a value of accuracy
    """
    
    k2outc = pd.read_csv(krk_path)
    rows_nams = pd.read_csv(gs_path, sep=' ', header = None,usecols=[1], engine='python', names = ['fastq']) # full header only
    info = pd.read_csv(gs_path, sep='[ ,:,|]', header = None, usecols=[5, 6, 7, 9, 10], names = ['tile', 'x', 'y','taxa1', 'taxa2'], engine='python') # new 
    
    info['truth_taxa'] = info['taxa1'] + ' ' + info['taxa2']
    
    ncbi = ete3.NCBITaxa()
    taxids = ncbi.get_name_translator(info["truth_taxa"])
    taxidsf = sum(list(taxids.values()), []) # flatten the list
    taxids_dict = dict(zip(taxids.keys(), taxidsf)) ## swapping values with keys
    
    info["taxids_full_gs"] = [*map(taxids_dict.get, info["truth_taxa"].tolist())] # mapping between species and tax ids, to get tax ids only
    info["taxids_full_k2"] = k2outc["taxid"]
    
    info = info[info['taxids_full_k2'] != 0]
    info = info[np.invert(pd.isnull(info['taxids_full_gs'].tolist()))]
    
    y_true = info["taxids_full_gs"].tolist()
    y_pred = info["taxids_full_k2"].tolist()
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    return accuracy
    

def format_bytes(size):
    current_suffix = "B"
    for suffix in ["kB", "MB", "GB", "TB", "PB", "EB"]:
        if size >= 1024:
            current_suffix = suffix
            size /= 1024
        else:
            break
    return "{:.2f}{:s}".format(size, current_suffix)


def clean_up(filenames):
    LOG.info("Removing the following files: {}\n".format(filenames))
    # walk the directory tree to get the size of the individual files
    # sum them up to get the usage stat
    space_freed = format_bytes(remove_files(filenames))
    LOG.info(
        "Cleaned up {} of space\n".format(space_freed)
    )

#
#def clean_db(args):
#    os.chdir(args.db)
#    if args.pattern:
#        clean_up(glob.glob(args.pattern, recursive=False))
#    else:
#        clean_up(
#            [
#                "data",
#                "library",
#                "taxonomy",
#                "seqid2taxid.map",
#                "prelim_map.txt",
#            ]
#        )
#
#         ]
#        )

