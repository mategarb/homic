<img src="imgs/logo_homics.png" width="200">

# Version 2.0.1.

A python package for HOst-MICrobiome (homic) spatial analysis 

# Installation guide

```python
import sys
sys.path.append('/path/to/the/package/folder/src')
from homic import file_readers, simulate_16S, dl_model
```
 
# Folders

* src
    - main scripts of the package
* tests
    - examples
* models
    - precomputed DL models and encoders for the species specified in the list files/65sp.txt
* files
    - collection of small-size input files, including barcodes, reference genomes and species lis
* imgs
    - images for the package, e.g. logo

# Examples

The example scripts for running the homic function, can be found in the "tests" folder.
It contains three examples:

* e1_simulate_data.py
    - example for generating synthetic training and validation data
* e2_run_dl_training.py
    - example for running training to generate deep learning model
* e3_run_dl_evaluation.py
    - benchmarkign the DL model against Kraken2
* e4_run_dl_prediction.py
    - predicting genus/species based on read sequence (for entire .fastq) without using spots information (global performance)