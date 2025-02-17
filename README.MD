<img src="imgs/logo_homics.png" width="200">

# homic v2.0.1.

A python package for HOst-MICrobiome (homic) spatial analysis 

# installation guide

```python
import sys
sys.path.append('/path/to/the/package/folder/src')
from homic import file_readers, simulate_16S, dl_model
```

# examples

The example scripts for running the homic function, can be found in test folder.
It contains three examples:

* e1_simulate_data.py
    - example for generating synthetic training and validation data
* e2_run_dl_training.py
    - example for running training to generate deep learning model
* e3_run_dl_evaluation.py
    - benchmarkign the DL model against Kraken2
