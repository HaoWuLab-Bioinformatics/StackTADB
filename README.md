# StackTADB
StackTADB is a stacking-based ensemble learning model for accurately predicting the boundaries of Topologically Associating Domains (TADs) in fruit flies.

## Dataset
The dataset used in the study comes from Henderson et al. Download their study here https://doi.org/10.1093/nar/gkz315

## Overview
The "dm3.kc167.example" is a file containing one-hot-encoded DNA sequences, in which 15057 sequences are positive samples (non-TAD boundaries), and 15070 sequences are negative samples (TAD boundaries).
The "data prepare.py" is used for data processing and feature extraction, and we will obtain a kmers feature matrix named "k-mer_array_k=6.txt" with the top 600 counts.
The "model.py"  is used for model training and performance evaluation.  We use ten-fold cross-validation to evaluate the performance of StackTADB.

## Dependency
Python 3.6
Numpy
h5py
keras
sklearn

## Usage
If you want to compile and run "data prepare.py", you can run the script as:  
`python data prepare.py`  
If you want to compile and run "model.py", you can run the script as:  
`python model.py`
