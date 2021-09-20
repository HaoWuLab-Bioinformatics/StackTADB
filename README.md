# StackTADB
StackTADB is a stacking-based ensemble learning model for accurately predicting the boundaries of Topologically Associating Domains (TADs) in fruit flies.

## Dataset
The dataset used in the study comes from Henderson et al. Download their study here https://doi.org/10.1093/nar/gkz315

## Overview
The "dm3.kc167.example" is a file containing one-hot-encoded DNA sequences, in which 15057 sequences are positive samples (non-TAD boundaries), and 15070 sequences are negative samples (TAD boundaries).  
The "data prepare.py" is used for data processing.  
The "feature_code.py" and the "psednc.py" is used for feature extraction, containing a total of seven features. 
The "model.py" is used for model training and performance evaluation. We perform ten-fold cross-validation on the training set and evaluate the performance of StackTADB on the independent test set.

## Dependency
Python 3.6  
Numpy  
h5py  
keras  
sklearn

## Usage
First, you should perform data preprocessing, you can run the script as:  
`python data prepare.py`  
Then you should perform feature extraction you need through running the script as:  
`python feature_code.py` or `python psednc.py`  
Finally if you want to compile and run StackTADB, you can run the script as:  
`python model.py`
