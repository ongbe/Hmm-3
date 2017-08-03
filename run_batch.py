#!/usr/bin/env python3
from __future__ import print_function
from __future__ import division
from __future__ import with_statement
"""
-------------------------------------------------------------------------------
Classes to implement a Hidden Markov Model (HMM). 
-------------------------------------------------------------------------------
"""
import hmm

import numpy as np
import csv

import sys        # for sys.argv
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.DEBUG)




#------------------------------------------------------------------
def train(train_files, param_outfile, log_outfile):
    """ loads the training_files, initializes an Hmm, runs training,
        writes trained paramters to param_file, logP vs iter written 
        to log_file.
    """

    h = hmm.Hmm()
    
    X1 = h.parse_data_file('example/test.seq')

    h.X_mat_train = [X1]
    k = 3
    h.initialize_weights(k, X1.max() + 1)
    scores = h.em_train(50)
    print(hmm)
    h.write_weight_file(param_outfile)

    # load train sequence
    

    # train 
    
    # write train params
    
    # load sequence file


#------------------------------------------------------------------
def predict(param_file, test_files):
    
    # load predict sequences
    
    
    pass

#===============================================================================
if __name__ == '__main__':
    train('a','test.weights','c')
    
    print('run_batch.py COMPLETE')
    