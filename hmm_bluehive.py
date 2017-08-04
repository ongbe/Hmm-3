#!/usr/bin/env python3

"""
-------------------------------------------------------------------------------
Classes to implement a Hidden Markov Model (HMM). 
-------------------------------------------------------------------------------
"""
import numpy as np
import csv
import time

import sys        # for sys.argv
import glob
from scipy.stats import multivariate_normal
#import matplotlib.pyplot as plt

import random
import re
import itertools # for product

try:
    from scipy.misc  import logsumexp
except ImportError:
    def logsumexp(x,**kwargs):
        return np.log(np.sum(np.exp(x),**kwargs))

import unittest
import logging
logging.basicConfig(level=logging.DEBUG)
'''
you can set the logging level from a command-line option such as:
  --log=INFO
'''

#============================================================================
class Hmm():
    """ Hidden Markov Model
        Transition T and emission E probabilities (weights) are in log format,
        and are fully populated np arrays (not sparse).
        
    """
    
    MIN_LOG_P = -200
    
    #---------------------------------------------------------------------------
    def __init__(s): 
        """       
        """
        s.k = 0   # number of states
        s.d = 0   # number of outputs
        
        # Emission log probability E_kd[cur_state,output]
        s.E_kd = np.ones((s.k,s.d), dtype=float) 

        # Transition log probability T_kk[cur_state,next_state]
        s.T_kk = np.ones((s.k,s.k), dtype=float)
        
        # Prior log probability P_k[state]
        s.P_k  = np.ones((s.k), dtype=float)    
        
        s.X_mat_test  = [np.zeros(0)] # list of np.array of output sequences
        s.X_mat_train = [np.zeros(0)] # list of np.array of output sequences

        s.Y_mat_test  = [np.zeros(0)] # list of np.array of state sequences
        s.Y_mat_train = [np.zeros(0)] # list of np.array of state sequences
    
    #---------------------------------------------------------------------------
    def __str__(s):
        out = '------------------------------------\n'
        out += 'hmm data: (#states = ' + str(s.k)
        out += ', #outputs = ' + str(s.d) +')'
        out += '\n------------\nT_kk = \n' + str(s.T_kk)
        out += '\n------------\nE_kd = \n' + str(s.E_kd)
        out += '\n------------\nP_k = \n' + str(s.P_k)
        return out


    #---------------------------------------------------------------------------
    def print_percents(s):
        # Prints HMM weights as human-friendly percentages
        np.set_printoptions(suppress=True) # Suppress Sci-Notation 
        out = '------------------------------------\n'
        out += 'NOT IN LOG FORMAT, PROBABILITIES AS PERCENTS\n'
        out += 'hmm data: (#states = ' + str(s.k)
        out += ', #outputs = ' + str(s.d) +')'
        out += '\n------------\nT_kk = \n' + str(np.exp(s.T_kk) * 100)
        out += '\n------------\nE_kd = \n' + str(np.exp(s.E_kd) * 100)
        out += '\n------------\nP_k = \n' + str(np.exp(s.P_k) * 100)    
        np.set_printoptions(suppress=False)
        print(out)
    

    #---------------------------------------------------------------------------
    def initialize_weights(s,k_,d_):
        s.k = k_   # number of states
        s.d = d_   # number of outputs
        
        # Emission log probability E_kd[cur_state,output]
        s.E_kd = np.log(np.random.rand(s.k,s.d))

        # Transition log probability T_kk[cur_state,next_state]
        s.T_kk = np.log(np.random.rand(s.k,s.k))
        
        # Prior log probability P_k[state]
        s.P_k  = np.log(np.random.rand(s.k))
        
        s.log_normalize(s.E_kd)
        s.log_normalize(s.T_kk)
        s.log_normalize(s.P_k)
        
    #--------------------------------------------------------------
    def parse_weight_file(s,filename):
        """
        T and E parameters loaded from filename. Default value entered
        for weight matrix elements not specified.
        """
        logging.debug('parse_weight_file: ' + filename)

        with open(filename) as f:
            first_line = f.readline()
            first_tokens = first_line.split(',')
            k = int(first_tokens[0])
            d = int(first_tokens[1])
            s.initialize_weights(k,d)
            for line in f:
                sline = line.split(" ")
                preamble = sline[0].split("_")
        
                if preamble[0] == 'T':
                    cur_state  = int(preamble[1])
                    next_state = int(preamble[2])
                    s.T_kk[cur_state,next_state] = float(sline[1]) 
        
                elif preamble[0] == 'E':
                    state  = int(preamble[1])
                    output = int(preamble[2])
                    s.E_kd[state,output] = float(sline[1]) 
                
                elif preamble[0] == 'P':
                    state = int(preamble[1])
                    s.P_k[state] = float(sline[1])
                
                else:  
                    assert(True), "ERROR: unknown weight file entry"  
                    
        s.log_normalize(s.T_kk)
        s.log_normalize(s.E_kd)
        s.log_normalize(s.P_k)        
        

    #---------------------------------------------------------------------
    def write_weight_file(s,filename):
        """
        Writes the current weights to the given filename to be 
        read at a later time
        """
        logging.debug('...writing output weight file: ' + filename)

        with open(filename, 'w') as f:
            # k,d (# States, # Outputs)
            f.write(str(s.k) + ',' + str(s.d) + "\n")
            
            # E (Emission Probabilities)
            for i in range(s.k):
                for j in range(s.d):
                    f.write("E_{0}_{1} {2}\n".format(i,j,s.E_kd[i,j]))
            
            # T (Transition Probabilities)
            for i in range(s.k):
                for j in range(s.k):
                    f.write('T_{0}_{1} {2}\n'.format(i,j,s.T_kk[i,j]))
            
            # P (Prior Probabilities)
            for i in range(s.k):
                f.write("P_{0} {1}\n".format(i,s.P_k[i]))
              
    #---------------------------------------------------------------------------
    def parse_data_file(s,filename):
        """ returns np.array of points """
        logging.debug('parse_data_file: ' + filename)
        with open(filename) as f:
            seq = np.array([int(x) for x in f.read().split()])
            f.close()
        return seq
    
        #with open("unittest.seq1") as f:
            #seq = []
            #for line in f:
                ##sline = line.split(', ')
                #sline = re.findall(r'[^,;\s]+', line)
                #assert(len(sline) == 1) # TODO: handle multidim output 
                #seq.append(int(sline[0]))
        #return np.array(seq)    


    #---------------------------------------------------------------------------
    def read_sequences(s,foldername):
        # Reads all sequences from foldername that have 'input' in the name
        # and sets to X_mat_train for use in em_train
        s.X_mat_train = []
        for name in glob.glob(foldername + '/*'):
            if('input' in name): # TODO: This will probably be changed
                s.X_mat_train.append(s.parse_data_file(name))
        #print(s.X_mat_train)
        
    #--------------------------------------------------------------
    def log_normalize(s, M=None):
        """ given an array of log probabilities M, biases all elements equally 
            in order to normalize.
            M should be a 1D or 2D numpy array.
        """
        
        if M is None:
            s.log_normalize(s.P_k)
            s.log_normalize(s.T_kk)
            s.log_normalize(s.E_kd)
        else:
            M[np.isnan(M)] = s.MIN_LOG_P
            if(len(M.shape) == 1):
                Z = logsumexp(M)
                M -= Z            
            else:
                # row-wise probabilities must sum to 1 
                # for rowi in range(M.shape[0]):
                #     z= logsumexp(M[rowi,:])
                #     M[rowi,:] -= z            
                Z = logsumexp(M,axis=1) # sum over row, col is wildcard
                M -= Z[:,np.newaxis]    # broadcast row-wise
            
    #------------------------------------------------------------------
    def p_Z(s, Z_n):        
        """ return log[P(Z)], the log probability of the sequence of 
            states Zn.
        """
        from_indices = list(Z_n[:-1]) # concat
        to_indices = Z_n[1:]        
        p = np.sum(s.T_kk[from_indices,to_indices]) + s.P_k[Z_n[0]]
        return p   
    
    #------------------------------------------------------------------
    def p_XZ(s, X_n, Z_n):
        """ return log[P(X,Y)], the log prob of observations X with states Z """
        # make sure input is in index format
            
        pX = np.sum(s.E_kd[Z_n,X_n])
        p = s.p_Z(Z_n) + pX
        return p                                
            
    #---------------------------------------------------------------------------
    def viterbi_for(s, X):
        """
        Given observation X, and valid E,T, computes most likely Z.
        This is the easy to understand version with for loops. It is slower
        than the numpy vectorized version.
        
        We basically scan over time, for each state, calculating the probability
        of coming from each of the possible previous states. (ie. a trellis).
        
        v_nk[t,q] = log P(o1,o2...ot,q1,q2..qt=q|T,E)
   
        selecting the most probable q's as we progress over t
        """
        
        # Initialize
        v_nk = np.zeros((len(X),s.k)) 
        v_current_k = np.zeros(s.k) 
        bt_nk = np.zeros((len(X),s.k),dtype=int) # backtrace table
        Z_n = np.empty(len(X),dtype=int)
        
        # t = 0
        for state in range(s.k):
            v_nk[0,state] = s.P_k[state] + s.E_kd[state,X[0]]

        # t > 0
        for t in range(1,len(X)):
            for cur_state in range(s.k):
                for last_state in range(s.k):
                    v_current_k[last_state] = v_nk[t-1,last_state] + \
                                              s.T_kk[last_state,cur_state] + \
                                              s.E_kd[cur_state,X[t]]
                v_nk[t,cur_state] = np.amax(v_current_k)   # the max value
                bt_nk[t,cur_state] = np.argmax(v_current_k) # index of max value
 
        # t = last
        t = len(X) - 1
        v_max, v_max_q = np.amax(v_nk[t,:]), np.argmax(v_nk[t,:])
        
        # backtrace
        Z_n[t] = v_max_q
        for t in range(len(X)-2,-1,-1):
            Z_n[t] = bt_nk[t+1,Z_n[t+1]]

        return Z_n, v_max
 
    #---------------------------------------------------------------------------
    def viterbi_v(s, X):
        """
        Given observation X, and valid E,T, computes most likely Z.
        This is the easy to understand version with for loops. It is slower
        than the numpy vectorized version.
        
        We basically scan over time, for each state, calculating the probability
        of coming from each of the possible previous states. (ie. a trellis).
        
        v_nk[t,q] = log P(o1,o2...ot,q1,q2..qt=q|T,E)
   
        selecting the most probable q's as we progress over t
        """
        
        # Initialize
        v_nk = np.zeros((len(X),s.k)) 
        v_current_k = np.zeros(s.k) 
        bt_nk = np.zeros((len(X),s.k),dtype=int) # backtrace table
        Z_n = np.empty(len(X),dtype=int)
        
        # t = 0
        v_nk[0,:] = s.P_k + s.E_kd[:,X[0]]

        # t > 0
        for t in range(1,len(X)):
            v_t = s.T_kk + s.E_kd[:,X[t]] + v_nk[t-1,:,np.newaxis]
            v_nk[t,:], bt_nk[t,:] = np.amax(v_t,axis=0), np.argmax(v_t,axis=0)
 
        # t = last
        t = len(X) - 1
        v_max, v_max_q = np.amax(v_nk[t,:]), np.argmax(v_nk[t,:])
        
        # backtrace
        Z_n[t] = v_max_q
        for t in range(len(X)-2,-1,-1):
            Z_n[t] = bt_nk[t+1,Z_n[t+1]]

        return Z_n, v_max    
    
   
    #---------------------------------------------------------------------------
    def forward_for(s, X):
        """
            Given observation X, and valid E,T, returns the forward probability
            alpha = a, in matrix form.
    
            a_nk[time t,tag j] = logsum P(o1,o2...ot,qt=j|T,E)
        """
        
        # Initialize
        a_nk = np.zeros((len(X),s.k)) 
        a_current_k = np.zeros(s.k) 
                
        # t = 0
        for state in range(s.k):
            a_nk[0,state] = s.P_k[state] + s.E_kd[state,X[0]]

        # t > 0
        for t in range(1,len(X)):
            for cur_state in range(s.k):
                for last_state in range(s.k):
                    a_current_k[last_state] = a_nk[t-1,last_state] + \
                                              s.T_kk[last_state,cur_state] + \
                                              s.E_kd[cur_state,X[t]]
                a_nk[t,cur_state] = logsumexp(a_current_k)   # the max value

        return a_nk
    
    #---------------------------------------------------------------------------
    def forward_v(s, X):
        """
            Given observation X, and valid E,T, returns the forward probability
            alpha = a, in matrix form.
    
            a_nk[time t,tag j] = logsum P(o1,o2...ot,qt=j|T,E)
        """
        
        # Initialize
        a_nk = np.zeros((len(X),s.k)) 
        a_current_k = np.zeros(s.k) 
                
        # t = 0
        a_nk[0,:] = s.P_k[:] + s.E_kd[:,X[0]]

        # t > 0
        for t in range(1,len(X)):       
            a_current_k = s.T_kk + s.E_kd[:,X[t]] + a_nk[t-1,:,np.newaxis]
            a_nk[t,:] = logsumexp(a_current_k,axis=0)            

        return a_nk
    
    #---------------------------------------------------------------------------
    def backward_for(s, X):
        """  
            Given observation X, and valid E,T, returns the backward probability
            beta = b.
            b[t,j] = logsum P(ot+1, ot+2... oT |T,E, tth hidden state = j)
            b[t,j] = logsumexp(b[t+1,:] + T[j,:] + E[X[t+1],:])
        """       
        
        # Initialize
        b_nk = np.zeros((len(X),s.k)) 
        b_current_k = np.zeros(s.k) 
                
        for t in range(len(X)-2,-1,-1):
            for cur_state in range(s.k):
                for next_state in range(s.k):
                    b_current_k[next_state] = b_nk[t+1,next_state] + \
                                              s.T_kk[cur_state,next_state] + \
                                              s.E_kd[next_state,X[t+1]]
                b_nk[t,cur_state] = logsumexp(b_current_k)   # the max value

        return b_nk
    
    #---------------------------------------------------------------------------
    def backward_v(s, X):
        """  
            Given observation X, and valid E,T, returns the backward probability
            beta = b.
            b[t,j] = logsum P(ot+1, ot+2... oT |T,E, tth hidden state = j)
            b[t,j] = logsumexp(b[t+1,:] + T[j,:] + E[:,X[t+1]])
        """       
        
        # Initialize
        b_nk = np.zeros((len(X),s.k)) 
        b_current_k = np.zeros(s.k)

        # Starting at T-1 and going backwards...
        for t in range(len(X)-2,-1,-1):
            b_current_k = b_nk[np.newaxis,t+1] + s.T_kk + s.E_kd[np.newaxis,:,X[t+1]]
            b_nk[t] = logsumexp(b_current_k, axis=1)   # the max value

        return b_nk
    
    def em_train_alt(s, n_iter):
        for _ in range(n_iter):
            gamma, epsilon = s.e_step_alt()
            s.m_step_alt(gamma, epsilon)
    
    def e_step_alt(s):
        n = len(s.X_mat_train)
        gamma = np.zeros(shape=(n,s.k)) # Looks like a Y in the book
        epsilon = np.zeros(shape=(n,s.k,s.k)) # Looks like an E in the book
        
        for X in s.X_mat_train:
            a_nk = s.forward_alt(X)
            b_nk = s.backward_alt(X)
            m = len(X)
            # Gamma 
            for t in range(m):
                for j in range(s.k):
                    den = np.log(np.exp(a_nk[t,:]).sum())
                    gamma[t,j] += np.exp(a_nk[t,j] + b_nk[t,j] - den)
                    # y[t,j] = alpha[t,j] * beta[t,j] / P(O|A,B)
                    # P(O|A,B) = alpha[T,X[T]] says the book
                    
            # Epsilon
            for t in range(m-1):
                for i in range(s.k):
                    for j in range(s.k):
                        den = np.exp(a_nk[t,:]).sum() + np.exp(s.T_kk[:,:]).sum() \
                            + np.exp(b_nk[t+1,:]).sum() + np.exp(s.E_kd[:,X[t+1]]).sum()
                        epsilon[t,i,j] += np.exp(a_nk[t,i] + s.T_kk[i,j] \
                                                + s.E_kd[j,X[t+1]] + b_nk[t+1,j] \
                                                - np.log(den))
                        # e[t,i,j] = (forward(t,i) * transition(i,j) * emmision(j,X[t+1]) 
                        #                * backwards[t+1,j] / forward(???)
                    
            return np.log(gamma), np.log(epsilon)
        
    def m_step_alt(s, gamma, epsilon):
        n = len(s.X_mat_train)
        # T
        for i in range(s.k):
            for j in range(s.k):
                num, den = 0,0
                for t in range(n-1):
                    num += np.exp(epsilon[t,i,j])
                    den += np.exp(gamma[t,i])
                s.T_kk[i,j] = np.log(num/den)
        
        # E
        for j in range(s.k):
            for k in range(s.d):
                num, den = 0,0
                for X in s.X_mat_train:
                    for t in range(len(X)):                    
                        if(X[t] == k):
                            num += np.exp(gamma[t,j])
                        den += np.exp(gamma[t,j])
                s.E_kd[j,k] = np.log(num/den)
        
        # P
        s.P_k = gamma[0]
        
        s.log_normalize(M=None)
        
    def forward_alt(s,X):
        # Initialize
        a_nk = np.zeros((len(X),s.k)) 
        
        # t = 0
        for state in range(s.k):
            a_nk[0,state] = s.P_k[state] + s.E_kd[state,X[0]]
            
        # t > 0
        for t in range(len(X) - 1):
            temp = 0
            for state in range(s.k):
                for last_state in range(s.k):
                    temp += np.exp(s.T_kk[last_state,state] + a_nk[t,last_state] \
                                   + s.E_kd[state,X[t+1]])
                a_nk[t+1,state] = np.log(temp)
        
        return a_nk
    
    def backward_alt(s,X):
        b_nk = np.zeros((len(X),s.k)) 
        
        for state in range(s.k):
            b_nk[len(X)-1,state] = 0
        
        for t in range(len(X)-2,-1,-1):
            for prev_state in range(s.k):
                temp = 0                
                for state in range(s.k):
                    temp += np.exp(b_nk[t+1,state] + s.T_kk[prev_state,state] \
                                   + s.E_kd[state,X[t+1]])
                b_nk[t,prev_state] = np.log(temp)
        
        return b_nk
     
    
    #---------------------------------------------------------------------------
    def e_step(s):
        """ calculate state occupancy count [s.gamma_k] and
                      state transition count [s.eta_kk]
            based on 
            parameters [s.E_kd, s.T_kk, s.P_k] 
            
            sum_counts_Y_k = gamma_k[i] is the expected count of (Y = i)
            sum_counts_T_kk = eta_kk[i,j] is the expected count of i to j trans
        """
        sum_counts_P_k  = np.zeros_like(s.P_k, dtype=float)
        sum_counts_E_kd = np.zeros_like(s.E_kd, dtype=float)
        sum_counts_T_kk = np.zeros_like(s.T_kk, dtype=float)

        # for each sequence in training set
        for X in s.X_mat_train: 
            a_nk = s.forward_v(X)
            b_nk = s.backward_v(X)            
            #b_nk = s.backward_for(X)            
            counts_Y_nk = np.exp(a_nk + b_nk) # convert out of logspace
            counts_Y_nk /= np.exp(a_nk[len(X)-1,X[len(X)-1]]) # Y /= P(O | A,B) 
            # P(O|A,B) = a_nk[T,X[T]] according to the book
            
            sum_counts_P_k += counts_Y_nk[0]
           
            for t in range(len(X)-1):
                for cur in range(s.k):
                    for to in range(s.k):
                        sum_counts_T_kk[cur,to] += np.exp( \
                                                    a_nk[t][cur] + \
                                                    b_nk[t+1][to] + \
                                                    s.E_kd[to,X[t+1]] + \
                                                    s.T_kk[cur,to])
                        
                for state in range(s.k):
                    sum_counts_E_kd[state,X[t]] += counts_Y_nk[t,state]
            for state in range(s.k):
                sum_counts_E_kd[state,X[-1]] += counts_Y_nk[-1,state]
                
        sum_counts_T_kk /= np.exp(a_nk[len(X)-1,X[len(X)-1]]) # T /= a[T,X[T]]
            
        return sum_counts_P_k, sum_counts_E_kd, sum_counts_T_kk

    #---------------------------------------------------------------------------
    def e_step_v(s):
        """ calculate state occupancy count [s.gamma_k] and
                      state transition count [s.eta_kk]
            based on 
            parameters [s.E_kd, s.T_kk, s.P_k] 
    
            sum_counts_Y_k = gamma_k[i] is the expected count of (Y = i)
            sum_counts_T_kk = eta_kk[i,j] is the expected count of i to j trans
        """
        sum_counts_P_k  = np.zeros_like(s.P_k, dtype=float)
        sum_counts_E_kd = np.zeros_like(s.E_kd, dtype=float)
        sum_counts_T_kk = np.zeros_like(s.T_kk, dtype=float)
    
        # for each sequence in training set
        for X in s.X_mat_train: 
            a_nk = s.forward_v(X)
            b_nk = s.backward_v(X)            
            counts_Y_nk = np.exp(a_nk + b_nk) # convert out of logspace
            
            counts_Y_nk /= np.exp(a_nk[len(X)-1,X[len(X)-1]]) # Y /= P(O | A,B)             
            sum_counts_P_k += counts_Y_nk[0]
    
            for t in range(len(X)-1):
                for cur in range(s.k):
                    sum_counts_T_kk[cur,:] += np.exp( \
                        a_nk[t][cur] + \
                        b_nk[t+1][:] + \
                        s.E_kd[:,X[t+1]] + \
                        s.T_kk[cur,:])
                sum_counts_E_kd[:,X[t]] += counts_Y_nk[t,:]
            
            sum_counts_E_kd[:,X[-1]] += counts_Y_nk[-1,:]
    
        sum_counts_T_kk /= np.exp(a_nk[len(X)-1,X[len(X)-1]]) # T /= a[T,X[T]]
        
        return sum_counts_P_k, sum_counts_E_kd, sum_counts_T_kk        
        
    
    #---------------------------------------------------------------------------
    def m_step(s, counts):
        """ calculate parameters [s.E_kd, s.T_kk, s.P_k] 
            based on 
            responsibilities [s.Resp_nk]
        """
        smoothing_count = np.e**s.MIN_LOG_P
        
        sum_counts_P_k, sum_counts_E_kd, sum_counts_T_kk = counts
        s.P_k = np.log(sum_counts_P_k + smoothing_count)
        s.T_kk = np.log(sum_counts_T_kk + smoothing_count)
        s.E_kd = np.log(sum_counts_E_kd + smoothing_count)
    
        s.log_normalize()
        
    #---------------------------------------------------------------------------
    def em_train(s, n_iter):
        """ train parameters using em algorithm """
        for i in range(n_iter):
            counts = s.e_step()
            s.m_step(counts)
    
    #---------------------------------------------------------------------------
    def em_train_v(s, n_iter):
        """ train parameters using vectorized em algorithm"""
        for i in range(n_iter):
            counts = s.e_step_v()
            s.m_step(counts)
   
    #---------------------------------------------------------------------------
    def mle_train(s, smoothing_count=None):
        """ Calculates parameters: [s.E_kd, s.T_kk, s.P_k] 
            given:                 [s.X_mat_train, s.Y_mat_train]
            
            Each of the probabilities is determined solely by counts. After 
            counting, probabilities are normalized and converted to log.
        """
        if smoothing_count==None:
            smoothing_count = np.e**s.MIN_LOG_P
            
        assert(len(s.X_mat_train) == len(s.Y_mat_train), \
              "ERROR: bad len(s.Y_mat_train)")
        
        counts_P_k  = np.zeros_like(s.P_k, dtype=float)
        counts_T_kk = np.zeros_like(s.T_kk, dtype=float)
        counts_E_kd = np.zeros_like(s.E_kd, dtype=float)

        counts_P_k  += smoothing_count
        counts_T_kk += smoothing_count
        counts_E_kd += smoothing_count
        
        # Main loop for counting
        for X,Y in zip(s.X_mat_train, s.Y_mat_train):    
            counts_P_k[Y[0]] += 1
            for t in range(len(X)-1):
                counts_T_kk[Y[t],Y[t+1]] += 1
                counts_E_kd[Y[t],X[t]] += 1
            counts_E_kd[Y[-1],X[-1]] += 1           
    
        s.P_k = np.log(counts_P_k)
        s.T_kk = np.log(counts_T_kk)
        s.E_kd = np.log(counts_E_kd)
    
        s.log_normalize(s.P_k) 
        s.log_normalize(s.T_kk)
        s.log_normalize(s.E_kd)              
        
    #---------------------------------------------------------------------------    
    """ Generates n observation sequences of length m randomly based on valid P,T,E """
    def generate_sequences(s, n, m):
        hidden_sequences = np.zeros((n,m), dtype=int) # Generated hidden sequences
        obs_sequences = np.zeros((n,m), dtype=int) # Generated observation sequences
        
        for i in range(n): # For each sequence to be generated
            
            # First, use P_k and T_kk to generate a hidden state sequence
            state_seq = np.zeros(m, dtype=int) # Curr generated hidden state sequence
            # t = 0
            at = 0 # aggregator for random selection
            rand = np.random.rand() # random seed of when to stop
            for j in range(s.k): # Go thru states until we reach the rand_seed
                at += np.exp(s.P_k[j])
                if(rand <= at): # Reached the random seed, use this as selection
                    state_seq[0] = j # Selected start state
                    break
            # t > 0
            for t in range(1, m):
                at = 0 # aggregator for random selection
                rand = np.random.rand()  # random seed of when to stop                 
                for j in range(s.k): # Go thru states until we reach the rand_seed
                    at += np.exp(s.T_kk[state_seq[t-1],j])
                    if(rand <= at): # Reached the random seed, use this as selection
                        state_seq[t] = j
                        break
            
            # Next, Generate observations from hidden state sequence using E_kd
            obs_seq = np.zeros(m, dtype=int) # Current observation sequence
            for t in range(0, m):
                at = 0
                rand = np.random.rand()
                for j in range(s.d):
                    at += np.exp(s.E_kd[state_seq[t],j])
                    if(rand <= at):
                        obs_seq[t] = j
                        break

            #print(state_seq, '-->', obs_seq)
            hidden_sequences[i] = state_seq
            obs_sequences[i] = obs_seq
            
        return hidden_sequences, obs_sequences # Returns the generated sequences

    #---------------------------------------------------------------------------    
    """ Returns the score of the HMM from the Viterbi on X_mat_train """ 
    def score(s):
        p = 0;
        for X in s.X_mat_train:
            p += s.viterbi_v(X)[1]
        return p
    
    def score_test(s):
        p = 0;
        for X in s.X_mat_test:   
            p += s.viterbi_v(X)[1]
        return p        
    
    
#============================================================================
class TestHmm(unittest.TestCase):
    """ Self testing of each method """
    
    @classmethod
    def setUpClass(s):
        """ runs once before ALL tests """
        print("\n...........unit testing class Hmm..................")

    def setUp(s):
        """ runs once before EACH test """
        pass

    @unittest.skip
    def test_init(s):
        print("\n...testing init(...)")
        hmm = Hmm()
        hmm.initialize_weights(2,3)
        hmm.log_normalize(hmm.T_kk)
        hmm.log_normalize(hmm.E_kd)
        hmm.log_normalize(hmm.P_k)
        
        print(hmm)

    @unittest.skip
    def test_parse_weight_file(s):
        print("\n...testing parse_weight_file(...)")
        hmm = Hmm()
        hmm.parse_weight_file('unittest.weights')
        print(hmm)
        
    @unittest.skip
    def test_parse_data_file(s):
        print("\n...testing parse_data_file(...)")
        hmm = Hmm()
        seq = hmm.parse_data_file('unittest.seq1')
        print(seq)
    
    @unittest.skip    
    def test_forward_for(s):
        print("\n...testing forward_for(...)")

        hmm = Hmm()
        hmm.initialize_weights(2,3)
        hmm.P_k = np.log(np.array([1,0]))
        hmm.T_kk = np.log(np.array([[0, 1],[1./3, 2./3]]))
        hmm.E_kd = np.log(np.array([[1./3, 1./3, 1./3], \
                                    [0.50, 0.50, 0.00]]))
        hmm.log_normalize()
    
        X1 = np.array([0])
    
        a_nk1 = hmm.forward_for(X1)
        s.assertAlmostEqual(np.log(1./3),a_nk1[0,0])
        s.assertAlmostEqual(-np.inf,a_nk1[0,1])
        
        X2 = np.array([0,1])
        a_nk2 = hmm.forward_for(X2)
        
        # use brute force to check all permutations for a better score
        Zpermutations = list(itertools.product(range(hmm.k), repeat=len(X2)))
        p_a = []
        for Z in Zpermutations:
            p_a.append(hmm.p_XZ(X2,Z))
        p_tot_brute_force = logsumexp(p_a)
        p_tot_alpha = logsumexp(a_nk2[-1])
        s.assertEqual(p_tot_brute_force, p_tot_alpha)
        
        s.assertAlmostEqual(np.log(1./3), a_nk2[0,0])
        s.assertAlmostEqual(-np.inf,a_nk2[0,1])
        s.assertAlmostEqual(-np.inf, a_nk2[1,0])
        s.assertAlmostEqual(np.log(1./3) + np.log(1./2), a_nk2[1,1])
        
        X3 = np.array([0,1,1])
        a_nk3 = hmm.forward_for(X3)
        s.assertAlmostEqual(np.log(1./3), a_nk3[0,0])
        s.assertAlmostEqual(-np.inf,a_nk3[0,1])
        s.assertAlmostEqual(-np.inf, a_nk3[1,0])
        s.assertAlmostEqual(np.log(1./3) + np.log(1./2), a_nk3[1,1])

        P_010 = hmm.p_XZ(X3,[0,1,0])
        P_011 = hmm.p_XZ(X3,[0,0,0])
        s.assertAlmostEqual(logsumexp([P_010,P_011]), a_nk3[2,0])
        
        hmm.parse_weight_file('unittest.weights')
        hmm.log_normalize()
        
        X_n = hmm.parse_data_file('unittest.seq1')        
        a_nk = hmm.forward_for(X_n)
        print(a_nk)
        # use brute force to check all permutations for a better score
        Zpermutations = list(itertools.product(range(hmm.k), repeat=len(X_n)))
        p_a = []
        for Z in Zpermutations:
            p_a.append(hmm.p_XZ(X_n,Z))
        p_tot = logsumexp(p_a)
         
    @unittest.skip    
    def test_backward_for(s):
        print("\n...testing backward_for(...)")

        hmm = Hmm()
        hmm.initialize_weights(2,3)
        hmm.P_k = np.log(np.array([1,0]))
        hmm.T_kk = np.log(np.array([[0, 1],[1./3, 2./3]]))
        hmm.E_kd = np.log(np.array([[1./3, 1./3, 1./3], \
                                        [0.50, 0.50, 0.00]]))
        hmm.log_normalize()
    
        X1 = np.array([0])
    
        b_nk1 = hmm.backward_for(X1)
        s.assertAlmostEqual(0, b_nk1[0,0])
        s.assertAlmostEqual(0, b_nk1[0,1])
    
        X2 = np.array([0,1])
        b_nk2 = hmm.backward_for(X2)        
        P_01 = np.log(1./2)
        P_00 = -np.inf        
        s.assertAlmostEqual(logsumexp([P_00,P_01]), b_nk2[0,0])
        P_10 = np.log(1./3) + np.log(1./3)
        P_11 = np.log(2./3) + np.log(1./2)        
        s.assertAlmostEqual(logsumexp([P_10,P_11]), b_nk2[0,1])
        s.assertAlmostEqual(0, b_nk2[1,0])
        s.assertAlmostEqual(0, b_nk2[1,1])        
        
        a_nk2 = hmm.forward_for(X2)
        y_nk2 = a_nk2 + b_nk2
        Zpermutations = list(itertools.product(range(hmm.k), repeat=len(X2)))
        p_a = []
        for Z in Zpermutations:
            p_a.append(hmm.p_XZ(X2,Z))
        p_tot_brute_force = logsumexp(p_a)
        
        p_tot_alpha_beta = logsumexp(y_nk2[0])
        s.assertEqual(p_tot_brute_force, p_tot_alpha_beta)        
        
        hmm.parse_weight_file('unittest.weights')
        X_n = hmm.parse_data_file('unittest.seq1')        
        print(hmm.backward_for(X_n))
        
    @unittest.skip    
    def test_forward_v(s):
        print("\n...testing forward_v(...)")
        hmm = Hmm() # Set up
        hmm.parse_weight_file('unittest.weights')
        X_n = hmm.parse_data_file('unittest.seq1')
        
        # Not vectorized
        t = time.time()
        withFor = hmm.forward_for(X_n)
        unvectorziedTime = time.time() - t
        
        # Vectorized
        t = time.time()
        withV =  hmm.forward_v(X_n)
        vectorziedTime = time.time() - t
        
        print("UnVectorized Time: ", unvectorziedTime)           
        print("Vectorized Time: ", vectorziedTime) 
        print("Speedup = ", unvectorziedTime / vectorziedTime, "x")
        
        if(np.allclose(withV, withFor)): # Check if they're ~equal
            print('Equal! Vectorized and UnVectorized are equal!')
        else:
            print('Not Equal! Vectorized and UnVectorized are not equal!')
            print("V = ", withV)
            print("For = ", withFor)
            s.fail(msg="Failure! test_forward test failed")
          
    @unittest.skip 
    def test_backward_v(s):
        print("\n...testing backward_v(...)")
        hmm = Hmm() # Set up
        hmm.parse_weight_file('unittest.weights')
        X_n = hmm.parse_data_file('unittest.seq1')
        
        # Not vectorized
        t = time.time()
        withFor = hmm.backward_for(X_n)
        unvectorziedTime = time.time() - t
        
        # Vectorized
        t = time.time()
        withV =  hmm.backward_v(X_n) 
        vectorziedTime = time.time() - t
        
        print("UnVectorized Time: ", unvectorziedTime)           
        print("Vectorized Time: ", vectorziedTime) 
        print("Speedup = ", unvectorziedTime / vectorziedTime, "x")
        
        if(np.allclose(withV, withFor)): # Check if they're ~equal
            print('Equal! Vectorized and UnVectorized are equal!')
        else:
            print('Not Equal! Vectorized and UnVectorized are not equal!')
            print("V = ", withV)
            print("For = ", withFor) 
            s.fail(msg="Failure! test_backward_v failed")
            
    @unittest.skip
    def test_mle_train(s):
        print("\n...testing mle_train(...)")
        hmm = Hmm()
        hmm.initialize_weights(2,3)
        X1 = np.array([0,0,1,1,2])
        Y1 = np.array([0,1,0,1,0])
        hmm.X_mat_train = [X1]
        hmm.Y_mat_train = [Y1]
        hmm.mle_train()
        s.assertTrue(np.allclose(np.e**hmm.P_k, np.array([1,0])))
        s.assertTrue(np.allclose(np.e**hmm.T_kk, np.array([[0, 1],
                                                           [1, 0]])))
        s.assertTrue(np.allclose(np.e**hmm.E_kd, 
                                 np.array([[  1./3, 1./3, 1./3],
                                           [  0.50, 0.50, 0.00]])))
   
    #@unittest.skip
    def test_generate_sequences(s):
        # Generates sequences based on model, runs em_train with random models        
        # Parameters #
        n = 300 # Generate n sequences
        m = 300 # Length of each is m
        n_init = 10 # Number of different random initializations
        n_iter = 250 # Number of iterations for each diff initialization
        k = 2 # Number states
        d = 2 # Number different observations
        
        # Generate random sequences from known parameters
        hmm = Hmm()
        hmm.k = k
        hmm.d = d
        hmm.initialize_weights(hmm.k, hmm.d)
        
        #NOT RANDOM anymore
        hmm.T_kk = np.log(np.array([[0.2,0.8],[0.9,0.1]]))
        hmm.E_kd = np.log(np.array([[0.2,0.8],[0.9,0.1]]))
        hmm.P_k = np.log(np.array([0.2,0.8]))
        #hmm.T_kk = np.log(np.array([[1,0],[1,0]]))
        #hmm.E_kd = np.log(np.array([[1,0],[0,1]]))
        #hmm.P_k = np.log(np.array([0.2,0.8]))
        
        hmm.log_normalize(M=None)
        
        sequences = hmm.generate_sequences(n,m)
        print('\nSource HMM weights (expected to converge to this):')        
        hmm.print_percents()
        hmm.X_mat_train = sequences[1]
        print('Original Score (used to generate sequences): ', hmm.score())
        print('\nGenerated Obs-Sequences to be used in em_train: \n', sequences[1])
        
        best = -np.inf
        avgScore = 0
        contains_zero = 0
        # Train from random weight initialization
        for i in range(n_init): # Number of Initializations
            hmm2 = Hmm()
            hmm2.initialize_weights(hmm.k, hmm.d)            
            hmm2.X_mat_train = sequences[1] # Just need the observation sequences
            #hmm2.em_train(n_iter)  # Number of iterations in EM-Train
            hmm2.em_train_alt(n_iter)  # Number of iterations in EM-Train
            
            
            score = hmm2.score()
		
            print('\nTrained HMM weights: #{0}   Score = {1}'.format(i+1,score))
            #print("Trained Score: ", score)
            if(score > best):
                best = score
                bestWeights = [hmm2.P_k, hmm2.T_kk, hmm2.E_kd]
            avgScore += score
            for p in hmm2.P_k:
                if(np.allclose(p,0,atol=0.03)):
                    contains_zero += 1
       
        # Rebuild the best one
        bestHmm = Hmm()
        bestHmm.P_k, bestHmm.T_kk, bestHmm.E_kd = bestWeights
        bestHmm.k, bestHmm.d = k, d

        
        print("\nOriginal (used to generate sequences):")
        hmm.print_percents()
        print("\nBest Trained:")
        bestHmm.print_percents()
        print("\nBest Score: ", best)
        print("Score from Original: ", hmm.score())
        print("Average Score: ", avgScore / n_init)
        print("n = {0}, m = {1}, n_iter = {2}, n_init = {3}".format(n,m,n_iter,n_init))
        
        #s.assertTrue(np.allclose(hmm.P_k, hmm2.P_k) and \
                     #np.allclose(hmm.T_kk, hmm2.T_kk) and \
                     #np.allclose(hmm.E_kd, hmm2.E_kd), \
                     #msg="Did not converge!")
        print(contains_zero, 'out of', n_init, 'contained ~100% in P')
       
    @unittest.skip
    def test_em_train(s):
        print("\n...testing em_train(...)")
        hmm = Hmm()
        hmm.initialize_weights(2,2)
        #hmm.E_kd = np.array([[0,-200, -200],[-200,0,-200],[-200,-200,0]])
        #hmm.T_kk = np.array([[-200,0,-200],[-200,-200,0],[0,-200, -200]])
        #hmm.P_k = np.array([0,-200,-200])
        
        X1 = np.array([0,1])
        X2 = np.array([1,0])
        hmm.P_k = np.log(np.array([0.5,0.5]))
        hmm.X_mat_train = [X1,X2]
        #hmm.X_mat_train = [X1]
        
        # Read sequences from folder and set hmm.X_mat_train to them
        # hmm.read_sequences('input_sequences') # TODO: Change the folder name?
        hmm.em_train(500)
        #print(hmm)
        hmm.print_percents() # Print HMM as human-friendly percents
        
        print('\nHidden Sequences Found with Viterbi:\nObserved --> Hidden   p')
        for seq in hmm.X_mat_train:
            v = hmm.viterbi_v(seq)
            print(seq, '   --> ', v[0], np.exp(v[1])*100, '%')
            #total *= v[1]
       
        #TODO: add assert to check that weights ended up in a good place
        print('Avg p: ', hmm.score())
    
    @unittest.skip
    def test_em_train_v(s):
        # Tests using e_step_v from same initialization to see if same results
        print("\n...testing em_train_v...")
        
        # Parameters #
        n = 20 # Generate n sequences
        m = 10 # Length of each is m
        # n_init = 10 # Number of different random initializations
        n_iter = 20 # Number of iterations for each diff initialization
        k = 5 # Number states
        d = 5 # Number different observations        
        
        # Generate random sequences from known parameters to be used in training
        hmm = Hmm()
        hmm.k = k
        hmm.d = d
        hmm.initialize_weights(hmm.k, hmm.d)
        sequences = hmm.generate_sequences(n,m)
        print('\nSource HMM weights (expected to converge to this):')        
        hmm.print_percents()
        hmm.X_mat_train = sequences[1]
        print('Original Score (used to generate sequences): ', hmm.score())
        print('\nGenerated Obs-Sequences to be used in em_train: \n', sequences[1])
        
        # HMM to use Non-Vectorized em_train
        hmm_for = Hmm()
        hmm_for.k, hmm_for.d = k,d
        hmm_for.initialize_weights(k,d)
        hmm_for.X_mat_train = sequences[1]
        
        # HMM to use Vectorized em_train_v
        hmm_v = Hmm()
        hmm_v.k, hmm_v.d = k,d
        # Copy the weights from hmm_for so they start at the same point
        hmm_v.P_k, hmm_v.T_kk, hmm_v.E_kd = hmm_for.P_k, hmm_for.T_kk, hmm_for.E_kd
        hmm_v.X_mat_train = sequences[1]
        
        # Run training on each
        print("Training Non-Vectorized...")
        t = time.time()        
        hmm_for.em_train(n_iter) # Non-Vectorized
        for_time = time.time() - t
        print("Training Vectorized...")
        t = time.time()
        hmm_v.em_train_v(n_iter) # Vectorized
        v_time = time.time() - t
        
        # Print Speedup info
        print('Non-Vectorized time: ', for_time)
        print('Vectorized time: ', v_time)
        print('Speedup = ', for_time / v_time, 'x')
        
        # Check if they got the same results
        assertTrue = np.allclose(hmm_for.P_k, hmm_v.P_k)
        assertTrue &= np.allclose(hmm_for.T_kk, hmm_v.T_kk)
        assertTrue &= np.allclose(hmm_for.E_kd, hmm_v.E_kd)
        
        # TODO: Assert
        print("\nSuccess = ", assertTrue) 
        if(assertTrue):
            print('The Vectorized and Non-Vectorized both trained to the same weights.')
        else:
            print("!THEY DID NOT TRAIN TO THE SAME WEIGHTS: em_train and em_train_v NOT EQUAL!")
        # s.assertTrue(assertTrue, msg='Vectorized and non-vectorized got different results')
             
    @unittest.skip
    def test_viterbi_for(s):
        print("\n...testing viterbi_for(...)")
        hmm = Hmm()
        hmm.parse_weight_file('unittest.weights')
        X_n = hmm.parse_data_file('unittest.seq1')
        Z_n, v_max = hmm.viterbi_for(X_n)
        p = hmm.p_XZ(X_n,Z_n)
        print("Z:",Z_n, "v_max:",v_max,"p:",p)
        s.assertAlmostEqual(p, v_max, 3)
        
        # use brute force to check all permutations for a better score
        Zpermutations = list(itertools.product(range(hmm.k), repeat=len(X_n)))
        p_max = -np.inf
        for Z in Zpermutations:
            p = hmm.p_XZ(X_n,Z)
            p_max = max(p,p_max)
            print(Z, "\tp:", p)

        s.assertAlmostEqual(p_max, v_max, 3)        

    @unittest.skip
    def test_viterbi_v(s):
        print("\n...testing viterbi_v(...)")
        
        hmm = Hmm() # Set up
        hmm.parse_weight_file('unittest.weights')
        X_n = hmm.parse_data_file('unittest.seq1')
       
        # With For loops
        t = time.time()
        Z_n_for, v_max_for = hmm.viterbi_for(X_n)
        p_for = hmm.p_XZ(X_n,Z_n_for)
        forTime = time.time() - t
       
        # Vectorized version
        t = time.time()
        Z_n, v_max = hmm.viterbi_v(X_n)
        p = hmm.p_XZ(X_n,Z_n)     
        vTime = time.time() - t   

        print("Z_for:",Z_n_for, "v_max_for:",v_max_for,"p_for:",p_for)
        print("Z_vec:",Z_n, "v_max_vec:",v_max,"p_vec:",p)             
        
        print("UnVectorized Time: ", forTime) 
        print("Vectorized Time: ", vTime) 
        print("Speedup = ", forTime / vTime, "x")
        
        if(not all(Z_n == Z_n_for)):
            s.fail(msg="Failure! test_viterbi_v failed")
        s.assertAlmostEqual(p, p_for, places=3,)
        s.assertAlmostEqual(v_max, v_max_for, places=3)
    
    @unittest.skip                 
    def test_write_weights(s):
        print("\n...testing write_weight_file(...)")
        hmm = Hmm() # Set up
        hmm.parse_weight_file('unittest.weights')
        #X_n = hmm.parse_data_file('unittest.seq1')
        for i in range(hmm.k):
            print("P_{0} = {1}".format(i,hmm.P_k[i]))
        
        hmm.write_weight_file('unittest.weights.output')
    
    @unittest.skip
    def test_truth_bluff_train(s):
        """ Trains an HMM on Truth-Tellers and one on Bluffers then 
            writes the weight files to truthers.weights and bluffers.weights """
        print('\n...Testing truthers vs bluffers...')
        
        # Parameters #
        n_init = 3
        n_iter = 40
        k = 5
        d = 5
        
        truthHmm = Hmm()
        bluffHmm = Hmm()
        
        truthHmm.read_sequences('input_sequences/truthers')
        bluffHmm.read_sequences('input_sequences/bluffers')
        truthHmm.X_mat_train = truthHmm.X_mat_train[:30] # Only first 30 for now
        bluffHmm.X_mat_train = bluffHmm.X_mat_train[:30]
        
        print('# Truth Sequences: {0}\n# Bluff Sequences: {1}'.format(\
            len(truthHmm.X_mat_train), len(bluffHmm.X_mat_train)))
        print('k = {0}, d = {1}\nn_init = {2}, n_iter = {3}'.format(k,d,n_init,n_iter))
        
        print('Beginning training on Truth-Tellers....')
        bestScore = -np.inf
        # Run em_train for Truth-Tellers multiple times, finding the best-scoring one
        for i in range(n_init):
            truthHmm.initialize_weights(k,d)
            truthHmm.em_train_v(n_iter)
            score = truthHmm.score()
            if(score > bestScore):
                bestScore = score
                bestWeights = truthHmm.P_k, truthHmm.T_kk, truthHmm.E_kd
            truthHmm.print_percents()
            print('Trained truthHmm #',i+1,' Score = ',score)
        # Rebuild the best truthHmm
        truthHmm.P_k, truthHmm.T_kk, truthHmm.E_kd = bestWeights
        
        print('Best Trained Truth-Tellers HMM:')
        truthHmm.print_percents()
        
        print('Beginning training on Bluffers....')        
        bestScore = -np.inf # Reset for bluffers
        # Run em_train for Bluffers multiple times, finding the best-scoring one     
        for i in range(n_init):
            bluffHmm.initialize_weights(k,d)
            bluffHmm.em_train_v(n_iter)
            score = bluffHmm.score()
            if(score > bestScore):
                bestScore = score
                bestWeights = bluffHmm.P_k, bluffHmm.T_kk, bluffHmm.E_kd
            bluffHmm.print_percents()
            print('Trained bluffHmm #',i+1,' Score = ',score)
        # Rebuild the best bluffHMM
        bluffHmm.P_k, bluffHmm.T_kk, bluffHmm.E_kd = bestWeights    
        
        print('\nBest Trained Truth-Tellers HMM:')
        truthHmm.print_percents()
        print('\nBest Trained Liars HMM:')
        bluffHmm.print_percents()
        
        # Test if they fit their own data better than the other
        bluffHmm.X_mat_test = truthHmm.X_mat_train
        truthHmm.X_mat_test = bluffHmm.X_mat_train
        
        bluffScoreOnTruth = bluffHmm.score_test()
        truthScoreOnBluff = truthHmm.score_test()
        # Below two should be higher than the above two
        # Meaning that they fit their own data better than the others, thus there 
        # are differences between facial features of liars and truthers
        bluffScore = bluffHmm.score()
        truthScore = truthHmm.score()

        # In this order because #truthSeqs != #bluffSeqs so can't compare fairly
        # Can only really compare this way with even # of sequences used
        print('TruthOnTruth: ', truthScore)    
        print('BluffOnTruth: ', bluffScoreOnTruth)
        print('BluffOnBluff: ', bluffScore)        
        print('TruthOnBluff: ', truthScoreOnBluff)
        print('k = {0}, d = {1}, n_init = {2}, n_iter = {3}'.format(k,d,n_init,n_iter))
        
        
        # Write weight files for later usage
        truthHmm.write_weight_file('truthers.weights')
        bluffHmm.write_weight_file('bluffers.weights')        
    
    def tearDown(s):
        """ runs after each test """
        pass
    
    @classmethod
    def tearDownClass(s):
        print("\n...........unit testing of class Hmm complete..............\n")
        


#===============================================================================
if __name__ == '__main__':
    if (len(sys.argv) > 1):        
        unittest.main()
    else:
        unittest.main()
