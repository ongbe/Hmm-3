#!/usr/bin/env python

import argparse
import hmm
import numpy as np
import random
import csv
import time
from multiprocessing import Pool
from sklearn.model_selection import KFold

# Wrapper for KFold validation (parallelized)
def cross_validate(args):
    # Run cross validation with config from args
    
    # Parse infolder argument then read
    if(args.i == 'listening'):
        infolder = 'input_sequences/AU06_AU12_KM_5/listening/every_5_frames'
    elif(args.i == 'every_5'):
        infolder = 'input_sequences/AU06_AU12_KM_5/by_answer/every_5_frames'
    elif(args.i == 'by_answer'):
        infolder = 'input_sequences/AU06_AU12_KM_5/by_answer'
    elif(args.i == 'changes_only'):
        infolder = 'input_sequences/AU06_AU12_KM_5/changes_only'
    else:
        infolder = 'input_sequences/AU06_AU12_KM_5/'

    truthHmm = hmm.Hmm()
    bluffHmm = hmm.Hmm()
    truthHmm.read_sequences(infolder + '/truthers')
    bluffHmm.read_sequences(infolder + '/bluffers')
    
    if(len(truthHmm.X_mat_train) == 0 or len(bluffHmm.X_mat_train) == 0):
        print('ERR: No data found, make sure {0} contains truthers/bluffers folders'.format(infolder))
        return    
     
    # Split the sequences into args.n folds for truth and then bluff
    kf = KFold(n_splits=args.n)
    np.random.seed = args.seed
    
    X = truthHmm.X_mat_train
    truthSets = []
    for train, test in kf.split(X):
        trainSet = []
        testSet = []
        for i in train:
            trainSet.append(X[i])
        for i in test:
            testSet.append(X[i])
        #print('Train = ', trainSet)
        #print('Test = ', testSet)   
        truthSets.append([trainSet,testSet])
    
    X = bluffHmm.X_mat_train
    bluffSets = []
    for train, test in kf.split(X):
        trainSet = []
        testSet = []
        for i in train:
            trainSet.append(X[i])
        for i in test:
            testSet.append(X[i])
        #print('Train = ', trainSet)
        #print('Test = ', testSet)   
        bluffSets.append([trainSet,testSet])
    
    func_args = []
    for i in range(len(truthSets)):
        func_args.append([args, truthSets[i], bluffSets[i]])
    
    # Run them all in parallel 
    p = Pool(args.n)
    results = p.map(train_test, func_args)
    
    # Write results to a csv for later graphing
    with open('results.csv', 'a+') as f:
        # TODO: Take an average for the percent correct for the folds
        writer = csv.writer(f)
	total_correct = 0
	total_size = 0
        for r in results:
            total_correct += r[0]
	    total_size += r[1]
        avg_percent = total_correct * 100 / total_size
                   
        writer.writerow([time.ctime(),args.k,'5',args.n_init,args.n_iter,args.seed,\
                          'totalCorrect ->',total_correct,'NFolds ->',\
	                  args.n,'AvgPercent ->',avg_percent,infolder])          
        
    
# A single 'fold' in the KFold validation
def train_test(args):
    #  Parameters  #
    # args[0] is normal args
    # args[1] is [truthTrainSequences, truthTestSequences]
    # args[2] is [bluffTrainSequences, bluffTestSequences]
    n_init = args[0].n_init  # Random initializations to try
    n_iter = args[0].n_iter # Iterations in each initialization
    k = args[0].k # Hidden States
    d = 5 # Outputs (number of clusters used) 
    
    truthHmm = hmm.Hmm()
    bluffHmm = hmm.Hmm()
    
    truthHmm.X_mat_train = args[1][0]
    truthHmm.X_mat_test = args[1][1]
    bluffHmm.X_mat_train = args[2][0]
    bluffHmm.X_mat_test = args[2][1]
    testSize = len(truthHmm.X_mat_test) + len(bluffHmm.X_mat_test)
    
   
    print('# Truth Training Sequences: {0}\n# Bluff Training Sequences: {1}'.format(\
        len(truthHmm.X_mat_train), len(bluffHmm.X_mat_train)))
    print('k = {0}, d = {1}, n_init = {2}, n_iter = {3}, testSize = {4}'.format(\
        k,d,n_init,n_iter,testSize))   
    
    print('Beginning training on Truth-Tellers....')
    bestScore = -np.inf
    # Run em_train for Truth-Tellers multiple times, finding the best-scoring one
    for i in range(n_init):
        truthHmm.initialize_weights(k,d)
        truthHmm.em_train_v(n_iter)
        score = truthHmm.p_X_mat(truthHmm.X_mat_train)
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
        score = bluffHmm.p_X_mat(bluffHmm.X_mat_train)
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
    
    
    # Write weight files for later usage
    truthHmm.write_weight_file('truthers.weights')
    bluffHmm.write_weight_file('bluffers.weights')
    
    # Evaluate on Testing sequences
    # TODO: Would it be helpful to have a separate counter for truth/bluff?
    correct = 0
    for X in truthHmm.X_mat_test:
        if(truthHmm.p_X(X) > bluffHmm.p_X(X)):
            correct += 1
    for X in bluffHmm.X_mat_test:
        if(bluffHmm.p_X(X) > truthHmm.p_X(X)):
            correct += 1
    
    print('Out of {0} test cases, {1} were correctly classified.'.format(\
        testSize, correct))
    
    truthScore = truthHmm.p_X_mat(truthHmm.X_mat_train)
    bluffScore = bluffHmm.p_X_mat(bluffHmm.X_mat_train)
    
    # Return the number correct, testSize to be averaged and written to CSV
    return correct, testSize

#------------------------------------------------------------------------
def run(args):
    """ Trains an HMM on Truth-Tellers and one on Bluffers then 
        writes the weight files to truthers.weights and bluffers.weights
        Uses test data to test classification (if proper HMM scores higher) """
    print('\n...Testing truthers vs bluffers...')
    
    #  Parameters  #
    n_init = args.n_init  # Random initializations to try
    n_iter = args.n_iter # Iterations in each initialization
    k = args.k # Hidden States
    d = 5 # Outputs (number of clusters used)
    testSize = args.testSize # Number seq's to be used in each X_mat_test and not in training
    # If testSize is 15 for example, 15 truthers and 15 blufers will be withheld from training
    seed = args.seed # Random seed so we can recreate runs (for Test vs Train data)
    
    truthHmm = hmm.Hmm()
    bluffHmm = hmm.Hmm()
    
    # Parse infolder argument then read
    if(args.i == 'listening'):
        infolder = 'input_sequences/AU06_AU12_KM_5/listening/every_5_frames'
    elif(args.i == 'every_5'):
        infolder = 'input_sequences/AU06_AU12_KM_5/by_answer/every_5_frames'
    elif(args.i == 'by_answer'):
        infolder = 'input_sequences/AU06_AU12_KM_5/by_answer'
    elif(args.i == 'changes_only'):
        infolder = 'input_sequences/AU06_AU12_KM_5/changes_only'
    else:
        infolder = 'input_sequences/AU06_AU12_KM_5/'
    
    truthHmm.read_sequences(infolder + '/truthers')
    bluffHmm.read_sequences(infolder + '/bluffers')
    
    if(len(truthHmm.X_mat_train) == 0 or len(bluffHmm.X_mat_train) == 0):
        print('ERR: No data found, make sure {0} contains truthers/bluffers folders'.format(infolder))
        return
    
    # Separate Test and Train data
    random.seed(seed)
    truthHmm.X_mat_test = []
    bluffHmm.X_mat_test = []        
    for i in range(testSize):
        truthHmm.X_mat_test.append(truthHmm.X_mat_train.pop(random.randrange(len(
            truthHmm.X_mat_train))))
        bluffHmm.X_mat_test.append(bluffHmm.X_mat_train.pop(random.randrange(len(
            bluffHmm.X_mat_train))))          
    
    print('# Truth Training Sequences: {0}\n# Bluff Training Sequences: {1}'.format(\
        len(truthHmm.X_mat_train), len(bluffHmm.X_mat_train)))
    print('k = {0}, d = {1}, n_init = {2}, n_iter = {3}, seed = {4}'.format(\
        k,d,n_init,n_iter,seed))
    
    print('Beginning training on Truth-Tellers....')
    bestScore = -np.inf
    # Run em_train for Truth-Tellers multiple times, finding the best-scoring one
    for i in range(n_init):
        truthHmm.initialize_weights(k,d)
        truthHmm.em_train_v(n_iter)
        score = truthHmm.p_X_mat(truthHmm.X_mat_train)
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
        score = bluffHmm.p_X_mat(bluffHmm.X_mat_train)
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
    
    print('k = {0}, d = {1}, n_init = {2}, n_iter = {3}, seed = {4}'.format(\
        k,d,n_init,n_iter,seed)) # Print this again for convenience
    
    # Write weight files for later usage
    truthHmm.write_weight_file('truthers.weights')
    bluffHmm.write_weight_file('bluffers.weights')
    
    # Evaluate on Testing sequences
    # TODO: Would it be helpful to have a separate counter for truth/bluff?
    correct = 0
    for X in truthHmm.X_mat_test:
        if(truthHmm.p_X(X) > bluffHmm.p_X(X)):
            correct += 1
    for X in bluffHmm.X_mat_test:
        if(bluffHmm.p_X(X) > truthHmm.p_X(X)):
            correct += 1
    
    print('Out of {0} test cases, {1} were correctly classified.'.format(\
        testSize + testSize, correct))
    
    truthScore = truthHmm.p_X_mat(truthHmm.X_mat_train)
    bluffScore = bluffHmm.p_X_mat(bluffHmm.X_mat_train)
    
    # Write results to text file for easy reading
    with open('results.txt', 'a+') as f:
        f.write('\n\n-----\n')
        f.write('k = {0}, d = {1}, n_init = {2}, n_iter = {3}, seed = {4}'.format(\
                k,d,n_init,n_iter,seed))        
        f.write('\nOut of {0} test cases, {1} were correctly classified.'.format(\
        testSize + testSize, correct))
        f.write('\ntruthHmm score on training data = {0}'.format(truthScore))
        f.write('\nbluffHmm score on training data = {0}'.format(bluffScore))
        f.write(infolder + ' used for input sequences.')

    # Write results to a csv for later graphing
    with open('results.csv', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow([time.ctime(),k,d,n_init,n_iter,seed,correct,testSize*2,\
                         testSize,truthScore,bluffScore,100*correct/(testSize*2),infolder])        

#------------------------------------------------------------------------
if __name__ == '__main__':
    # Setup commandline parser
    help_intro = 'Program to train two HMMs and classify testing sequences as truthers/bluffers.' 
    parser = argparse.ArgumentParser(description=help_intro)

    parser.add_argument('-k',help='k (number hidden states), ex:3',\
                        type=int, default=4)
    parser.add_argument('-n_init', help='Number of random initializations used to train each HMM', \
                        type=int, default=3)
    parser.add_argument('-n_iter', help='Number of iterations for each initialization', \
                        type=int, default=400)
    parser.add_argument('-testSize', help='Number of truthers and bluffers EACH used for Testing', \
                        type=int, default=15)
    parser.add_argument('-seed', help='Random seed used to select Testing sequences', \
                        type=int, default=15)  
    parser.add_argument('-i',help='Input folder (listening, every_5, by_answer, default, changes_only)',\
                        type=str, default='changes_only')    
    parser.add_argument('-n', help='Number of KFold folds', \
                        type=int, default=6)      
    args = parser.parse_args()
    
    print('args: ', args)

    # run(args) # Run and write to results.txt and results.csv (append)
    cross_validate(args)
    
    print('PROGRAM COMPLETE')