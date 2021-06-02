# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 16:57:05 2021

@author: frede
"""

import pickle
#import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict
from sklearn import svm
import time
import os
#import sys

tf.compat.v1.disable_eager_execution()

# path to locate data
dirname = os.path.dirname(__file__)
train_path = os.path.join(dirname, 'traindata/').replace("\\","/")



# =============================================================================
# Helper functions for data processing
# Note: All helper functions require a specific data format (unpickle, e.g., "Right_Hand.pickle" to see which format specifically)
# =============================================================================

# standardize a single channel
def standardize_single_channel(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma

# standardize all channels (i.e., six per hand) for selected indices (train or test sets)
# exclude first two and last column (containing the candidate identifier, class affiliation and total performance time)
def standardize(data, indices):
    for j in range(2, data.shape[1] - 1):
        data[[i for i in range(len(data[:,0])) if data[i,0] in indices],j] = standardize_single_channel(data[[i for i in range(len(data[:,0])) if data[i,0] in 
                                                                                                   indices],j])
    return data

#helper funtion for segment extraction: select n random segments of a specified size 
def random_windows(n, data, size):
    start = 0
    values = []
    while start < n:
        lower = np.random.randint(low = 0, high = len(data) - size + 1, size = 1)
        if lower in values:
            continue
        yield int(lower), int(lower + size)
        start += 1
        
#extract n random segments of a specified size
def random_signal(n, data, window_size = 70):
    """
    n : number of segments to extract per participant
    data : movement data in matrix form. 
            Column 1 and 2 contain the candidate identifier and the class affiliation, respectively. The last column contains the number of total time points 
            registered for the respective participant.
    window_size : segment size.

    Returns
    -------
    segments : segment vector of shape [total number of segments {i.e. n x number of participants}, window_size, number of channels]
    targets : group affiliation of segments

    """
    data_org = data
    segments = np.empty((0,window_size,data.shape[1]-3))
    targets = np.empty((0))
    while(len(data_org) > 0):
        number_of_timepoints = int(data_org[0,data.shape[1] - 1])
        data = data_org[:number_of_timepoints,:]
        for (start, end) in random_windows(n, data, window_size):
            new_segment = []
            for j in range(data.shape[1] - 3):
                new_segment.append(data[:, (j+2)][start:end])
            if(len(data[:,1][start:end]) == window_size):
                segments = np.vstack([segments,np.dstack(new_segment)])
                targets = np.append(targets,stats.mode(data[:,1][start:end])[0][0]) #select target that appears most frequently in segment
        data_org = np.delete(data_org, np.s_[:number_of_timepoints], axis = 0)
    return segments, targets



#72 movement patterns are split into test and training set by target (i.e., class affiliation). 
#6 of each of the two targets (0 and 1) determine the test set.
def train_test_indices():
    #load arbitrary movement data
    pickle_in = open(train_path + "Right_Hand.pickle", "rb")
    target_values = pickle.load(pickle_in)[:,1][::4970] #first column contains the class affiliation per time point; each participant was tracked for 4970 time points
    
    control_indices = np.where(np.asarray(target_values) == 0)[0]
    experimental_indices = np.where(np.asarray(target_values) == 1)[0]    
    control_test_indices = np.random.choice(control_indices, size = 6, replace = False)
    control_train_indices = np.delete(control_indices, [control_indices.tolist().index(i) for i in control_test_indices])
    experimental_test_indices = np.random.choice(experimental_indices, size = 6, replace = False)
    experimental_train_indices = np.delete(experimental_indices, [experimental_indices.tolist().index(i) for i in experimental_test_indices])
    train_indices = sorted(control_train_indices.tolist() + experimental_train_indices.tolist()) #lists mit '+' verbinden
    test_indices = sorted(control_test_indices.tolist() + experimental_test_indices.tolist())
    
    return train_indices, test_indices
    


# =============================================================================
# Setup Neural Networks (CNN and FF). All other methods are standard scikit implementations (see below)
# =============================================================================

#parameters for neural network
num_targets = 2
num_hidden = 100 

#parameters for optimizing
learning_rate = 0.0001
training_epochs = 10


def build_cnn(num_channels,window_size = 70):
    kernel_size = int(window_size/2)
    #kernel_size = 35
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(70, kernel_size = kernel_size, input_shape = (window_size, num_channels), data_format = 'channels_last',
                                  kernel_initializer = 'TruncatedNormal',
                                  bias_initializer = 'Constant',
                                  activation = 'relu'))
    model.add(keras.layers.MaxPool1D(pool_size = 3, strides = 2, padding = 'valid'))
    model.add(keras.layers.Conv1D(70, kernel_size = 10, data_format = 'channels_last',
                                  kernel_initializer = 'TruncatedNormal',
                                  bias_initializer = 'Constant',
                                  activation = 'relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(num_hidden, activation = 'tanh'))
    model.add(keras.layers.Dense(num_targets, activation = 'softmax'))
    #model.add(keras.layers.Dense(3, activation = 'linear'))
    #print(model.summary()) für Info zu Outputgrößen etc
    model.compile(optimizer = tf.optimizers.Adam(learning_rate = learning_rate), loss ='sparse_categorical_crossentropy',metrics =['accuracy'])
    #model.compile(optimizer = tf.optimizers.Adam(learning_rate = learning_rate), loss ='mse',metrics =['accuracy'])
    return model

#build simple ff
def build_ff(input_dim):
    ff = keras.Sequential()    
    ff.add(keras.layers.Dense(100, input_dim = input_dim, activation ='relu')) #mit Min,Max
    ff.add(keras.layers.Dense(100, activation='relu'))
    ff.add(keras.layers.Dense(2,activation ='softmax'))  
    ff.compile(optimizer ='adam', loss ='sparse_categorical_crossentropy',metrics =['accuracy'])
    return ff

# =============================================================================
# Evaluations for Whole Session Insertion
# =============================================================================

def evaluate_whole_sessions(feature):
    """
    feature: Either "Right_Hand", "Left_Hand" or "Both_Hands" 

    Returns
    -------
    accuracy rates for LDA, FF, RF, SVM(LIN), SVM(RBF) and CNN. 

    """
    try:
        pickle_in = open(train_path + feature + ".pickle", "rb")
        df = pickle.load(pickle_in)
    except: 
        print("Invalid Body Part entered. Must be 'Right_Hand', 'Left_Hand' or 'Both_Hands' ")
    
    train_indices, test_indices = train_test_indices()
    
    df = standardize(df, train_indices)
    df = standardize(df, test_indices)
    df_res, targets = random_signal(1, df, window_size = 4970)
    
    
    pca_train = np.zeros((len(np.array(train_indices)),df_res.shape[2]*4)) #4 pca components per channel
    pca_test = np.zeros((len(np.array(test_indices)),df_res.shape[2]*4))
    for i in range(df_res.shape[2]):
        pca = PCA(n_components = 4)
        pca.fit(df_res[:,:,i][train_indices,:])
        pca_train[:,(4*i):(4*(i+1))] = pca.transform(df_res[:,:,i][train_indices,:])
        pca_test[:,(4*i):(4*(i+1))] = pca.transform(df_res[:,:,i][test_indices,:])


    y_train = targets[train_indices]
    y_test = targets[test_indices]
    
    
    #LDA, FF and RF for PCA
    clf_pca = LinearDiscriminantAnalysis()
    _ = clf_pca.fit(pca_train, y_train)
        
    ff_pca = keras.wrappers.scikit_learn.KerasRegressor(build_fn = build_ff, input_dim = pca_train.shape[1], batch_size=32, epochs =15)
    _ = ff_pca.fit(pca_train,y_train)
    
    rf_pca = RandomForestClassifier(n_estimators = 1000, random_state = 42)
    _ = rf_pca.fit(pca_train, np.reshape(y_train, (len(y_train),)))
    
    svm_pca = svm.SVC(kernel = 'linear')
    _ = svm_pca.fit(pca_train, y_train)
    
    svrbf_pca = svm.SVC(kernel = 'rbf')
    _ = svrbf_pca.fit(pca_train, y_train)
    
    pred_lda_pca = clf_pca.predict(pca_test)
    number_of_correct_pred_pca_lda = np.sum(np.transpose(pred_lda_pca) == np.transpose(y_test))
    #accuracy_lda_pca = 100/len(y_test) *np.sum(np.transpose(pred_lda_pca) == np.transpose(y_test))
    #print(accuracy_lda_pca)
    pred_ff_pca = ff_pca.predict(pca_test)
    predictions_ff_pca = [np.argmax(pred_ff_pca[i]) for i in range(len(y_test))]
    number_of_correct_pred_pca_ff = np.sum(np.transpose(predictions_ff_pca) == np.transpose(y_test))
    #accuracy_ff_pca = 100/len(y_test) *np.sum(np.transpose(predictions_ff_pca) == np.transpose(y_test))
    #print(accuracy_ff_pca)
    pred_rf_pca = rf_pca.predict(pca_test)
    number_of_correct_pred_pca_rf = np.sum(np.transpose(pred_rf_pca) == np.transpose(y_test))
    #accuracy_rf_pca = 100/len(y_test) *np.sum(np.transpose(pred_rf_pca) == np.transpose(y_test))   
    #print(accuracy_rf_pca)
    pred_svm_pca = svm_pca.predict(pca_test)
    number_of_correct_pred_pca_svm = np.sum(np.transpose(pred_svm_pca) == np.transpose(y_test))
    #accuracy_svm_pca = 100/len(y_test)*np.sum(np.transpose(pred_svm_pca) == np.transpose(y_test))
    #print(accuracy_svm_pca)
    pred_svrbf_pca = svrbf_pca.predict(pca_test)
    number_of_correct_pred_pca_svrbf = np.sum(np.transpose(pred_svrbf_pca) == np.transpose(y_test))
    #accuracy_svrbf_pca = 100/len(y_test)*np.sum(np.transpose(pred_svrbf_pca) == np.transpose(y_test))
    #print(accuracy_svrbf_pca)
    
    
    means = np.mean(df_res, axis = 1)
    stds = np.std(df_res, axis = 1)
    mins = np.min(df_res, axis = 1)
    maxs = np.max(df_res, axis = 1)
    df_bf = np.concatenate((means, stds, mins, maxs), axis = 1)
    bf_train = df_bf[train_indices]
    bf_test = df_bf[test_indices]
    

    clf_bf = LinearDiscriminantAnalysis()
    _ = clf_bf.fit(bf_train, y_train)
    
    ff_bf = keras.wrappers.scikit_learn.KerasRegressor(build_fn = build_ff, input_dim = bf_train.shape[1], batch_size=32, epochs =15)
    _ = ff_bf.fit(bf_train,y_train)
    
    rf_bf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
    _ = rf_bf.fit(bf_train, np.reshape(y_train, (len(y_train),)))
    
    svm_bf = svm.SVC(kernel = 'linear')
    _ = svm_bf.fit(bf_train, y_train)
    
    svrbf_bf = svm.SVC(kernel = 'rbf')
    _ = svrbf_bf.fit(bf_train, y_train)
    
    pred_lda_bf = clf_bf.predict(bf_test)
    number_of_correct_pred_bf_lda = np.sum(np.transpose(pred_lda_bf) == np.transpose(y_test))
    #accuracy_lda_bf = 100/len(y_test) *np.sum(np.transpose(pred_lda_bf) == np.transpose(y_test))
    #print(accuracy_lda_bf)
    pred_ff_bf = ff_pca.predict(bf_test)
    predictions_ff_bf = [np.argmax(pred_ff_bf[i]) for i in range(len(y_test))]
    number_of_correct_pred_bf_ff = np.sum(np.transpose(predictions_ff_bf) == np.transpose(y_test))
    #accuracy_ff_bf = 100/len(y_test) *np.sum(np.transpose(predictions_ff_bf) == np.transpose(y_test))
    #print(accuracy_ff_bf)
    pred_rf_bf = rf_bf.predict(bf_test)
    number_of_correct_pred_bf_rf = np.sum(np.transpose(pred_rf_bf) == np.transpose(y_test))
    #accuracy_rf_bf = 100/len(y_test) *np.sum(np.transpose(pred_rf_bf) == np.transpose(y_test))
    #print(accuracy_rf_bf)
    pred_svm_bf = svm_bf.predict(bf_test)
    number_of_correct_pred_bf_svm = np.sum(np.transpose(pred_svm_bf) == np.transpose(y_test))
    #accuracy_svm_bf = 100/len(y_test)*np.sum(np.transpose(pred_svm_bf) == np.transpose(y_test))
    #print(accuracy_svm_bf)
    pred_svrbf_bf = svrbf_bf.predict(bf_test)
    number_of_correct_pred_bf_svrbf = np.sum(np.transpose(pred_svrbf_bf) == np.transpose(y_test))
    #accuracy_svrbf_bf = 100/len(y_test)*np.sum(np.transpose(pred_svrbf_bf) == np.transpose(y_test))
    #print(accuracy_svrbf_bf)
    
    cnn_train = df_res[train_indices,:,:]
    cnn = keras.wrappers.scikit_learn.KerasClassifier(build_fn = build_cnn, num_channels = cnn_train.shape[2], epochs = 10, window_size = 4970)
    _ = cnn.fit(cnn_train, targets[train_indices])
    
    cnn_test = df_res[test_indices,:,:]
    pred_cnn = (cnn.predict(cnn_test) > 0.5).astype("int32")
    number_of_correct_pred_cnn = np.sum(np.transpose(pred_cnn) == np.transpose(targets[test_indices]))
    #accuracy_cnn = 100/len(targets[test_indices])*np.sum(np.transpose(pred_cnn) == np.transpose(targets[test_indices]))
    
    return [number_of_correct_pred_bf_lda, number_of_correct_pred_pca_lda, number_of_correct_pred_bf_ff, number_of_correct_pred_pca_ff, number_of_correct_pred_bf_rf, number_of_correct_pred_pca_rf, number_of_correct_pred_bf_svm, number_of_correct_pred_pca_svm,number_of_correct_pred_bf_svrbf, number_of_correct_pred_pca_svrbf, number_of_correct_pred_cnn]
    #return accuracy_svrbf_pca, accuracy_svrbf_bf, accuracy_svm_pca, accuracy_svm_bf, accuracy_lda_pca, accuracy_lda_bf, accuracy_ff_pca, accuracy_ff_bf, accuracy_rf_pca, accuracy_rf_bf, accuracy_cnn# accuracy_dlda_pca, accuracy_dlda_bf

def evaluate_whole_sessions_in_multiple_runs(n, feature):       
    acc = []
    for i in range(n):
        print(i)
        keras.backend.clear_session()
        run = evaluate_whole_sessions(feature)
        acc.append(run)
    return acc


# =============================================================================
# Evaluations for Segment Insertion:
#    30 random movement segments of a certain window size (here: a grid of several window sizes) are drawn per participant. 
#    All six algorithms (LDA, FF, RF, SVM(LIN), SVM(RBF) and CNN) are trained on the resulting 60 x 30 training segments. 
#    The optimal window size is selected via cross-validation. Predictions for the 12 x 30 testing segments are evaluated, and a final prediction assessed by majority
#    voting is established for each of the 12 test patterns. 
# =============================================================================

def evaluate_segments(feature):
    """
    feature: Either "Right_Hand", "Left_Hand" or "Both_Hands" 

    Returns
    -------
    accuracy rates for LDA, FF, RF, SVM(LIN), SVM(RBF) and CNN. The final prediction per participant in the test set (12 total) is drawn via majority vote.

    """
    start = time.time()    
    train_indices, test_indices = train_test_indices()
    
    try:
        pickle_in = open(train_path + feature + ".pickle", "rb")
        df = pickle.load(pickle_in)
        #df = df[:,[0,1,3,6]]
    except: 
        print("Invalid Body Part entered. Must be 'Head', 'Left', 'Right' or 'All'")
    
    df = standardize(df, train_indices)
    df = standardize(df, test_indices)
    

    window_grid =  [50,200]#, 350, 500, 650]
    
    n=30
    #n_train = 30
    #n_test = 30
    
    best_acc_clf = 0
    best_size_clf = 0
    accuracy_lda_maj = 0
    
    best_acc_ff = 0
    best_size_ff = 0
    accuracy_ff_maj = 0
    
    best_acc_rf = 0
    best_size_rf = 0
    accuracy_rf_maj = 0
    
    best_acc_svc = 0
    best_size_svc = 0
    accuracy_svc_maj = 0
    
    best_acc_svrbf = 0
    best_size_svrbf = 0
    accuracy_svrbf_maj = 0
    
    best_acc_cnn = 0
    best_size_cnn = 0
    accuracy_cnn_maj = 0
    
    
    
    for i in window_grid:
        df_seg, targets = random_signal(n, df, window_size = i)
        
        means = np.mean(df_seg, axis = 1)
        stds = np.std(df_seg, axis = 1)
        mins = np.min(df_seg, axis = 1)
        maxs = np.max(df_seg, axis = 1)
        df_bf = np.concatenate((means, stds, mins, maxs), axis = 1)
        extended_train_indices = [[i for i in range(n*j, n*(j+1))] for j in train_indices]
        extended_test_indices = [[i for i in range(n*j, n*(j+1))] for j in test_indices]
        df_train = df_bf[[val for sublist in extended_train_indices for val in sublist]]
        df_test = df_bf[[val for sublist in extended_test_indices for val in sublist]]
        
        cnn_train = df_seg[[val for sublist in extended_train_indices for val in sublist],:,:]
        cnn_test = df_seg[[val for sublist in extended_test_indices for val in sublist],:,:]
        
        targets_train = targets[[val for sublist in extended_train_indices for val in sublist]]
        targets_test = targets[[val for sublist in extended_test_indices for val in sublist]]
        
        np.save(train_path + "xtrain" + str(i) + ".npy", df_train)
        np.save(train_path + "ytrain" + str(i) + ".npy", targets_train)
        np.save(train_path + "xtest" + str(i) + ".npy", df_test)
        np.save(train_path + "ytest" + str(i) + ".npy", targets_test)
        
        np.save(train_path + "cnntrain" + str(i) + ".npy", cnn_train)
        np.save(train_path + "cnntest" + str(i) + ".npy", cnn_test)
        
        
        def cross_validation_accuracy(alg, data_train):
            cv_pred = cross_val_predict(alg, data_train, targets_train, cv = 5)
            if cv_pred.ndim > 1:
                cv_pred = [np.argmax(cv_pred[i]) for i in range(len(cv_pred))]
            cv_acc = 100/len(targets_train)*np.sum(np.transpose(cv_pred) == np.transpose(targets_train))
            
            return cv_acc
        
        
        lda = LinearDiscriminantAnalysis()
        accuracy_clf_seg = cross_validation_accuracy(lda, df_train)
        
        ff = keras.wrappers.scikit_learn.KerasRegressor(build_fn = build_ff, input_dim = df_train.shape[1], batch_size=32, epochs =15)#, verbose = 0)
        accuracy_ff_seg = cross_validation_accuracy(ff, df_train)
        
        rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
        accuracy_rf_seg = cross_validation_accuracy(rf, df_train)
        
        svc = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr')
        accuracy_svc_seg = cross_validation_accuracy(svc, df_train)
        
        svrbf = svm.SVC(kernel = 'rbf', decision_function_shape = 'ovr')
        accuracy_svrbf_seg = cross_validation_accuracy(svrbf, df_train)
        
        cnn = keras.wrappers.scikit_learn.KerasClassifier(build_fn = build_cnn, num_channels = cnn_train.shape[2], epochs = 10, window_size = i)
        accuracy_cnn_seg = cross_validation_accuracy(cnn, cnn_train)
        
        

        if(accuracy_clf_seg > best_acc_clf):
            best_acc_clf = accuracy_clf_seg
            best_size_clf = i
                
        if(accuracy_ff_seg > best_acc_ff):
            best_acc_ff = accuracy_ff_seg
            best_size_ff = i
                                          
        if(accuracy_rf_seg > best_acc_rf):
            best_acc_rf = accuracy_rf_seg
            best_size_rf = i
       
        if(accuracy_svc_seg > best_acc_svc):
            best_acc_svc = accuracy_svc_seg
            best_size_svc = i
            
        if(accuracy_svrbf_seg > best_acc_svrbf):
            best_acc_svrbf = accuracy_svrbf_seg
            best_size_svrbf = i
            
        if(accuracy_cnn_seg > best_acc_cnn):
            best_acc_cnn = accuracy_cnn_seg
            best_size_cnn = i

        #print(accuracy_clf_seg)
    
    best_sizes = {"lda": best_size_clf, "ff": best_size_ff, "rf": best_size_rf, "svm_lin": best_size_svc, "svm_rbf": best_size_svrbf, "cnn": best_size_cnn}
    classifiers = {"lda": lda, "ff": ff, "rf": rf, "svm_lin": svc, "svm_rbf": svrbf, "cnn": keras.wrappers.scikit_learn.KerasClassifier(build_fn = build_cnn, num_channels = cnn_train.shape[2], epochs = 10, window_size = best_sizes["cnn"])}    

    
    def fit_and_predict_with_best_size(alg):
        best_size_for_alg = best_sizes[alg]
        if alg != "cnn":
            df_train = np.load(train_path + "xtrain" + str(best_size_for_alg) + ".npy")
            df_test = np.load(train_path + "xtest" + str(best_size_for_alg) + ".npy")
        else:
            df_train = np.load(train_path + "cnntrain" + str(best_size_for_alg) + ".npy")
            df_test = np.load(train_path + "cnntest" + str(best_size_for_alg) + ".npy")
        targets_train = np.load(train_path + "ytrain" + str(best_size_for_alg) + ".npy")
        targets_test = np.load(train_path + "ytest" + str(best_size_for_alg) + ".npy")
        
        clf = classifiers[alg]
        _ = clf.fit(df_train, targets_train)
        pred = clf.predict(df_test)
        if alg == "ff":
            pred = [np.argmax(pred[i]) for i in range(len(pred))]
        #accuracy_lda_seg = 100/len(y_bf_seg_test) *np.sum(np.transpose(pred_lda_bf_seg) == np.transpose(y_bf_seg_test))
        majority_pred = [stats.mode(pred[n*i:n*(i+1)])[0][0] for i in range(len(test_indices))]
        accuracy_maj = np.sum(np.transpose(majority_pred) == np.transpose(targets_test[::n]))
        #accuracy_maj = 100/len(targets_test[::n])*np.sum(np.transpose(majority_pred) == np.transpose(targets_test[::n]))
        
        return accuracy_maj
    
    accuracy_lda_maj = fit_and_predict_with_best_size("lda")
    accuracy_ff_maj = fit_and_predict_with_best_size("ff")
    accuracy_rf_maj = fit_and_predict_with_best_size("rf")
    accuracy_svc_maj = fit_and_predict_with_best_size("svm_lin")
    accuracy_svrbf_maj = fit_and_predict_with_best_size("svm_rbf")
    
    accuracy_cnn_maj = fit_and_predict_with_best_size("cnn")
        
    

    end = time.time()
    print(end-start)
    

    return [accuracy_lda_maj, best_size_clf, accuracy_ff_maj, best_size_ff, accuracy_rf_maj, best_size_rf, accuracy_svc_maj, best_size_svc, accuracy_svrbf_maj, best_size_svrbf, accuracy_cnn_maj, best_size_cnn]


def evaluate_segments_in_multiple_runs(n, feature):
    #best_accuracy_lda = []
    acc = []
    for i in range(n):
        print(i)
        keras.backend.clear_session()
        run = evaluate_segments(feature)
        acc.append(run)
    return acc

# =============================================================================
# Prepare Results for Plotting
# Note: Since selection of training and testing sets as well as choice of segments is random, 
# your results will most likely not match the ones listed in the paper exactly. 
# =============================================================================


def prepare_results_for_plot_whole_sessions(n):
    """
    n: number of runs. 
    
    Returns: dictionaries, where keys represent the number of possible correct predictions per run (i.e., 0 - 12) 
             and corresponding values the actual number that a certain algorithm achieved that number of correct predictions in a total of n runs.
             Values are saved as three-entry-list, where the first entry corresponds to left hand patterns, the second entry to right hand patterns and
             the last entry to both hand patterns. 
             example:
             The lda_bf_dict has an entry of {"3": [5,2,3]}, where the key "3" indicates, that only 3 out of 12 test patterns were predicted correctly, and this 
             occurred in 5 cases for left hand pattern insertion, in 2 cases for right hand pattern insertion, and in 3 cases for both hand pattern insertion. 
    -------
    lda_bf_dict : results for LDA with basic feature input.
    ff_bf_dict : results for FF with basic feature input.
    rf_bf_dict : results for RF with basic feature input.
    svmlin_bf_dict : results for SVM(LIN) with basic feature input. 
    svrbf_bf_dict : results for SVM(RBF) with basic feature input. 
    lda_pca_dict : results for LDA with pc input.
    ff_pca_dict : results for FF with pc input.
    rf_pca_dict : results for RF with pc input.
    svmlin_pca_dict : rsults for SVM(LIN) with pc input.
    svrbf_pca_dict : results for SVM(RBF) with pc input.
    cnn_dict : results for CNN.
    """
    right_hand_results = evaluate_whole_sessions_in_multiple_runs(n, "Right_Hand")
    left_hand_results = evaluate_whole_sessions_in_multiple_runs(n, "Left_Hand")
    both_hand_results = evaluate_whole_sessions_in_multiple_runs(n, "Both_Hands")
    
    
    lda_bf = [[left_hand_results[i][1] for i in range(len(left_hand_results))],
              [right_hand_results[i][1] for i in range(len(right_hand_results))],
              [both_hand_results[i][1] for i in range(len(both_hand_results))]]
        
    ff_bf = [[left_hand_results[i][3] for i in range(len(left_hand_results))],
              [right_hand_results[i][3] for i in range(len(right_hand_results))],
              [both_hand_results[i][3] for i in range(len(both_hand_results))]]
    
    rf_bf = [[left_hand_results[i][5] for i in range(len(left_hand_results))],
              [right_hand_results[i][5] for i in range(len(right_hand_results))],
              [both_hand_results[i][5] for i in range(len(both_hand_results))]]
    
    svmlin_bf = [[left_hand_results[i][7] for i in range(len(left_hand_results))],
              [right_hand_results[i][7] for i in range(len(right_hand_results))],
              [both_hand_results[i][7] for i in range(len(both_hand_results))]]

    svrbf_bf = [[left_hand_results[i][9] for i in range(len(left_hand_results))],
              [right_hand_results[i][9] for i in range(len(right_hand_results))],
              [both_hand_results[i][9] for i in range(len(both_hand_results))]]
        
    lda_pca = [[left_hand_results[i][0] for i in range(len(left_hand_results))],
              [right_hand_results[i][0] for i in range(len(right_hand_results))],
              [both_hand_results[i][0] for i in range(len(both_hand_results))]]
       
    ff_pca = [[left_hand_results[i][2] for i in range(len(left_hand_results))],
              [right_hand_results[i][2] for i in range(len(right_hand_results))],
              [both_hand_results[i][2] for i in range(len(both_hand_results))]]

    rf_pca = [[left_hand_results[i][4] for i in range(len(left_hand_results))],
              [right_hand_results[i][4] for i in range(len(right_hand_results))],
              [both_hand_results[i][4] for i in range(len(both_hand_results))]]

    svmlin_pca = [[left_hand_results[i][6] for i in range(len(left_hand_results))],
              [right_hand_results[i][6] for i in range(len(right_hand_results))],
              [both_hand_results[i][6] for i in range(len(both_hand_results))]]
    
    svrbf_pca = [[left_hand_results[i][8] for i in range(len(left_hand_results))],
              [right_hand_results[i][8] for i in range(len(right_hand_results))],
              [both_hand_results[i][8] for i in range(len(both_hand_results))]]
    
    cnn = [[left_hand_results[i][10] for i in range(len(left_hand_results))],
              [right_hand_results[i][10] for i in range(len(right_hand_results))],
              [both_hand_results[i][10] for i in range(len(both_hand_results))]]

    all_results = [lda_bf, ff_bf, rf_bf, svmlin_bf, svrbf_bf, lda_pca, ff_pca, rf_pca, svmlin_pca, svrbf_pca, cnn]
    
    #base_dict = {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]}
    
    #unnessary replica due to base_dict.copy() not working...
    #lda_bf_dict = base_dict.copy()
    result_dicts = [{'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]}]
    
    dict_counter = 0
    for result in all_results:
        #print(dict_counter)
        result_dict = result_dicts[dict_counter]
        pred_list = [np.unique(result[0]), np.unique(result[1]), np.unique(result[2])]
        for key in result_dict.keys():
            #print(int(key))           
            for i in range(len(pred_list)):
                #print(i)
                if int(key) in pred_list[i]:
                    #print(key)
                    result_dict[key][i] = np.sum(np.array(result[i]) == int(key)) 
        dict_counter = dict_counter + 1
                    
        
    lda_bf_dict = result_dicts[0]
    ff_bf_dict = result_dicts[1]
    rf_bf_dict = result_dicts[2]
    svmlin_bf_dict = result_dicts[3]
    svrbf_bf_dict = result_dicts[4]
    lda_pca_dict = result_dicts[5]
    ff_pca_dict = result_dicts[6]
    rf_pca_dict = result_dicts[7]
    svmlin_pca_dict = result_dicts[8]
    svrbf_pca_dict = result_dicts[9]
    cnn_dict = result_dicts[10]
    
    return lda_bf_dict, ff_bf_dict, rf_bf_dict, svmlin_bf_dict, svrbf_bf_dict, lda_pca_dict, ff_pca_dict, rf_pca_dict, svmlin_pca_dict, svrbf_pca_dict, cnn_dict
   
    
def prepare_results_for_plot_segments(n):
    """
    n: number of runs. 
    
    Returns: dictionaries, where keys represent either the chosen window sizes or the number of possible correct predictions per run (i.e., 0 - 12)
             and corresponding values the actual number that a certain algorithm chose that window size/achieved that number of correct predictions in a total of n runs.
             Values are saved as three-entry-list, where the first entry corresponds to left hand patterns, the second entry to right hand patterns and
             the last entry to both hand patterns. 
    -------
    lda_win_dict : window sizes for CNN.
    ff_win_dict : window sizes for CNN.
    rf_win_dict : window sizes for CNN.
    svmlin_win_dict : window sizes for CNN.
    svrbf_win_dict : window sizes for CNN.
    cnn_win : window sizes for CNN.
    lda_dict : results for LDA.
    ff_dict : results for FF.
    rf_dict : results for RF.
    svmlin_dict : rsults for SVM(LIN).
    svrbf_dict : results for SVM(RBF).
    cnn_dict : results for CNN.
    """
     
    right_hand_results = evaluate_segments_in_multiple_runs(n, "Right_Hand") 
    left_hand_results = evaluate_segments_in_multiple_runs(n, "Left_Hand")
    both_hand_results = evaluate_segments_in_multiple_runs(n, "Both_Hands")
    
    
    lda_win = [[left_hand_results[i][1] for i in range(len(left_hand_results))],
              [right_hand_results[i][1] for i in range(len(right_hand_results))],
              [both_hand_results[i][1] for i in range(len(both_hand_results))]]
        
    ff_win = [[left_hand_results[i][3] for i in range(len(left_hand_results))],
              [right_hand_results[i][3] for i in range(len(right_hand_results))],
              [both_hand_results[i][3] for i in range(len(both_hand_results))]]
    
    rf_win = [[left_hand_results[i][5] for i in range(len(left_hand_results))],
              [right_hand_results[i][5] for i in range(len(right_hand_results))],
              [both_hand_results[i][5] for i in range(len(both_hand_results))]]
    
    svmlin_win = [[left_hand_results[i][7] for i in range(len(left_hand_results))],
              [right_hand_results[i][7] for i in range(len(right_hand_results))],
              [both_hand_results[i][7] for i in range(len(both_hand_results))]]

    svrbf_win = [[left_hand_results[i][9] for i in range(len(left_hand_results))],
              [right_hand_results[i][9] for i in range(len(right_hand_results))],
              [both_hand_results[i][9] for i in range(len(both_hand_results))]]
    
    cnn_win = [[left_hand_results[i][11] for i in range(len(left_hand_results))],
              [right_hand_results[i][11] for i in range(len(right_hand_results))],
              [both_hand_results[i][11] for i in range(len(both_hand_results))]]
        
    lda = [[left_hand_results[i][0] for i in range(len(left_hand_results))],
              [right_hand_results[i][0] for i in range(len(right_hand_results))],
              [both_hand_results[i][0] for i in range(len(both_hand_results))]]
       
    ff = [[left_hand_results[i][2] for i in range(len(left_hand_results))],
              [right_hand_results[i][2] for i in range(len(right_hand_results))],
              [both_hand_results[i][2] for i in range(len(both_hand_results))]]

    rf = [[left_hand_results[i][4] for i in range(len(left_hand_results))],
              [right_hand_results[i][4] for i in range(len(right_hand_results))],
              [both_hand_results[i][4] for i in range(len(both_hand_results))]]

    svmlin = [[left_hand_results[i][6] for i in range(len(left_hand_results))],
              [right_hand_results[i][6] for i in range(len(right_hand_results))],
              [both_hand_results[i][6] for i in range(len(both_hand_results))]]
    
    svrbf = [[left_hand_results[i][8] for i in range(len(left_hand_results))],
              [right_hand_results[i][8] for i in range(len(right_hand_results))],
              [both_hand_results[i][8] for i in range(len(both_hand_results))]]
    
    cnn = [[left_hand_results[i][10] for i in range(len(left_hand_results))],
              [right_hand_results[i][10] for i in range(len(right_hand_results))],
              [both_hand_results[i][10] for i in range(len(both_hand_results))]]

    all_results = [lda_win, ff_win, rf_win, svmlin_win, svrbf_win, cnn_win, lda, ff, rf, svmlin, svrbf, cnn]
    
    #base_dict = {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]}
    
    #unnessary replica due to base_dict.copy() not working...
    result_dicts = [{'50': [0,0,0], '200': [0,0,0], '350': [0,0,0], '500':[0,0,0], '650': [0,0,0]},
                  {'50': [0,0,0], '200': [0,0,0], '350': [0,0,0], '500':[0,0,0], '650': [0,0,0]},
                  {'50': [0,0,0], '200': [0,0,0], '350': [0,0,0], '500':[0,0,0], '650': [0,0,0]},
                  {'50': [0,0,0], '200': [0,0,0], '350': [0,0,0], '500':[0,0,0], '650': [0,0,0]},
                  {'50': [0,0,0], '200': [0,0,0], '350': [0,0,0], '500':[0,0,0], '650': [0,0,0]},
                  {'50': [0,0,0], '200': [0,0,0], '350': [0,0,0], '500':[0,0,0], '650': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]},
                  {'0': [0,0,0], '1': [0,0,0], '2': [0,0,0],'3':[0,0,0], '4':[0,0,0], '5': [0,0,0], '6': [0,0,0], '7': [0,0,0], '8': [0,0,0], '9': [0,0,0], '10': [0,0,0], '11': [0,0,0], '12': [0,0,0]}]
               
    dict_counter = 0
    for result in all_results:
        #print(dict_counter)
        result_dict = result_dicts[dict_counter]
        pred_list = [np.unique(result[0]), np.unique(result[1]), np.unique(result[2])]
        for key in result_dict.keys():
            #print(int(key))           
            for i in range(len(pred_list)):
                #print(i)
                if int(key) in pred_list[i]:
                    #print(key)
                    result_dict[key][i] = np.sum(np.array(result[i]) == int(key)) 
        dict_counter = dict_counter + 1
                    
        
    lda_win_dict = result_dicts[0]
    ff_win_dict = result_dicts[1]
    rf_win_dict = result_dicts[2]
    svmlin_win_dict = result_dicts[3]
    svrbf_win_dict = result_dicts[4]
    cnn_win_dict = result_dicts[5]
    lda_dict = result_dicts[6]
    ff_dict = result_dicts[7]
    rf_dict = result_dicts[8]
    svmlin_dict = result_dicts[9]
    svrbf_dict = result_dicts[10]
    cnn_dict = result_dicts[11]
    
    return lda_win_dict, ff_win_dict, rf_win_dict, svmlin_win_dict, svrbf_win_dict, cnn_win_dict, lda_dict, ff_dict, rf_dict, svmlin_dict, svrbf_dict, cnn_dict


if __name__ == '__main__':
    #n = int(sys.argv[1])
    whole_session_results_after_n_runs = prepare_results_for_plot_whole_sessions(100)
    segment_results_after_n_runs = prepare_results_for_plot_segments(100)
    
    np.save(train_path + "whole_session_results.npy", whole_session_results_after_n_runs)
    np.save(train_path + "segment_results.npy", segment_results_after_n_runs)




