# -*- coding: utf-8 -*-

#import time
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from tensorflow import keras
from eval_base_study import  standardize, random_signal, build_ff, build_cnn
import os

dirname = os.path.dirname(__file__)
train_path = os.path.join(dirname, 'traindata/').replace("\\","/")

# =============================================================================
# Evaluations for Reproducibility Study:
#    30 random movement segments of a fixed window size (here: 200) are drawn per participant who was part of our base study (72 total) and the 
#    reproducibility study (43 total).
#    All six algorithms (LDA, FF, RF, SVM(LIN), SVM(RBF) and CNN) are trained on the resulting 72 x 30 training segments. 
#    Predictions for the 43 x 30 testing segments are evaluated, and a final prediction assessed by majority
#    voting is established for each of the 43 reproducibility patterns. 
# =============================================================================

def evaluate_rep():
    #start = time.time()
    pickle_in = open(train_path + "Eval_Data" + ".pickle", "rb")
    df_rep = pickle.load(pickle_in)
    
    pickle_in = open(train_path + "Right_Hand" + ".pickle", "rb")
    df_right = pickle.load(pickle_in)
    
    df_rep = standardize(df_rep, [i for i in range(43)])
    df_right = standardize(df_right, [i for i in range(72)])
    
    
    window_grid =  [200]
    n = 30
    for i in window_grid:
        df_hof, targets_hof = random_signal(n, df_right, window_size = i)
        means_hof = np.mean(df_hof, axis = 1)
        stds_hof = np.std(df_hof, axis = 1)
        mins_hof = np.min(df_hof, axis = 1)
        maxs_hof = np.max(df_hof, axis = 1)
        
        df_hof_bf = np.concatenate((means_hof, stds_hof, mins_hof, maxs_hof), axis = 1)
   
        x_rep, y_rep = random_signal(n, df_rep, window_size = i)
        means_rep = np.mean(x_rep, axis = 1)
        stds_rep = np.std(x_rep, axis = 1)
        mins_rep = np.min(x_rep, axis = 1)
        maxs_rep = np.max(x_rep, axis = 1)
        
        df_rep = np.concatenate((means_rep, stds_rep, mins_rep, maxs_rep), axis = 1)
        
                
        lda = LinearDiscriminantAnalysis()
        _ = lda.fit(df_hof_bf, targets_hof)
        
        ff = keras.wrappers.scikit_learn.KerasRegressor(build_fn = build_ff, input_dim = df_hof_bf.shape[1], batch_size=32, epochs =15)
        _ = ff.fit(df_hof_bf, targets_hof)
    
        rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
        _ = rf.fit(df_hof_bf, np.reshape(targets_hof, (len(targets_hof),)))
    
        svm_lin = svm.SVC(kernel = 'linear')
        _ = svm_lin.fit(df_hof_bf, targets_hof)
        
        svm_rbf = svm.SVC()
        _ = svm_rbf.fit(df_hof_bf, targets_hof)
        
        cnn = keras.wrappers.scikit_learn.KerasRegressor(build_fn = build_cnn, num_channels = df_hof.shape[2], window_size = i, epochs = 10)
        _ = cnn.fit(df_hof, targets_hof)
        

        
        pred = lda.predict(df_rep)
        maj_lda = [stats.mode(pred[i*n:(i+1)*n])[0][0] for i in range(43)]
        acc_lda = np.sum(np.transpose(maj_lda) == np.transpose(y_rep[::n]))
        #acc_lda = 100/len(y_rep[::n]) * np.sum(np.transpose(maj_lda) == np.transpose(y_rep[::n]))
        
        pred_ff = ff.predict(df_rep)
        #acc = 100/len(y_rep) * np.sum(np.transpose(y_rep) == np.transpose(pred))
        pred_ff = [np.argmax(i) for i in pred_ff]
        maj_ff = [stats.mode(pred_ff[i*n:(i+1)*n])[0][0] for i in range(43)]
        acc_ff = np.sum(np.transpose(maj_ff) == np.transpose(y_rep[::n]))
        #acc_ff = 100/len(y_rep[::n]) * np.sum(np.transpose(maj_ff) == np.transpose(y_rep[::n]))
        
        pred_rf = rf.predict(df_rep)
        #acc = 100/len(y_rep) * np.sum(np.transpose(y_rep) == np.transpose(pred))
        maj_rf = [stats.mode(pred_rf[i*n:(i+1)*n])[0][0] for i in range(43)]
        acc_rf = np.sum(np.transpose(maj_rf) == np.transpose(y_rep[::n]))
        #acc_rf = 100/len(y_rep[::n]) * np.sum(np.transpose(maj_rf) == np.transpose(y_rep[::n]))
        
        pred_svm = svm_lin.predict(df_rep)
        #acc = 100/len(y_rep) * np.sum(np.transpose(y_rep) == np.transpose(pred))
        maj_svm = [stats.mode(pred_svm[i*n:(i+1)*n])[0][0] for i in range(43)]
        acc_svm = np.sum(np.transpose(maj_svm) == np.transpose(y_rep[::n]))
        #acc_svm = 100/len(y_rep[::n]) * np.sum(np.transpose(maj_svm) == np.transpose(y_rep[::n]))
        
        pred_svmrbf = svm_rbf.predict(df_rep)
        #acc = 100/len(y_rep) * np.sum(np.transpose(y_rep) == np.transpose(pred))
        maj_svmrbf = [stats.mode(pred_svmrbf[i*n:(i+1)*n])[0][0] for i in range(43)]
        acc_svmrbf = np.sum(np.transpose(maj_svmrbf) == np.transpose(y_rep[::n]))
        #acc_svmrbf = 100/len(y_rep[::n]) * np.sum(np.transpose(maj_svmrbf) == np.transpose(y_rep[::n]))
        
        pred_cnn = cnn.predict(x_rep)
        pred_cnn = [np.argmax(i) for i in pred_cnn]
        #acc = 100/len(y_rep) * np.sum(np.transpose(y_rep) == np.transpose(pred))
        maj_cnn = [stats.mode(pred_cnn[i*n:(i+1)*n])[0][0] for i in range(43)]
        acc_cnn = np.sum(np.transpose(maj_cnn) == np.transpose(y_rep[::n]))
        
        
    #end = time.time()
    #print(end - start)
        
    return [acc_lda, acc_ff, acc_rf, acc_svm, acc_svmrbf, acc_cnn]

def eval_acc_rep(n):
    accf = []
    for i in range(n):
        print(i)
        keras.backend.clear_session()
        acc = evaluate_rep()
        accf.append(acc)
    return accf

def prepare_validation_results_for_plot(n):
    """
    

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.

    Returns : dictionaries, where keys represent the number of possible correct predictions per run (i.e., 0 - 43) 
             and corresponding values the actual number that a certain algorithm achieved that number of correct predictions in a total of n runs. 
             example:
             The lda_dict has an entry of {"36": 5}, where the key "36" indicates, that 36out of 43 validation patterns were predicted correctly, and this 
             occurred in 5 cases out of n total.
    -------
    lda_dict : TYPE
        DESCRIPTION.
    ff_dict : TYPE
        DESCRIPTION.
    rf_dict : TYPE
        DESCRIPTION.
    svmlin_dict : TYPE
        DESCRIPTION.
    svrbf_dict : TYPE
        DESCRIPTION.
    lda_dict : TYPE
        DESCRIPTION.

    """
    results = eval_acc_rep(n)
    
    
    lda = [results[i][0] for i in range(len(results))]
    
    ff = [results[i][1] for i in range(len(results))]
    
    rf = [results[i][2] for i in range(len(results))]
    
    svmlin = [results[i][3] for i in range(len(results))]
    
    svrbf = [results[i][4] for i in range(len(results))]
    
    cnn = [results[i][5] for i in range(len(results))]
        
    all_results = [lda, ff, rf, svmlin, svrbf, cnn]
    
    #unnessary replica due to base_dict.copy() not working...
    #base_dict = {'0': 0, '1': 0, '2': 0,'3':0, '4':0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0,
    #             '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 0, '24': 0,
    #             '25': 0, '26': 0, '27': 0, '28': 0, '29': 0, '30': 0, '31': 0, '32': 0, '33': 0, '34': 0, '35': 0, '36': 0,
    #             '37': 0, '38': 0, '39': 0, '40': 0, '41': 0, '42': 0, '43': 0}
    
    #lda_bf_dict = base_dict.copy()
    result_dicts = [{'0': 0, '1': 0, '2': 0,'3':0, '4':0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0,
                 '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 0, '24': 0,
                 '25': 0, '26': 0, '27': 0, '28': 0, '29': 0, '30': 0, '31': 0, '32': 0, '33': 0, '34': 0, '35': 0, '36': 0,
                 '37': 0, '38': 0, '39': 0, '40': 0, '41': 0, '42': 0, '43': 0},
                    {'0': 0, '1': 0, '2': 0,'3':0, '4':0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0,
                 '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 0, '24': 0,
                 '25': 0, '26': 0, '27': 0, '28': 0, '29': 0, '30': 0, '31': 0, '32': 0, '33': 0, '34': 0, '35': 0, '36': 0,
                 '37': 0, '38': 0, '39': 0, '40': 0, '41': 0, '42': 0, '43': 0},
                    {'0': 0, '1': 0, '2': 0,'3':0, '4':0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0,
                 '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 0, '24': 0,
                 '25': 0, '26': 0, '27': 0, '28': 0, '29': 0, '30': 0, '31': 0, '32': 0, '33': 0, '34': 0, '35': 0, '36': 0,
                 '37': 0, '38': 0, '39': 0, '40': 0, '41': 0, '42': 0, '43': 0},
                    {'0': 0, '1': 0, '2': 0,'3':0, '4':0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0,
                 '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 0, '24': 0,
                 '25': 0, '26': 0, '27': 0, '28': 0, '29': 0, '30': 0, '31': 0, '32': 0, '33': 0, '34': 0, '35': 0, '36': 0,
                 '37': 0, '38': 0, '39': 0, '40': 0, '41': 0, '42': 0, '43': 0},
                    {'0': 0, '1': 0, '2': 0,'3':0, '4':0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0,
                 '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 0, '24': 0,
                 '25': 0, '26': 0, '27': 0, '28': 0, '29': 0, '30': 0, '31': 0, '32': 0, '33': 0, '34': 0, '35': 0, '36': 0,
                 '37': 0, '38': 0, '39': 0, '40': 0, '41': 0, '42': 0, '43': 0},
                    {'0': 0, '1': 0, '2': 0,'3':0, '4':0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0,
                 '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 0, '24': 0,
                 '25': 0, '26': 0, '27': 0, '28': 0, '29': 0, '30': 0, '31': 0, '32': 0, '33': 0, '34': 0, '35': 0, '36': 0,
                 '37': 0, '38': 0, '39': 0, '40': 0, '41': 0, '42': 0, '43': 0}]
    
    dict_counter = 0
    for result in all_results:
        #print(dict_counter)
        result_dict = result_dicts[dict_counter]
        pred_list = np.unique(result)
        for key in result_dict.keys():
            #print(int(key))           
            if int(key) in pred_list:
                    #print(key)
                result_dict[key] = np.sum(np.array(result) == int(key)) 
        dict_counter = dict_counter + 1
                    
        
    lda_dict = result_dicts[0]
    ff_dict = result_dicts[1]
    rf_dict = result_dicts[2]
    svmlin_dict = result_dicts[3]
    svrbf_dict = result_dicts[4]
    lda_dict = result_dicts[5]

    
    return lda_dict, ff_dict, rf_dict, svmlin_dict, svrbf_dict, lda_dict
   