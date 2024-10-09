#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code from paper Chiossi et al, 2024
Source: https://github.com/hchiossi/hpc-hierarchy

"""

import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import pandas as pd


#%%

def SVM_global(data, labels_vec, decoded_var, reps=100, sample_fraction=1, shuffle_labels=False):
    """
    
    Parameters
    ----------
    data : 2D array
        Array containing the neural population vectors. It should have shape nsamples x nfeatures, where nfeatures 
        is the number of neurons recorded.
    labels_vec : DataFrame
        Pandas DataFrame containing all the labels associated with each population vector in data. Should contain at least two columns
        one of which is the decoded variable. The remaining variables will just be monitored but not used by the decoder.
    decoded_var : str
        Variable to be decoded. It should match the name of one of the columns in labels_vec
    reps : int, optional
        Number of bootstrap cycles. The default is 100.
    sample_fraction : float or int, optional
        The size of each bootstrap sample in relation to the full data, so needs to be between 0 and 1. The default is 1.
    shuffle_labels : bool, optional
        If True, the labels for the decoded variable will be shuffled between samples at every iteration. The default is False.

    Returns
    -------
    svm_acc : 1D array
        Classification accuracy in each bootstrap sample. Its length equal reps.
    pred_vs_monitoredvar : DataFrame
        Each row contains one sample in the test set of each bootstrap sample, the real and predicted value of the decoded_var,
        the bootstrap sample it originates from and the value of any other variable that was monitored.
    weights : 2D array
        Weights given to each feature at each bootstrap sample. It has shape reps x nfeatures.

    """


    labels =  labels_vec[decoded_var].astype(str).to_numpy()
    nclasses = len(np.unique(labels))
    monitored_vec = labels_vec.drop(decoded_var, axis=1)    
    
    #to make sure all positions are sampled equally always, this is redundant if the decoded variable is position
    pos_labels = labels_vec['Region'].astype(str).to_numpy() #coarse-grained position
    npos_labels = len(np.unique(pos_labels))   
    
    combined_labels = labels+pos_labels #just for the straified bootstrap to make sure to sample from all position/cat combinations
  
    if nclasses == 2:
        weights = np.zeros((reps,np.shape(data)[1]))
        
    else:
        dims = int((nclasses*(nclasses-1)) /2) #from the sklearn svm documentation
        weights = np.zeros((reps,dims,np.shape(data)[1]))   

    df = pd.DataFrame(data)
    svm_acc = np.zeros(reps)
    for rep in range(reps):
        if rep%10==0: #to keep track of progress, optional
            print(f'{rep}/{reps}')
        #do train/test split
        correct_nclasses = False
        while correct_nclasses == False: #in case the bootstrapping generate samples missing a class
            X_train, X_test = train_test_split(df, test_size=0.3, shuffle=True, stratify=combined_labels)
            
            #the bootstrap sampling has to come second since there is replacement, to make sure values are not both in train and test
            data_train = resample(X_train, replace=True, n_samples=int(sample_fraction*X_train.shape[0]), stratify=combined_labels[X_train.index]) #resample with replacement, keeping label proportions
            data_test = resample(X_test, replace=True, n_samples=int(sample_fraction*X_test.shape[0]), stratify=combined_labels[X_test.index])
            label_train, label_test = labels[data_train.index], labels[data_test.index]
            pos_label_train = pos_labels[data_train.index]
            correct_nclasses = True
            
            if len(np.unique(label_train+pos_label_train)) != nclasses*npos_labels: #make sure that it trained in all classes
                correct_nclasses = False
                #attention: intentionally, the test set might not have all classes!!!
        
        if shuffle_labels:
            label_train = np.random.permutation(label_train)
        
        svm_model = svm.SVC(kernel='linear')
        svm_model.fit(data_train, label_train)
        pred_test = svm_model.predict(data_test)
        if rep==0: #only c==0 for other version
            predictions = pred_test
            real_labels = label_test
            mon_labels = monitored_vec.iloc[data_test.index] #monitored labels, not used for decoding
            rep_num = [rep]*len(label_test)
        else:
            predictions = np.concatenate((predictions,pred_test))
            real_labels = np.concatenate((real_labels, label_test)) #just to be in the same order as the predictions
            mon_labels = pd.concat([mon_labels, monitored_vec.iloc[data_test.index]], ignore_index=True) #just to be in the same order as the predictions
            rep_num.extend([rep]*len(label_test))
        svm_acc[rep] = accuracy_score(label_test,pred_test)
        weights[rep,:] = svm_model.coef_   

    pred_df = pd.DataFrame({'Real_label':real_labels,'Pred_label':predictions,'Boot_rep':rep_num})
    pred_vs_monitoredvar = pd.concat([pred_df,mon_labels],axis=1)

    return svm_acc, pred_vs_monitoredvar, weights

#%%

def SVM_conditional(data, labels_vec, decoded_var, conditional_var, reps=100, sample_fraction=1): 
    """
    

    Parameters
    ----------
    data : 2D array
        Array containing the neural population vectors. It should have shape nsamples x nfeatures, where nfeatures 
        is the number of neurons recorded.
    labels_vec : DataFrame
        Pandas DataFrame containing all the labels associated with each population vector in data. Should contain at least two columns
        one of which is the decoded variable. The remaining variables will just be monitored but not used by the decoder.
    decoded_var : str
        Variable to be decoded. It should match the name of one of the columns in labels_vec
    conditional_var: str
        Variable to condition the decoding on. The data will be split according to the value of this variable and the decoding of
        decoded_var will happen separately in each subset. It should match the name of one of the columns in labels_vec.
    reps : int, optional
        Number of bootstrap cycles. The default is 100.
    sample_fraction : float or int, optional
        The size of each bootstrap sample in relation to the full data, so needs to be between 0 and 1. The default is 1.
    shuffle_labels : bool, optional
        If True, the labels for the decoded variable will be shuffled between samples at every iteration. The default is False.

    Returns
    -------
    svm_acc : 1D array
        Classification accuracy in each bootstrap sample. Its length equal reps.
    pred_vs_monitoredvar : DataFrame
        Each row contains one sample in the test set of each bootstrap sample, the real and predicted value of the decoded_var,
        the bootstrap sample it originates from and the value of any other variable that was monitored.
    weights : 2D array
        Weights given to each feature at each bootstrap sample. It has shape reps x nfeatures.

    """

    labels =  labels_vec[decoded_var].astype(str).to_numpy()
    conditional_labels = labels_vec[conditional_var].astype(str).to_numpy()
    monitored_vec = labels_vec.drop([decoded_var,conditional_var], axis=1)    

    nclasses = len(np.unique(labels))
    nconditions = len(np.unique(conditional_labels))
    conditions = np.unique(conditional_labels)
    svm_acc = np.zeros(reps)
    weights = {} 
    for cond in conditions:
        if nclasses == 2: 
            weights[cond] = np.zeros((reps,np.shape(data)[1]))
            
        else:
            dims = int((nclasses*(nclasses-1)) /2) #from the sklearn svm documentation
            weights[cond] = np.zeros((reps,dims,np.shape(data)[1]))
    
    df = pd.DataFrame(data)

    combined_labels = labels+conditional_labels #just for the straified bootstrap to make sure to sample from all position/cat combinations

    for rep in range(reps):
        #do train/test split
        correct_nclasses = False
        while correct_nclasses == False: #in case the bootstrapping generate samples missing a class
            X_train, X_test = train_test_split(df, test_size=0.3, shuffle=True, stratify=combined_labels)
            
            #the bootstrap sampling has to come second since there is replacement, to make sure values are not both in train and test
            data_train = resample(X_train, replace=True, n_samples=int(sample_fraction*X_train.shape[0]), stratify=combined_labels[X_train.index]) #resample with replacement, keeping label proportions
            data_test = resample(X_test, replace=True, n_samples=int(sample_fraction*X_test.shape[0]), stratify=combined_labels[X_test.index])
            label_train, label_test = labels[data_train.index], labels[data_test.index]
            cond_label_train, cond_label_test = conditional_labels[data_train.index], conditional_labels[data_test.index]
            correct_nclasses = True
            
            if len(np.unique(label_train+cond_label_train)) != nclasses*nconditions: #make sure that it trained in all classes
                correct_nclasses = False
                #attention: the test set might not have all classes!!!

        
        svm_models = [[] for _ in range(len(conditions))]
        for c, cond in enumerate(conditions):
            subdata_train, subdata_test = data_train[cond_label_train==cond], data_test[cond_label_test==cond]
            sublabel_train, sublabel_test = label_train[cond_label_train==cond], label_test[cond_label_test==cond]
            svm_models[c] = svm.SVC(kernel='linear')
            svm_models[c].fit(subdata_train, sublabel_train)
            if subdata_test.shape[0]!=0: #if test set has this position included
                if c==0 and rep==0: #only c==0 for other version
                    predictions = svm_models[c].predict(subdata_test)
                    real_labels = sublabel_test
                    mon_labels = monitored_vec[cond_label_train==cond]
                    cond_labels_result = [cond]*len(sublabel_test)
                    rep_num = [rep]*len(sublabel_test)
                else:
                    predictions = np.concatenate((predictions,svm_models[c].predict(subdata_test)))
                    real_labels = np.concatenate((real_labels, sublabel_test)) #just to be in the same order as the predictions
                    mon_labels = pd.concat([mon_labels, monitored_vec[cond_label_train==cond]], ignore_index=True) #just to be in the same order as the predictions
                    cond_labels_result.extend([cond]*len(sublabel_test))
                    rep_num.extend([rep]*len(sublabel_test))
            weights[cond][rep] = svm_models[c].coef_
    pred_df = pd.DataFrame({'Real_label':real_labels,'Pred_label':predictions,'Conditional':cond_labels_result, 'Boot_rep':rep_num})
    pred_vs_monitoredvar = pd.concat([pred_df,mon_labels],axis=1)

    return svm_acc, pred_vs_monitoredvar, weights

#%%

def SVM_both_perPC(data, labels_vec, decoded_var, conditional_var, reps=100, sample_fraction=1,maxpc_fordec=False, skip=(0,0)):
    """
    This function runs the global and conditional decoder repeatedly using bootstrapping. It runs automatically on the original data set, 
    as well as on shuffled versions (shuffled decoded or conditional variable). The samples used on each decoder are the same at every
    bootstrap iteration. For every sample it tries to decode the decoded_var using an increasing number of features in the data, which are
    expected to be order from first to last principal component.

    Parameters
    ----------
    data : 2D array
        Array containing the neural population vectors. It should have shape nsamples x nfeatures, where nfeatures 
        is the number of principal components in the data.
    labels_vec : DataFrame
        Pandas DataFrame containing all the labels associated with each population vector in data. Should contain at least two columns
        one of which is the decoded variable. The remaining variables will just be monitored but not used by the decoder.
    decoded_var : str
        Variable to be decoded. It should match the name of one of the columns in labels_vec
    conditional_var: str
        Variable to condition the decoding on. The data will be split according to the value of this variable and the decoding of
        decoded_var will happen separately in each subset. It should match the name of one of the columns in labels_vec.
    reps : int, optional
        Number of bootstrap cycles. The default is 100.
    sample_fraction : float or int, optional
        The size of each bootstrap sample in relation to the full data, so needs to be between 0 and 1. The default is 1.
    maxpc_fordec : int, optional
        If provided, this is the maximum number of principal components that will be used for decoding. The default is False.
    skip : tuple, optional
        Fitting using these quantities of principal components will be skipped. Skipping makes the code faster if you don't need
        this data. This value defines a range to skip (start,end). The default is (0,0).

    Returns
    -------
    test_set_labels : DataFrame
        Pandas DataFrame containing information of original and predicted label for the test set samples from every bootstrap iteration,
        for each of the decoders, original and shuffled data.

    """

    def single_dec(datatrain, datatest, labeltrain, labeltest):
        svm_model = svm.SVC(kernel='linear')
        svm_model.fit(datatrain, labeltrain)
        predictions = svm_model.predict(datatest)
        #svm_acc = accuracy_score(labeltest,predictions)
        single_df = pd.DataFrame({'Real_label':labeltest,'Pred_label':predictions})
        return single_df
    
    def cond_dec(datatrain, datatest, labeltrain, labeltest, condlabeltrain, condlabeltest, shuffle_labels=False):
        
        if shuffle_labels==True:
            nclasses_ok = False
            while nclasses_ok == False:
                new_labeltrain = np.random.permutation(label_train)
                #make sure that the shuffle has all label/condition combinations, since you are only shuffling labels
                if len(np.unique(labeltrain+condlabeltrain)) == len(np.unique(new_labeltrain+condlabeltrain)):  
                       nclasses_ok = True
                       labeltrain = new_labeltrain                       
        
        svm_models = [[] for _ in range(len(np.unique(condlabeltrain)))]
        for c, cond in enumerate(conditions):
            subdata_train, subdata_test = datatrain[condlabeltrain==cond], datatest[condlabeltest==cond]
            sublabel_train, sublabel_test = labeltrain[condlabeltrain==cond], labeltest[condlabeltest==cond]
            svm_models[c] = svm.SVC(kernel='linear')
            svm_models[c].fit(subdata_train, sublabel_train)
            if subdata_test.shape[0]!=0: #if test set has this position included
                if c==0:
                    predictions = svm_models[c].predict(subdata_test)
                    real_labels = sublabel_test
                    cond_label_result = np.array([cond]*len(sublabel_test))
                else:
                    predictions = np.concatenate((predictions,svm_models[c].predict(subdata_test)))
                    real_labels = np.concatenate((real_labels, sublabel_test)) #just to be in the same order as the predictions
                    cond_label_result = np.concatenate((cond_label_result,np.array([cond]*len(sublabel_test))))
        #svm_acc = accuracy_score(real_labels,predictions)
        cond_df = pd.DataFrame({'Real_label':real_labels,'Pred_label':predictions, 'Cond_label':cond_label_result})
        return cond_df

    labels =  labels_vec[decoded_var].to_numpy()
    conditional_labels = labels_vec[conditional_var].to_numpy()

    nclasses = len(np.unique(labels))
    nconditions = len(np.unique(conditional_labels))
    conditions = np.unique(conditional_labels)    
    df = pd.DataFrame(data)
    combined_labels = labels+conditional_labels #just for the stratified bootstrap to make sure to sample from all position/cat combinations

    if maxpc_fordec==False:
        maxpc_fordec = data.shape[1]
    
    test_set_labels = pd.DataFrame(columns=['Real_label','Pred_label','Cond_label','Boot_rep', 'PC', 'Decoder'])

    for rep in range(reps):
        if rep%10==0:
            print(f'{rep}/{reps}')
        #do train/test split
        correct_nclasses = False
        while correct_nclasses == False: #in case the bootstrapping generate samples missing a class
            X_train, X_test = train_test_split(df, test_size=0.3, shuffle=True, stratify=combined_labels)
            
            #the bootstrap sampling has to come second since there is replacement, to make sure values are not both in train and test
            data_train = resample(X_train, replace=True, n_samples=int(sample_fraction*X_train.shape[0]), stratify=combined_labels[X_train.index]) #resample with replacement, keeping label proportions
            data_test = resample(X_test, replace=True, n_samples=int(sample_fraction*X_test.shape[0]), stratify=combined_labels[X_test.index])
            label_train, label_test = labels[data_train.index], labels[data_test.index]
            cond_label_train, cond_label_test = conditional_labels[data_train.index], conditional_labels[data_test.index]
            correct_nclasses = True
            
            if len(np.unique(label_train+cond_label_train)) != nclasses*nconditions: #make sure that it trained in all classes
                correct_nclasses = False
                #attention: intentionally, the test set might not have all classes!!!

        nclasses_ok = False
        while nclasses_ok == False:
            shuf_condlabels = np.random.permutation(cond_label_train)
            #make sure that the shuffle has all label/condition combinations, since you are only shuffling labels
            if len(np.unique(label_train+shuf_condlabels)) == len(np.unique(label_train+shuf_condlabels)):  
                   nclasses_ok = True 
                   
        for pc in range(1,maxpc_fordec+1):
            if pc in range(skip[0],skip[1]):
                continue
            df_single = single_dec(data_train.iloc[:,0:pc], data_test.iloc[:,0:pc], label_train, label_test)
            df1 = pd.DataFrame({'Real_label':df_single['Real_label'],'Pred_label':df_single['Pred_label'],'Cond_label':cond_label_test,\
                                                     'Boot_rep':[rep]*len(label_test),'PC':[pc]*len(label_test),'Decoder':['single']*len(label_test)})
            
            df_single_shuffle = single_dec(data_train.iloc[:,0:pc], data_test.iloc[:,0:pc], np.random.permutation(label_train), label_test)
            df2 = pd.DataFrame({'Real_label':df_single_shuffle['Real_label'],'Pred_label':df_single_shuffle['Pred_label'],'Cond_label':cond_label_test,\
                                                     'Boot_rep':[rep]*len(label_test), 'PC':[pc]*len(label_test),'Decoder':['single_decshuffle']*len(label_test)})
                
            df_cond = cond_dec(data_train.iloc[:,0:pc], data_test.iloc[:,0:pc], label_train, label_test, cond_label_train, cond_label_test)
            df3 = pd.DataFrame({'Real_label':df_cond['Real_label'],'Pred_label':df_cond['Pred_label'],'Cond_label':df_cond['Cond_label'],\
                                                     'Boot_rep':[rep]*len(label_test),'PC':[pc]*len(label_test), 'Decoder':['cond']*len(label_test)})
            
            df_cond_shufdec = cond_dec(data_train.iloc[:,0:pc], data_test.iloc[:,0:pc], label_train, label_test, cond_label_train, cond_label_test, shuffle_labels=True)  
            df4 = pd.DataFrame({'Real_label':df_cond_shufdec['Real_label'],'Pred_label':df_cond_shufdec['Pred_label'],'Cond_label':df_cond_shufdec['Cond_label'],\
                                                     'Boot_rep':[rep]*len(label_test), 'PC':[pc]*len(label_test),'Decoder':['cond_decshuffle']*len(label_test)})
            df_cond_shufcond = cond_dec(data_train.iloc[:,0:pc], data_test.iloc[:,0:pc], label_train, label_test, shuf_condlabels, cond_label_test)  
            df5 = pd.DataFrame({'Real_label':df_cond_shufcond['Real_label'],'Pred_label':df_cond_shufcond['Pred_label'],'Cond_label':df_cond_shufcond['Cond_label'],\
                                                     'Boot_rep':[rep]*len(label_test), 'PC':[pc]*len(label_test), 'Decoder':['cond_condshuffle']*len(label_test)})
            
            test_set_labels = pd.concat([test_set_labels, df1, df2, df3, df4, df5], ignore_index=True)

    print('Done')
    return test_set_labels

#%%

def SVM_sequential(data, labels_vec, decoded_var, conditional_var, reps=100, parts=5, sample_fraction=1):
    """
    This decoder fuctions in two steps: First it does a global SVM decoding of the conditional_var, using k-fold cross validation.
    Second, it uses the predicted values from the test set of the k-fold for the conditional decoding of the decoded_var. This second
    step is calculated the same as SVM_conditional.

    Parameters
    ----------
    data : 2D array
        Array containing the neural population vectors. It should have shape nsamples x nfeatures, where nfeatures 
        is the number of principal components in the data.
    labels_vec : DataFrame
        Pandas DataFrame containing all the labels associated with each population vector in data. Should contain at least two columns
        one of which is the decoded variable. The remaining variables will just be monitored but not used by the decoder.
    decoded_var : str
        Variable to be decoded. It should match the name of one of the columns in labels_vec
    conditional_var: str
        Variable to condition the decoding on. The data will be split according to the value of this variable and the decoding of
        decoded_var will happen separately in each subset. It should match the name of one of the columns in labels_vec.
    reps : int, optional
        Number of bootstrap cycles. The default is 100.
    parts : int, optional
        The first fit is done using k-fold instead of bootstrapping to assure a prediction for every data point. This is the number
        of folds to split the data. The default is 5.
    sample_fraction : float or int, optional
        The size of each bootstrap sample in relation to the full data, so needs to be between 0 and 1. The default is 1.

    Returns
    -------
    svm_acc : 1D array
        Classification accuracy in each bootstrap sample. Its length equal reps.
    prediction_df : DataFrame
        Each row contains one sample in the test set of each bootstrap sample, the real and predicted value of each of the two variables
        decoded sequentially.
    weights : 2D array
        Weights given to each feature at each bootstrap sample. It has shape reps x nfeatures.

    """

    labels =  labels_vec[decoded_var].astype(str).to_numpy()
    conditional_labels = labels_vec[conditional_var].astype(str).to_numpy()

    nclasses = len(np.unique(labels))
    nconditions = len(np.unique(conditional_labels))
    conditions = np.unique(conditional_labels)
    svm_acc = np.zeros(reps)
    combined_labels = labels+conditional_labels #just for the straified bootstrap to make sure to sample from all position/cat combinations
    
    weights = {} #added as dict on 24.05.2024
    for cond in conditions:
        if nclasses == 2: #if statement added 19.01.23
            weights[cond] = np.zeros((reps,np.shape(data)[1]))
            
        else:
            dims = int((nclasses*(nclasses-1)) /2) #from the sklearn svm documentation
            weights[cond] = np.zeros((reps,dims,np.shape(data)[1]))
    
    df = pd.DataFrame(data)
    
    #global decoding of conditional variable - k-fold
    splitdata = StratifiedKFold(n_splits=parts, shuffle=True)
    splitdata.get_n_splits(df, combined_labels) #inputs are ignored, are just there for compatibility
    
    inferred_condlabels=-np.ones(conditional_labels.shape, dtype=object)    
    for train_index, test_index in splitdata.split(df, conditional_labels):
        data_train, data_test = df.iloc[train_index], df.iloc[test_index]
        label_train, label_test =  conditional_labels[train_index], conditional_labels[test_index]
        svm_model = svm.SVC(kernel='linear')
        svm_model.fit(data_train, label_train)
        pred = svm_model.predict(data_test)
        inferred_condlabels[test_index] = pred

    for rep in range(reps):
        if rep%10==0:
            print(f'{rep}/{reps}')
        #do train/test split
        correct_nclasses = False
        while correct_nclasses == False: #in case the bootstrapping generate samples missing a class
            X_train, X_test = train_test_split(df, test_size=0.3, shuffle=True, stratify=combined_labels)
            
            #the bootstrap sampling has to come second since there is replacement, to make sure values are not both in train and test
            data_train = resample(X_train, replace=True, n_samples=int(sample_fraction*X_train.shape[0]), stratify=combined_labels[X_train.index]) #resample with replacement, keeping label proportions
            data_test = resample(X_test, replace=True, n_samples=int(sample_fraction*X_test.shape[0]), stratify=combined_labels[X_test.index])
            label_train, label_test = labels[data_train.index], labels[data_test.index]
            cond_label_train, cond_label_test = inferred_condlabels[data_train.index], inferred_condlabels[data_test.index]
            #cond_label_train, cond_label_test = conditional_labels[data_train.index], conditional_labels[data_test.index]
            real_cond_test = conditional_labels[data_test.index]
            correct_nclasses = True
            
            if len(np.unique(label_train+cond_label_train)) != nclasses*nconditions: #make sure that it trained in all classes
                correct_nclasses = False
                #attention: intentionally, the test set might not have all classes!!!
        
        svm_models = [[] for _ in range(len(conditions))]
        for c, cond in enumerate(conditions):
            subdata_train, subdata_test = data_train[cond_label_train==cond], data_test[cond_label_test==cond]
            sublabel_train, sublabel_test = label_train[cond_label_train==cond], label_test[cond_label_test==cond]
            svm_models[c] = svm.SVC(kernel='linear')
            svm_models[c].fit(subdata_train, sublabel_train)
            if subdata_test.shape[0]!=0: #if test set has this position included
                if c==0 and rep==0: #only c==0 for other version
                    predictions = svm_models[c].predict(subdata_test)
                    real_labels = sublabel_test
                    cond_labels_result = [cond]*len(sublabel_test)
                    real_cond_labels = real_cond_test[cond_label_test==cond]
                    rep_num = [rep]*len(sublabel_test)
                else:
                    predictions = np.concatenate((predictions,svm_models[c].predict(subdata_test)))
                    real_labels = np.concatenate((real_labels, sublabel_test)) #just to be in the same order as the predictions
                    cond_labels_result.extend([cond]*len(sublabel_test))
                    rep_num.extend([rep]*len(sublabel_test))
                    real_cond_labels = np.concatenate((real_cond_labels,real_cond_test[cond_label_test==cond]))
            weights[cond][rep] = svm_models[c].coef_

    predict_df = pd.DataFrame({'Real_label':real_labels,'Pred_label':predictions,'Cond_infer':cond_labels_result, 'Cond_real':real_cond_labels, 'Boot_rep':rep_num})

    return svm_acc, predict_df, weights