#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code from paper Chiossi et al, 2024
Source: https://github.com/hchiossi/hpc-hierarchy

"""

import json
import numpy as np
import func_popanalysis as pop
import func_decoder as decoder
from func_behaviour import load_behaviour
import os
from datetime import datetime

## With this script you can run any of the decoder types, by changing the user-defined variables below
## Make sure you have all the necessary data: behaviour files + ClusterPopulation and/or GLM fitted data
## This will automatically save an .csv file with all the predictions vs real label for each bootstrap run of the chosen decoder

#%%User-defined variables

#Options: 'global', 'conditional', 'both_fromPCA', or 'sequential'
decoder_type = 'global' 

#number of repetitions during bootstrapping
boot_reps = 100

#Options: 'Context', 'Side', 'Position',
#'All' (only valid with 'global' and 'conditional'), #'Merged' (only valid with 'global')
decoded_var = 'Context'

#if specific decoded var is used and decoder type in 'conditional', this is needed
conditional_var = 'Position' 

#If you use the 'all' option, define the variables here in a list
all_vars = ['Context','Side','Position']

#If you want to do decoding on the original data or the GLM speed-regressed data
data_source = 'original' #'original' or 'GLM'

#If doing the decoding on GLM data, also define these
pos_binsize=4 #in cm, the size of the spatial bins used in the GLM
n_spdbins=10 #number of speed bins used in the GLM calculation

#File containing all the experimental information, see gen_experiment_info_dict.py
info_file ='/example/hpc-hierarchy/fullexp_info.json'

#Path to where GLM parameters are saved
glm_path = '/example/GLMfit/'

#Path to export decoding results. A new folder will be created inside
outpath = '/example/decoder/' 

#%% Load experimental info

with open(info_file, 'r') as openfile: 
   exp_info = json.load(openfile)

animals, dates = exp_info['animals'], exp_info['dates']

#Create outpath folder with today's date
today = str(datetime.now().date()) 
outpath=outpath+f'{today}/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

#%% Main loop
for a, animal in enumerate(animals):
    
    pos_edges = np.array(exp_info['position_edges'][a])/pos_binsize
    
    for d,date in enumerate(dates[a]):
        
        print(f'{animal} {date} : Start {datetime.now().strftime("%H:%M")}')

        date_idx = a*len(exp_info['animals'])+d
        
        # Load behavior data        
        behaviour = load_behaviour(exp_info['bhv_path'], [animal], [[date]])
        behaviour= behaviour.iloc[exp_info['ntrials'][a][d]]
        
        ntrials=exp_info['ntrials'][a][d]
        nposbins=int(exp_info['max_pos']/pos_binsize)

        #Population vectors per position are equivalent to stacking ratemaps per trial of each cell.
        #These were pre-calculated both in the original data when generating a ClusterPopulation or when calculating the GLM
        pop_vec_flat = pop.load_pvs(a,d, data_source, exp_info,pos_binsize,glm_path)
        projections = pop_vec_flat #for compatibility, this is replaced if PCA is calculated
        max_pcs_fordec = pop_vec_flat.shape[1] #for compatibility, this is replaced if PCA is calculated
  
        #Get the labels for each PV
        #you can potentially prune columns that are not of your interest before continuing
        vec_labels = pop.get_vec_labels(behaviour, ntrials, pos_edges)        
    
        #Optionally, run PCA on the data and do analysis on the projections instead of the pop_vec_flat
        # projections, var_explained = pop.pca_projections(pop_vec_flat, cum_var=True)
        # nPCs = projections.shape[1]
        # max_pcs_fordec = nPCs #you can change this if you don't need to decode everything. The code will be faster then.

    
        #%% Run the decoder        
        
        if decoder_type=='global':            
            if decoded_var=='All':                      
                for current in all_vars:
                    #it outputs a dataframe that keeps track of the value of all variables in the test set, including those not decoded
                    _, prediction_df, svm_weights = decoder.SVM_global(pop_vec_flat, vec_labels, current, shuffle_labels=False)
                    filename = outpath+f"{decoder_type}_{data_source}_{current}_day{date_idx}" 
                    prediction_df.to_csv(filename+'.csv', index=False)
                    np.save(filename+'_weights', svm_weights)
            
            else:
                if decoded_var=='Merged':
                    vec_labels['Merged'] = vec_labels.astype(str)[all_vars].agg(''.join, axis=1)
         
                elif decoded_var not in vec_labels.columns:
                    raise Exception('Decoded variable unknown')
   
                _, prediction_df, svm_weights = decoder.SVM_global(pop_vec_flat, vec_labels, decoded_var, shuffle_labels=False)                     
                filename = outpath+f"{decoder_type}_{data_source}_{decoded_var}_day{date_idx}"
                prediction_df.to_csv(filename+'.csv', index=False)
                np.save(filename+'_weights', svm_weights)

        elif decoder_type=='conditional':
            if decoded_var =='All': #it will run all pairs of possible decoded and conditional variables
                for current_var in all_vars:
                    for current_cond in all_vars:
                        if current_var != current_cond:                            
                            _, prediction_df, svm_weights = decoder.SVM_conditional(pop_vec_flat, vec_labels, current_var, current_cond, reps=boot_reps)
                            filename = outpath+f"{decoder_type}_{data_source}_{current_var}_conditional_on_{current_cond}_day{date_idx}"
                            prediction_df.to_csv(filename+'.csv', index=False)
                            np.save(filename+'_weights', svm_weights)
            
            elif (decoded_var not in vec_labels.columns) or (conditional_var not in vec_labels.columns):
                raise Exception('Decoded or conditional variable unknown')
            
            else:                
                _, prediction_df, svm_weights = decoder.SVM_conditional(pop_vec_flat, vec_labels, decoded_var, conditional_var, reps=boot_reps)
                filename = outpath+f"{decoder_type}_{data_source}_{decoded_var}_conditional_on_{conditional_var}_day{date_idx}"
                prediction_df.to_csv(filename+'.csv', index=False)
                np.save(filename+'_weights', svm_weights)
                
                
        elif decoder_type=='both_fromPCA':
            #this function does not output the weights because it runs too many fits
            #it calculates global and conditional decoding for all requested PCs + shuffles
            #the data used in every bootstrap repetition is the same for all decoders
            prediction_df = decoder.SVM_both_perPC(projections, vec_labels, decoded_var, conditional_var, reps=boot_reps,\
                                                   maxpc_fordec=max_pcs_fordec, skip=(0,0))
            filename = outpath+f"{decoder_type}_{data_source}_{decoded_var}_conditional_on_{conditional_var}_day{date_idx}"
            prediction_df.to_csv(filename+'.csv', index=False)
            
        elif decoder_type=='sequential':
            _, prediction_df, svm_weights = decoder.SVM_sequential(pop_vec_flat, vec_labels, decoded_var, conditional_var,\
                                                                   pos_edges=pos_edges,reps=boot_reps)
            filename = outpath+f"{decoder_type}_{data_source}_1st{conditional_var}_2nd{decoded_var}_day{date_idx}"# 
            prediction_df.to_csv(filename+'.csv', index=False)
            np.save(filename+'_weights', svm_weights)
            
        else:
            raise Exception('Decoder type unknown')
