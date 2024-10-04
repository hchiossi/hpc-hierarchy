#!/usr/bin/env python
# coding: utf-8

'''
Code from paper Chiossi et al, 2024
Source: https://github.com/hchiossi/hpc-hierarchy

This script creates files in the format they are used in the GLM pipeline:
list of positions, speeds, spikes, and trial number for each time bin, 
filtered by periods where animal was in the maze and running
For this script you need: 
    file with experimental info (from gen_experiment_info_dict.py)
    a cluster population object (.pkl) for each recording day (from class_cluster.py)
    a session_shifts file per recording day (termination frame value of each trial or sleep session)
    animal position (.lwhl) and speed (.spd), linearized (see func_basics for linealization functions)
'''

import os
import json
import pickle
import numpy as np
import func_basics as basics
import func_mazetime as mazetime
from func_behaviour import load_behaviour  

#%% User-defined variables

#size of the position bins to compress - too small might lead to GLM overfitting
pos_binsize = 4 #in cm

#number of equally populated speed bins to which the data will be discretized
n_spdbins = 10

#speed threshold above which data should be used
spd_thres = 3 #in cm/s

#where the prepared data will be exported
outpath = '/example/ratemapsGLM'

#File containing all the experimental information, see gen_experiment_info_dict.
info_file ='/example/hpc-hierarchy/fullexp_info.json'


#%% function to remove numbers in list if some is missing, so if there is 1 3 5 just go 1 2 3
def remove_gaps(w):
    v = np.array(w).astype(int)
    v[v==np.min(v)] = 0
    for i in range(len(v)-1):
        if v[i+1] - v[i] > 1:
            v[i+1:] -= v[i+1] - v[i] - 1
    return v+1 # add the +1 for matlab indexing


#%% Load experimental variables
with open(info_file, 'r') as openfile: 
   exp_info = json.load(openfile)

animals, dates = exp_info['animals'], exp_info['dates']

#%% Main loop
for a, animal in enumerate(animals):
    for d,date in enumerate(dates[a]):
        # Load data
        print(f'-------- {animal} {date} --------')
        print('Loading data')
        
        #load pre-generated ClusterPopulation object
        with open(f"{exp_info['pop_path']}{animal}_{date}_population.pkl", 'rb') as file:
              all_cells = pickle.load(file)
        
        #load behavior
        behaviour = load_behaviour(exp_info['bhv_path'], [animal], [[date]])
        
        #load all other data for this recording session, such as linearized positions, speed, reward locations, etc
        rs = basics.RecordingSession(a, d, exp_info)
        
        #%% Get spike matrix aligned to the whl
        print('Formating vectors')
        
        spkmat = np.zeros((all_cells.nclusters,rs.lwhl.shape[0]))
        
        for cell in range(all_cells.nclusters):
            spktimes, counts = np.unique(all_cells.unit[cell].spiketimes, return_counts=True)
            spkmat[cell,spktimes]+=1
            #loop for the bins with more than one spike
            for idx in np.where(counts>1)[0]:
                stime = spktimes[idx]
                spkmat[cell,stime]=counts[idx] #replace the 1 by the actual number of spikes
        
        
        #%% make start side and ctx vectors with size = to whl
        ctxvec = np.zeros(np.size(rs.lwhl))
        sidevec = np.zeros(np.size(rs.lwhl))
        
        ctxvec[:] = np.nan
        sidevec[:] = np.nan
        inmaze = np.zeros((rs.ntrials,2),dtype=int)
        
        ctx_dict = {'A': 1, 'B': 2}
        start_dict = {'L': -1, 'R': 1}
        
        for trial in range(rs.ntrials):
            ctx, side = behaviour.iloc[trial]['Context'], behaviour.iloc[trial]['Start']
            start, end = mazetime.lin_inmaze_time(rs.lwhl, rs.whl_shifts[trial], rs.whl_shifts[trial+1])
            inmaze[trial,:]=[start,end]
            ctxvec[start:end] = ctx_dict[ctx]
            sidevec[start:end] = start_dict[side]
        print(inmaze)
        
        #%% Filter only periods in the maze (exlude inter-trial intervals)
        # and break into trials
        
        # filter pyramidal cells
        p1binspk = spkmat[all_cells.cell_types == 'p1']
        
        whl_trials = []
        speed_trials = []
        p1spk_trials = []
        trialnum = []
        side_trials = []
        movdir_trials = []
        ctx_trials = []
        
        for trial, (beg, end) in enumerate(inmaze):
            whl_trials+=rs.lwhl[beg:end].tolist()
            speed_trials+=rs.linspeed[beg:end].tolist()
            p1spk_trials.append(p1binspk[:,beg:end])
            trialnum.append(np.ones(end-beg)*trial)
            side_trials.append(sidevec[beg:end])
            movdir_trials.append(rs.movdir[beg:end])
            ctx_trials.append(ctxvec[beg:end])
        
        # turn into numpy arrays
        whl_trials = np.array(whl_trials)
        speed_trials = np.array(speed_trials)
        p1spk_trials = np.hstack(p1spk_trials)
        trialnum = np.hstack(trialnum)
        side_trials = np.hstack(side_trials)
        movdir_trials = np.hstack(movdir_trials)
        ctx_trials = np.hstack(ctx_trials)
        

        #%%
        
        # speed filter  - 3cm/s - and keep only where mov direction same as side
        speed_filter = (speed_trials > spd_thres) * (movdir_trials == side_trials)
        whl_trials = whl_trials[speed_filter]
        p1spk_trials = p1spk_trials[:,speed_filter]
        speed_trials = speed_trials[speed_filter]
        trialnum = trialnum[speed_filter]
        side_trials = side_trials[speed_filter]
        ctx_trials = ctx_trials[speed_filter]
        
        
        # In[8]:        
               
        # bin position        
        bins = np.arange(0,exp_info['max_pos']+1,pos_binsize)
        
        # binning digitize
        whlbin = np.digitize(whl_trials, bins)
        
        
        # In[9]:        
        
        # bin speed_trials into n_bins (equally populated) bins        
        bins = np.percentile(speed_trials, np.linspace(0,100,n_spdbins+1))
        bins[-1]+=1
        speedbin = np.digitize(speed_trials, bins)
        speedbin[speedbin>n_spdbins]=n_spdbins #otherwise highest value gets assigned to an extra bin
              
        
        # In[11]:
        print('Saving files')    
            
        # #outpath = f'/helo3/analysis/ratemapsGLM/pos{pos_binsize}_spd{spd_binsize}/{animal}/{date}/' 
        outpath_session = outpath+f'/pos{pos_binsize}_spd{n_spdbins}perc/{animal}/{date}/' 
        if not os.path.exists(outpath_session):
            os.makedirs(outpath_session)
            
        # save files
        # np.savetxt(f'{outpath}positions.all', whlbin)
        # np.savetxt(f'{outpath}spikes.all', p1spk_trials)
        # np.savetxt(f'{outpath}trials.all', trialnum)
        # np.savetxt(f'{outpath}speeds.all', speedbin)
