#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code from paper Chiossi et al, 2024
Source: https://github.com/hchiossi/hpc-hierarchy
This script calculates the hierarchy between population vectors calculated
as the average firing rate of each cell for each position and trial
"""

import pickle
import json
import numpy as np
import func_basics as basics
import func_popanalysis as pop
from scipy.cluster import hierarchy
from func_behaviour import load_behaviour
import matplotlib.pyplot as plt
import pandas as pd
import func_mazetime as mazetime


#%%User-defined variables

#File containing all the experimental information, see gen_experiment_info_dict.py
info_file ='/example/hpc-hierarchy/fullexp_info.json'

#If you will include speed as a variable
binsize=4 #position bin size, in cm
spd_nperc=10 #number of speed bins
spd_thrs=3 #speed threshold, in cm/s as used for ratemap calculation

#%%
def hierarchy_to_dataframe(hierarchy_array):
    #Make a pandas DataFrame from the hierarchical tree, where each row corresponds to a node
    #The first two coluns of a row will be NaN if a node does not have children (it is a leaf)
    
    nleaves = hierarchy.leaves_list(hierarchy_array).size
    empty = np.zeros((nleaves,2))
    empty[:]=np.nan
    linkage_df = pd.DataFrame(empty,columns=['Left_node','Right_node'])
    temp_df = pd.DataFrame(hierarchy_array[:,:2], columns=['Left_node','Right_node'])
    linkage_df = pd.concat([linkage_df,temp_df],ignore_index=True)
    ntotal = len(linkage_df) #total = nleaves + nnodes
    linkage_df['node_depth'] = np.zeros(nleaves+hierarchy_array.shape[0])
    linkage_df['children'] = [[] for _ in range(ntotal)]
    
    #first first level nodes
    idxs = linkage_df[(linkage_df['Left_node']<nleaves) &(linkage_df['Right_node']<nleaves)].index
    linkage_df.loc[idxs[idxs>=nleaves],'node_depth']=1
    for idx in idxs:
        linkage_df.at[idx,'children']=[int(linkage_df.iloc[idx]['Left_node']),int(linkage_df.iloc[idx]['Right_node'])]
    
    #deeper level nodes
    depth=2
    while (linkage_df.iloc[nleaves:]['node_depth']==0).any(): #none of the non-leaf nodes should have depth zero
        temp_df = linkage_df.copy()
        for idx in range(nleaves,ntotal):        
            if linkage_df.iloc[idx]['node_depth']==0: #if depth not yet found
                left = int(linkage_df.iloc[idx]['Left_node'])
                right = int(linkage_df.iloc[idx]['Right_node'])
                #if left/right is a node (not leaf) with non-classified depth, don't use it yet
                if (left>=nleaves and linkage_df.iloc[left]['node_depth']==0) or (right>=nleaves and linkage_df.iloc[right]['node_depth']==0):
                    continue
                else:
                    temp_df.loc[idx,'node_depth']=depth
                    if left<nleaves:
                        temp_df.at[idx,'children'] = [left]+linkage_df.iloc[right]['children']
                    elif right<nleaves:
                        temp_df.at[idx,'children'] = linkage_df.iloc[left]['children']+[right]
                    else:
                        temp_df.at[idx,'children'] = linkage_df.iloc[left]['children']+linkage_df.iloc[right]['children']
        linkage_df = temp_df
        depth+=1
    
    return linkage_df


#%% Load experimental data and pre set variables 

with open(info_file, 'r') as openfile: 
   exp_info = json.load(openfile)

animals, dates = exp_info['animals'], exp_info['dates']

ctx_dict = {'A':0, 'B':1}
side_dict = {'L':0, 'R':1}

labels_list = ['Ctx', 'Side','Pos', 'Speed']
mean_var = {i:[] for i in labels_list}

#%% Main loop
for a,animal in enumerate(animals):
    for d,date in enumerate(dates[a]):
        print(animal, date)
        #%% Load data for this session
        date_idx=a*5+d
        
        #load pre-generated ClusterPopulation object
        with open(f"{exp_info['pop_path']}{animal}_{date}_population.pkl", 'rb') as file:
              all_cells = pickle.load(file)
        
        #load behavior
        behaviour = load_behaviour(exp_info['bhv_path'], [animal], [[date]])
        
        #load all other data for this recording session, such as linearized positions, speed, reward locations, etc
        rs = basics.RecordingSession(a, d, exp_info)
        
        #%% Get pop. vectors
        print('Calculating PVs')
        
        
        pos_edges=exp_info['position_edges'][a] #where each position bin starts/ends, in cm
        npos = len(pos_edges)-1
        pop_vec_flat, vec_labels = pop.popvec_perpos_fromedges(all_cells, rs.lin_whl, rs.linspeed, rs.whl_shifts, pos_edges, with_labels=True,\
                                                      behaviour=behaviour, exclude_untracked=True, shift_pos=False, \
                                                          dir_filter=True, move_dir=rs.mov_dir, speed_regress=False)   
        
        pop_vec_flat = pop_vec_flat[:,all_cells.cell_types=='p1'] #filter only pyramidals
        
        
        #%% calculate velocity profile, if you want to include speed as a variable
        print('Calculating velocity')
                
        side_dict2 = {'L':-1, 'R':1}
        trial_side = np.zeros(rs.lin_whl.shape)
        side = behaviour['Start'].map(side_dict2)
        for trial in range(rs.ntrials):            
            trial_side[rs.whl_shifts[trial]:rs.whl_shifts[trial+1]]=side[trial]

        mean_speed = np.zeros((rs.ntrials,npos))
        for trial in range(rs.ntrials):
            start,end = mazetime.lin_inmaze_time(rs.lin_whl, rs.whl_shifts[trial], rs.whl_shifts[trial+1])
            speed_trial = rs.linspeed[start:end]
            whl_trial = rs.lin_whl[start:end].astype(int)
            whl_digital = np.digitize(whl_trial,pos_edges)
            for pos in range(npos):
                filtered = speed_trial[whl_digital==pos+1]
                mean_speed[trial,pos]=np.nanmean(filtered[filtered>spd_thrs])        
        
        spd_bins = np.nanpercentile(mean_speed, np.linspace(0,100,spd_nperc+1))
        spd_bins[-1]+=1
        mean_speed=np.digitize(mean_speed,spd_bins)
        
        vec_labels['Speed']= pd.Series(mean_speed.astype(int).flatten())

        #%% Do hierarchical clustering
        print('Calculating hierarchy')
        #it can take either the distance matrix in 1D or the data itself in 2D and it will calculate eucledian distance
        
        pv_hierarchy = hierarchy.linkage(pop_vec_flat, method="ward") 
        group_order = hierarchy.leaves_list(pv_hierarchy) #get leaf IDs     
        linkage_df = hierarchy_to_dataframe(pv_hierarchy)

        
        #%% Analyse label variances per depth of the tree
        
        vec_labels['Pos_rescale'] = vec_labels['Position'].astype(int)/int(vec_labels['Position'].max())
        vec_labels['Ctx_rescale'] = vec_labels['Context'].map(ctx_dict)
        vec_labels['Side_rescale'] = vec_labels['Side'].map(side_dict)
        vec_labels['Speed_rescale'] = vec_labels['Speed'].astype(int)/int(vec_labels['Speed'].max())        

        for lbl_name in labels_list:
            mean_var[lbl_name].append([])
            label = lbl_name+'_rescale'
            maxdepth = int(linkage_df['node_depth'].max())
            mean_var[lbl_name][-1] = np.zeros(maxdepth)    
            for depth in range(1,maxdepth):
                depth_var = []
                for node in np.where(linkage_df['node_depth']==depth)[0]:
                    node_children = linkage_df.iloc[node]['children']
                    children_labels = vec_labels.iloc[node_children][label]
                    depth_var.append(children_labels.var())
                mean_var[lbl_name][-1][depth]=np.mean(depth_var)    


#%% Example Plot
from scipy import stats

lbl_color = ['#4D6B80','#B56E76','darkgoldenrod','gray']
lbl_title=['Context','Direction','Position', 'Speed']

mean_perday = {}
fig, axs = plt.subplots(figsize=(3.5,4))
for l, lbl_name in enumerate(labels_list[:-1]): #do entire list if velocity is also calculated
    lengths = [i.shape[0] for i in mean_var[lbl_name]]
    mean_perday[lbl_name] = np.zeros((25,max(lengths)))
    mean_perday[lbl_name][:]=np.nan
    for i in range(len(mean_var[lbl_name])):
        mean_perday[lbl_name][i,:lengths[i]]=mean_var[lbl_name][i]/mean_var[lbl_name][i][-1]
    
    plot_mean = np.nanmean(mean_perday[lbl_name],axis=0)
    plot_sem = stats.sem(mean_perday[lbl_name],axis=0, nan_policy='omit')
    x_vals = range(plot_mean.size)
    y_vals = np.flip(plot_mean)
    
    #normal
    if lbl_name=='Speed': line='--'
    else: line='-' 
    axs.plot(x_vals,y_vals, color=lbl_color[l], linestyle=line, label=lbl_title[l])
    axs.fill_between(x_vals,y_vals+np.flip(plot_sem),y_vals-np.flip(plot_sem),alpha=0.3,color=lbl_color[l])


axs.set_xlabel('Hierarchical depth')
axs.set_ylabel('Mean in-cluster variance')
axs.set_xticks([min(x_vals),max(x_vals)])
axs.set_xticklabels(['Root','Leaves'])
axs.set_yticks([0,0.5,1])
axs.legend(title='Variable', frameon=False)
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)


