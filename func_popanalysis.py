#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code from paper Chiossi et al, 2024
Source: https://github.com/hchiossi/hpc-hierarchy

"""
import numpy as np
import pandas as pd

def pca_projections(pop_vec_flat, cum_var=False):
    #input population vector in the form nsamples x nclusters, outputs PCA projections
    
    from sklearn import decomposition
    
    pv_pca = decomposition.PCA()
    pv_pca.fit(pop_vec_flat)
    
    # get PCA projections
    projections = []
    nPCs = np.shape(pop_vec_flat)[1] #number of principal components to use for analysis
    for comp in range(nPCs):
        # data*eigenvector/eigvec norm = should give a single value per sample. Eingenvetors are already normalised
        projections.append(pop_vec_flat@pv_pca.components_[comp,:])
    
    if cum_var:
        var_explained = np.cumsum(pv_pca.explained_variance_ratio_)
        return np.array(projections).T, var_explained
    else:
        return np.array(projections).T
    

def lda_projections(pop_vec_flat,lda_var):
    #input population vector in the form nsamples x nclusters, outputs LDA projection
    #lda_var is the value of variable you want to separate using Linear Discriminar Analysis, it should have the shape nsamples
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    lda_model = LinearDiscriminantAnalysis()
    projections = lda_model.fit_transform(pop_vec_flat, lda_var)
    
    return projections

def popvec_perpos_fromedges(all_cells, lin_whl, lin_speed, whl_shifts, pos_edges, speed_thres=3, whl_rate=50, ntrials=40, \
                  with_labels=False, behaviour=None, exclude_untracked=True, shift_pos=False, dir_filter=False, move_dir=None,\
                      coarse_pos_edges=None):

    """
    
    Parameters
    ----------
    all_cells : ClusterPopulation object
        See classcluster.py for more information
    lin_whl : 1D array
        Linearized animal position. Array of lenght N, where N is the number of frames
    lin_speed : 1D array
        Linerized animal speed. Array of length where N is the number of frames. Should be the same length as lin_whl
    whl_shifts : 1D array
        End of each session file (sleep or trial) in frames, aligned with the whl
    pos_edges : list
        In case you are using coarse-grained linearized maze positions, you might want to define the edges of the bins per animal\
            to make sure a reward location does not lie in between edges. Keep all the same size and any shift should be done by\
                altering the size of the edge bins only. This list should be of size M+1, where M is the number of desired bins.\
                    It contains where each position bin starts and ends.
    speed_thres : int, optional
        Threshold for speed filtering. Datapoints below this values will be discarded. The default is 3 cm/s.
    whl_rate : int, optional
        Framerate of position tracking. The default is 50 Hz.
    ntrials : int, optional
        Number of trial per recording day. The default is 40.
    with_labels : bool, optional
        If you would like to output the labels associated with each population vector. In this case a behaviour DataFrame needs\
            to be provided. The default is False.
    behaviour : DataFrame, optional
        Pandas DataFrame containing information about each trial (one row per trial). It contains information about trial number,\
            maze start side, context, and performace (0 or 1). Only necessary if with_labels is True. The default is None.
    exclude_untracked : bool, optional
        If True, it filters out positions not tracked in a given trial. The default is True.
    shift_pos : bool, optional
        To shuffle data. If True, the place fields are randomly shifted between trials. The default is False.
    dir_filter : bool, optional
        If True, only datapoints from the main direction of the trial are used (based on the animal's start position). It requires\
                                                                                a move_dir array to be provided. The default is False.
    move_dir : 1D array, optional
        Movement direction (-1, 1) of the animal. Array of length where N is the number of frames. Should be the same length as lin_whl
    coarse_pos_edges : list, optional
        If provided, it will also output labels for a position bining defined by the bin edges in this list

    Returns
    -------
    pop_vec_flat: 2D array
        Array containing population vectors averaged per position and trial. It has shape nsamples x nclusters
    vec_labels: DataFrame, optional
        It contains the behavior variables associated with each population vectors from pop_vec_flat. Shape nsamples x nlabelcategories.

    """    

    
    pop_vec = np.zeros((ntrials, np.size(pos_edges)-1, all_cells.nclusters)) 
    speed_filter = np.where(lin_speed>speed_thres)[0]
    
    if dir_filter:
        trial_filt = np.zeros(len(lin_whl))
        side_map={'L':-1,'R':1}
        for trial in range(ntrials):
            trial_start = behaviour.iloc[trial]['Start']
            trial_filt[whl_shifts[trial]:whl_shifts[trial+1]]=side_map[trial_start] #assign 1 when R and -1 when L, for all frames
        trial_dir_filter = np.where(move_dir*trial_filt==1) #move dir * trial start will give 1 if they match
    
    for p in range(len(pos_edges)-1):
        track_times = np.where(np.logical_and(lin_whl>(pos_edges[p]),lin_whl<=(pos_edges[p+1])))[0]
        track_times = np.intersect1d(track_times, speed_filter)
        if dir_filter:
            track_times = np.intersect1d(track_times,trial_dir_filter)
        for cellid in range(all_cells.nclusters):
            spk_times = all_cells.unit[cellid].spiketimes
            cell_inpos = np.intersect1d(track_times, spk_times)
            for trial in range(ntrials):               
                time_in_pos = np.sum(np.logical_and(track_times>whl_shifts[trial], track_times<whl_shifts[trial+1]))/whl_rate         
                pop_vec[trial,p,cellid]= (np.sum(np.logical_and(cell_inpos>whl_shifts[trial], cell_inpos<whl_shifts[trial+1])))/time_in_pos

    if shift_pos: #shuffle effect for statistical comparison - it shifts the placefields between trials
        pop_vec_shifted = pop_vec.copy()
        for trial in range(ntrials):
            shift = np.random.choice(range(np.size(pos_edges)-1))
            pop_vec_shifted[trial,:,:] = np.roll(pop_vec[trial,:,:],shift, axis=0)
        pop_vec_flat = pop_vec_shifted.reshape(-1,all_cells.nclusters)        
    else:
        pop_vec_flat = pop_vec.reshape(-1,all_cells.nclusters)
        
    if exclude_untracked: 
        nan_row = np.where(np.isnan(pop_vec_flat[:,0]))[0]
        if len(nan_row)>0:
            pop_vec_flat = np.delete(pop_vec_flat,nan_row,0)

    if with_labels:
        #store datapoint labels       
        vec_labels = pd.DataFrame(columns=['Trial#','Position','Side','Context','Category','Correct','Error_type']) #label per datapoint, to be used for decoding
        for trial in range(ntrials):
            trial_cat = behaviour.iloc[trial]['Start']+behaviour.iloc[trial]['Context']
            for p in range(len(pos_edges)-1):
                vec_labels.loc[len(vec_labels)] = [trial,str(p),trial_cat[0],trial_cat[1],trial_cat,behaviour.iloc[trial]['Correct'],behaviour.iloc[trial]['error_type']]  
        #Here Region refers to coarse-grained positions
        if  coarse_pos_edges!=None:
            vec_labels['Region'] = np.digitize(vec_labels['Position'].to_numpy().astype(int), coarse_pos_edges).astype(str)
        if exclude_untracked:
            vec_labels = vec_labels.drop(nan_row)
        return pop_vec_flat, vec_labels
    else:
        return pop_vec_flat

def load_pvs(a,d, data_source,expinfo,pos_binsize,glm_path=None):
    
    import pickle
    
    animal = expinfo['animals'][a]
    date = expinfo['dates'][a][d]
    ntrials=expinfo['ntrials'][a][d]
    nposbins=int(expinfo['max_pos']/pos_binsize)
    
    
    if data_source == 'original':
        with open(expinfo['pop_path'] + animal + '_' + date + '_population.pkl', 'rb') as file:
            all_cells = pickle.load(file)        
        npyr = np.sum(all_cells.cell_types=='p1')
        ratemaps = np.zeros((all_cells.nclusters,ntrials,nposbins))
        for cell in range(all_cells.nclusters):
              ratemaps[cell,:,:]=all_cells.unit[cell].ratemap_pertrial[:ntrials,:]
        ratemaps = ratemaps[all_cells.cell_types=='p1',:,:]
        ratemaps[np.isnan(ratemaps)]=0
        #since the PVs are calculated per position/trial they are the same as the ratemap per trial
        pop_vecs = ratemaps 

    elif data_source == 'GLM':            
        pop_vecs=np.zeros((npyr,ntrials,nposbins))     
        params = np.load(glm_path+f'{animal}_{date}_params.npy')            
        #add all position parameters and the constant parameter, but not those for the speed bins
        #the parameters from the GLM are log firing rates, so take the exponential to get values in Hz
        ratemaps = [params[i,:ntrials*nposbins].reshape(ntrials,nposbins)+params[i,-1] for i in range(npyr)]
        for cell in range(npyr):
            pop_vecs[cell]=np.exp(ratemaps[cell])
            
    else:
        raise Exception('Data source unknown')
        
    npyr=pop_vecs.shape[0]

    #Flatten the array to 2D, nsamples x nneurons as required by sklearn
    pop_vec_flat = pop_vecs.reshape(npyr,-1).T
    pop_vec_flat[np.isnan(pop_vec_flat)]=0
    
    return pop_vec_flat


def get_vec_labels(behaviour, ntrials, npos, coarse_pos_edges):
    #get labels for population vectors already calculated for ntrials and npos
    #coarsepos_edges are the bin edges for large position bins, in case you want to label nearby positions all the same label
    
    vec_labels = pd.DataFrame(columns=['Trial#','Position','Side','Context','Category','Correct','Error_type']) #label per datapoint, to be used for decoding
    for trial in range(ntrials):
        trial_cat = behaviour.iloc[trial]['Start']+behaviour.iloc[trial]['Context']
        for p in range(npos):
            vec_labels.loc[len(vec_labels)] = [trial,str(p),trial_cat[0],trial_cat[1],trial_cat,behaviour.iloc[trial]['Correct'],behaviour.iloc[trial]['error_type']]     
    #Here Region refers to coarse-grained positions
    vec_labels['Region'] = np.digitize(vec_labels['Position'].to_numpy().astype(int), coarse_pos_edges).astype(str)
    
    return vec_labels