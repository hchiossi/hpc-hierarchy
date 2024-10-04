#!/usr/bin/env python
# coding: utf-8

"""
Code from paper Chiossi et al, 2024
Source: https://github.com/hchiossi/hpc-hierarchy
Fits GLM to single neuron responses
Parameters can be used to generate a "ratemap" with speed regressed out
"""

import json
import pickle
from func_behaviour import load_behaviour
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts


#%% Functions


# log likelihood Poisson GLM
def loglglm(X,Y,an):
    return np.mean(-np.exp(an@X) + Y*(an@X) - logfac[Y])
# log likelihood (without taking mean)
def logllk(X,Y,an):
    return -np.exp(an@X) + Y*(an@X) - logfac[Y]
# log factorial
logfac = np.array([0,0]+[np.sum(np.log(np.arange(2,n))) for n in range(3,10000)])

# fit GLM to single cell responses - do cross validation to find best lambda
# this was used before defining a lambda for all data
def cross_valid_sing_cell(X,Y):
    # possible L2 penalties
    pl = [0.0001,0.001,0.01,0.1]
    llk_all = []
    for i in range(len(Y)):
        print(f'cell {i}')
        llk = []
        for l in pl:
            print(f'penalty {l}')
            llk.append([])
            for jk in range(4): # 4-fold cross validation for each cell
                print(f'rep {jk}')
                X_train,  X_test, Y_train, Y_test = tts(X.T,Y[i].T,test_size=0.1,shuffle=True)
                g = sm.GLM(Y_train, X_train.astype(float), family=sm.families.Poisson())
                glm=g.fit_regularized(L1_wt=0, alpha=l)
                llk[-1].append(loglglm(X_test.T,Y_test,glm.params))
        llk_all.append(llk)
    lams = [pl[np.argmax(np.mean(llk_all[i],1))] for i in range(len(Y))]
    return lams, llk_all

# fit all cells after knowing best lambda for each cell
def fit_all(X,Y,lams):
    params = []
    N,T = Y.shape
    for i in range(N):
        g = sm.GLM(Y[i], X.astype(float), family=sm.families.Poisson())
        glm=g.fit_regularized(L1_wt=0, alpha=lams[i])
        params.append(glm.params)
    return np.array(params)

#%% User-defined variables
#File containing all the experimental information, see gen_experiment_info_dict.
info_file ='/example/hpc-hierarchy/fullexp_info.json'

#Path where you saved the files from GLM_data_prep.py
data_prep_outpath = '/example/ratemapsGLM/'

# animal and date to analyse (as a number within [0,#animals[ and  [0,#dates[ )
#we recomend running this for one session at a time, as it is a heavy computation. You can also parallelise for multiple sessions using a bash script
a,d = 0,0

#should be the same as in the data_prep
pos_binsize=4

#should be the same as in the data_prep
n_spdbins= 10

#%% Load experimental variables
with open(info_file, 'r') as openfile: 
   exp_info = json.load(openfile)

animal = exp_info['animals'][a]
date=exp_info['dates'][a][d]
pos_edges = exp_info['position_edges'][a]
session_path = data_prep_outpath + f'pos{pos_binsize}_spd{n_spdbins}perc/{animal}/{date}/' 

behaviour = load_behaviour(exp_info['bhv_path'], [animal], [[date]])
behaviour= behaviour.iloc[:40]

pop_path = '/helo3/analysis/clu_measures/'
with open(pop_path + animal + '_' + date + '_population.pkl', 'rb') as file:
      all_cells = pickle.load(file)

npyr = np.sum(all_cells.cell_types=='p1')

# In[52]:

# load stuff for DIR and CTX
x=np.loadtxt(session_path+'positions.all').astype(int)
y=np.loadtxt(session_path+'spikes.all').astype(int)
t=np.loadtxt(session_path+'trials.all').astype(int)
s=np.loadtxt(session_path+'speeds.all').astype(int)
    
    
# In[53]:


# number of cells, of bins, etc
ntime = y.shape[1] 
ncells = y.shape[0]
ntrials = np.max(t)
nspeedbin = np.max(s)
npositionbin = np.max(x)


# In[54]:


# create design matrix
# the regressors are position, speed, trial number
# we consider one-hot encoding for position X trial and speed
# so we have a binary vector long npositionbin * ntrials + nspeedbin
# one for each time bin of course

design = np.zeros((ntime, npositionbin*ntrials+nspeedbin))
for i in range(ntime):
    # position X trial
    design[i, x[i]-1 + npositionbin*(t[i]-1)] = 1
    # speed
    design[i, npositionbin*ntrials+s[i]-1] = 1


#%% If you want to determine the optimal penalty lambda for each cell independently using cross-validation
#lams, llk_all = cross_valid_sing_cell(design.T, y[selected_cells,:])
   
#otherwise set a fixed value for all cells
lams = [0.00001]*ncells

#%%fit GLM to single cell responses using the optimal penalty from above
params={}
params = fit_all(design, y.T, lams)
np.save(session_path+f'{animal}_{date}_params.npy',params)

#%%for reconstructing the ratemaps to use in other scripts, you can run the following
#this operation will create maps with log firing rates

# create speed-free maps by taking the first ntrials*npositionbin parameters and reshaping them
#maps = [params[i,:ntrials*npositionbin].reshape(ntrials,npositionbin) for i in range(len(selected_cells))]

# create map with a fixed speed
#chosen_bin=0  #zero for adding the first speed bin 
#maps = [params[i,:ntrials*npositionbin].reshape(ntrials,npositionbin)\
#                   +params[i,ntrials*npositionbin+chosen_bin] for i in range(ncells)]
