#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code from paper Chiossi et al, 2024
Source: https://github.com/hchiossi/hpc-hierarchy
This script calculates the evidence that an animal used different behaviour strategies at each trial.
It then calculates a behaviour score by summing the trials of unambiguous use of the correct strategy with
an estimate of the use of the same strategy when strategy is anbiguous.
"""

import func_behaviour as bhv
import func_basics as basics
import numpy as np
import pandas as pd
import json

#%%
#File containing all the experimental information, see gen_experiment_info_dict.py
info_file ='/example/hpc-hierarchy/fullexp_info.json'

with open(info_file, 'r') as openfile: 
   exp_info = json.load(openfile)

animals, dates = exp_info['animals'], exp_info['dates']

#expected score for eager strategy if the animal had a perfect contextual score, given the reward order on the maze
e_exp = [0.5,0.25,0.25,0.25,0.25] #length = n animals

#%%
"""
Strategies:
    
0. Eager
1. Contextual

"""

ntrials = np.max(exp_info['ntrials_all']) #the same was used for all the animals
holes = ['U','A','B'] #U=universal, context-independent reward, A,B=context-dependent rewards of contexts A and B
nstrat = 2 #number of stategies
possible_strat = np.zeros((len(animals),len(dates[0]),nstrat))
possible_strat_pertrial = np.zeros((len(animals),len(dates[0]),nstrat, ntrials))

for a,animal in enumerate(animals):
    
    _, reward_coord = basics.load_coord(exp_info['path'], animal)
    hole_order = [holes[i] for i in np.argsort(reward_coord[0:3])]
        
    for d,date in enumerate(dates[a]):   

        behaviour = bhv.load_behaviour(exp_info['bhv_path'], [animal], [[date]])
        behaviour = behaviour.drop(behaviour[behaviour['Type'] == 'Probe'].index)

        strategies = pd.DataFrame()
        
        #eager - whatever comes first, given the current start side
        strategies['side_strat'] = np.where(((behaviour['Start']=='L') & (behaviour[hole_order[-1]]==1) & (behaviour[hole_order[-2]]==1) )|\
                                           ((behaviour['Start']=='R') & (behaviour[hole_order[0]]==1) & (behaviour[hole_order[1]]==1)), 1, 0)
                    
        #contextual strategy
        strategies['ctx_strat'] = np.where(behaviour['Correct']==1,1,0)

        for s, strat in enumerate(strategies.keys()):
            possible_strat[a,d,s] = strategies[strat].sum()/len(behaviour)
            possible_strat_pertrial[a,d,s,:] = strategies[strat][:ntrials]            

    
#%% Separate ambiguous and unambiguous trials and calculate the score

eager_sure = np.zeros(25)
ctx_sure = np.zeros(25)
ambig = np.zeros(25)
for date in range(25):
    eager_sure[date]=np.mean(np.logical_and(possible_strat_pertrial[date,0,:]==1,possible_strat_pertrial[date,1,:]==0))
    ctx_sure[date]=np.mean(np.logical_and(possible_strat_pertrial[date,0,:]==0,possible_strat_pertrial[date,1,:]==1))
    ambig[date]=np.mean(np.logical_and(possible_strat_pertrial[date,0,:]==1,possible_strat_pertrial[date,1,:]==1))

nanimals = len(animals)
ndates=len(dates[a])

corrected_score = np.zeros((nanimals,ndates))
for a in range(nanimals):
    for d in range(ndates):
        date_id = a*5+d
        c_score = possible_strat[a,d,2] #contextual score
        rescaled_e = eager_sure[date_id]/ (1-e_exp[a])
        n_unamb = 1-e_exp[a]
        corrected_score[a,d] = ctx_sure[date_id] + ambig[date_id]*(1-(eager_sure[date_id]/n_unamb))


np.savetxt('/example/bhvstrategy_score.txt', corrected_score.flatten())
