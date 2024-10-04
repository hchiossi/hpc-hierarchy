#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code from paper Chiossi et al, 2024
Source: https://github.com/hchiossi/hpc-hierarchy

Example on how to generate ClusterPopulation objects
"""

import numpy as np
from func_behaviour import load_behaviour, get_trials_percat
import func_basics as basics
import func_tracking as tracking
from class_cluster import ClusterPopulation
import pickle

#%% User input variables
animal = 'jc259'
date = '210421'
nsessions = '46'   
path = '/helo3/processing/'
bhv_path = '/helo3/behaviour/'
export_path = '/helo3/analysis/clu_measures/'
basename= animal + '-'+ date

# Set recording rates in Hz
rec_rate = 24000 #as in the shifts file
res_rate = 20000
whl_rate = 50

#%% Load data

clu, res, allshifts, whl, des = basics.load_files(animal, date, path, nsessions)
maze_coord, reward_coord = basics.load_coord(path, animal)
behaviour = load_behaviour(bhv_path, [animal], [[date]])
#get linear coordinates over time
lin_whl, speed, occ = tracking.lin_vars(whl, maze_coord)

#align shifts with whl
whl_shifts = (allshifts/(rec_rate/whl_rate)).astype(int)


#%% Create cluster object
all_cells = ClusterPopulation(clu, res, res_rate=res_rate, whl_rate=whl_rate)


#%% Define trial categories and get trial numbers
trial_nums_percat = get_trials_percat(behaviour)


#%% Calculate ratemaps
all_cells.generate_ratemaps(lin_whl, speed, whl_shifts, trial_nums_percat)


#%% Calculate spatial measures
#These can be quite slow
all_cells.calculate_ratemap_measures(lin_whl, speed, occ, whl_shifts, trial_nums_percat)

#%% Export object 
with open(export_path + animal + '_' + date + '_population.pkl', 'wb') as file:
    pickle.dump(all_cells, file)

# To load:
#with open(export_path + animal + '_' + date + '_population.pkl', 'rb') as file:
#    all_cells = pickle.load(file)


#%% Retrieve some attribute

stability = {}
for cat in trial_nums_percat.keys():
    stability[cat] = all_cells.get_attribute_allcells('stability', cat)

#%%
si= {}
for cat in trial_nums_percat.keys():
   si[cat] = all_cells.get_attribute_allcells('spatialinfo', cat)