#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:59:55 2024

@author: helo
"""

#This script creates a dictionary with all experimental variable and paths to files to be used by other scripts

import numpy as np
import json
default_vars = dir().copy()

#%% Change here the variables to the experiment

#where all processed data is. The folder structure is path/animalID/animalID-date/
path = '/example/processing/'

#where behavior data is. Use .csv files, as in the example provided.
bhv_path = '/example/behaviour/'

#where the population objects are saved. Check the script example_clusterpopulation.py for more information
pop_path = '/example/analysis/clu_measures/'

 # in Hz, rate of original neural recording. It should match the shifts file
rec_rate = 24000

# in Hz, final rate of neural data after downsampling
res_rate = 20000 

# in Hz, rate of sampling of animal position in the maze
whl_rate = 50

#a list of animal IDs as strings, as it is on the file names
animals = ['jc233', 'jc243','jc250','jc253','jc259']

 # a list of lists, each containing the recording dates for each animal as strings, as it is on the file names
dates = [["170220","180220","190220","200220","210220"],\
          ["240820","260820","270820","280820","310820"],\
          ["010221","020221","030221", "050221","060221"],\
          ["160321","170321","180321","190321", "210321"],\
          ["210421","220421","230421","240421", "260421"]]

ndays_peranimal = 5 #this assumes that you are analysing the same amount of days for every animal

#list of lists, number of trials to use for analysis, in this case the learning but not the probe trials
ntrials_all = [[40]*ndays_peranimal for _ in range(len(animals))] #you can in principle define different number of trials for each day

#one list for each animal, containing the list of number of recorded files (sleep+trials) on each day
nsessions_all = [[56]+[46]*4,[46]*5,[46]*5,[46]*5,[46]*5] #session include trials and sleep, you can define a value for each day

#maximum maze position, or its length, in cm
max_pos = 360

# In case you are using coarse-grained linearized maze positions, you might want to define the edges of the bins per animal
# to make sure a reward location does not lie in between edges. Keep all the same size and any shift should be done by 
# altering the size of the edge bins only
position_edges = [[0,40,80,120,160,200,240,280,320,360],\
                  [0,45,90,130,170,210,250,290,325,360],\
                  [0,35,70,110,150,190,230,270,315,360],\
                  [0,40,80,120,160,195,235,275,320,360],\
                  [0,40,80,120,160,200,240,280,320,360]]    

#%% Create dictionary

info={}
default_vars = default_vars+['default_vars']+['info']

for v in dir():
    if v not in default_vars and v.startswith("__") == 0 :
        info[v]=eval(v)
        
export_file= '/example/hpc-hierarchy/fullexp_info.json'
with open(export_file, "w") as outfile:
    json.dump(info, outfile, indent=1)



