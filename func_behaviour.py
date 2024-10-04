#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code from paper Chiossi et al, 2024
Source: https://github.com/hchiossi/hpc-hierarchy
Functions to use with .csv behaviour files, for examples check the link above.
"""

import pandas as pd
import numpy as np

def load_behaviour(path, animal_list, dates):
    #Input: animals as a list and dates as a list of lists.
    #Creates a Pandas dataframe with all the data
    rat_list = []
    for i in range(len(animal_list)):
        rat_data = []
        animal = animal_list[i]
        for day in dates[i]:
            datatemp=pd.read_csv(path+animal+'-' + day + "_witherrors.csv")
            datatemp['Date']=day
            datatemp['Animal']=animal
            rat_data.append(datatemp)
        rat_list.append(pd.concat(rat_data))
    fulldata = pd.concat(rat_list)
    #reordering columns
    fulldata = fulldata[['Animal','Date','Context','Start','Correct','Type','U','A','B','Other','holesdug','error_type']]
    return fulldata


def get_trials_nums(behaviour, category='Context', correct=1, balanced=True):
    if category == 'Context':
        labels = ['A','B']
    if category == 'Start':
        labels = ['L','R']
       
    trial_nums = [] #first entry has trial numbers of first label, second from second label, etc 
    #select only correct trials and make sure same ntrials per ctx are used
    for label in labels:
        trial_nums.append(np.array(behaviour.index[(behaviour[category] == label) & (behaviour['Type']=='Learning') & (behaviour['Correct']==1)].tolist()) + 2) #Because first trial is file 2  
   
    if balanced: #will output same number of trials for each category
        ntrials_perctx = np.min([np.size(trial_nums[0]),np.size(trial_nums[1])]) 
        trial_nums = [np.random.choice(trial_nums[0], ntrials_perctx, replace=False),np.random.choice(trial_nums[1], ntrials_perctx, replace=False)]
    
    return trial_nums

def get_trials_percat(behaviour, correct=None):
    #Correct is 1 for only correct trials, 0 for only error trials, None for all trials
    #These are file numbers, for trial number subtract 2!
    contexts = behaviour['Context'].unique()
    starts = behaviour['Start'].unique()  
    trial_nums_percat = {}
    for s in starts:
        for c in contexts:
            cat = s + c
            if correct == None:
                trial_nums_percat[cat] = np.array(behaviour.index[(behaviour['Context'] == c) & (behaviour['Start'] == s)\
                                                              & ((behaviour['Type']=='Learning')|(behaviour['Type']=='Visible')) ].tolist()) + 2 #trial 0 is file 2
            else:
                trial_nums_percat[cat] = np.array(behaviour.index[(behaviour['Context'] == c) & (behaviour['Start'] == s)\
                                                              & ((behaviour['Type']=='Learning')|(behaviour['Type']=='Visible'))\
                                                                  & (behaviour['Correct']==correct)].tolist()) + 2 #trial 0 is file 2
    return trial_nums_percat