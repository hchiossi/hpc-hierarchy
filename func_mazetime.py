#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 10:07:38 2021

@author: hchiossi
"""

import numpy as np

def lin_inmaze_time(lin_whl, filestart, fileend):
    #This function return the time the animal enters and exits the maze.
    #function for linearized tracking positions
    #file start and end are consecutive values from the session_shifts file, converted to the whl rate

    #a number of continuous frames where if there is no tracking should be ignored
    spurious_range = 200 
    
    whl_trial = lin_whl[filestart:fileend]
    true_track = np.array([np.mean(whl_trial[i:i+spurious_range]>0) for i in range(np.size(whl_trial)-spurious_range)])
    start= filestart + np.argmax(true_track==1)
    
    fromstart = lin_whl[start:fileend] #starting from real tracking
    true_track = np.array([np.mean(fromstart[i:i+spurious_range]>0) for i in range(np.size(fromstart)-spurious_range)])
    last_pos = np.argmax(true_track==0)
    end = start + last_pos
    
    return [start, end] #in frames

def reward_time(lin_whl, inmaze_time, reward_pos, radius=10):
    #This function returns time spent within a radius of an specificied position
    times = []
    for r in range(np.size(reward_pos)):        
        t = np.sum(np.logical_and(lin_whl[inmaze_time[0]:inmaze_time[1]]>=(reward_pos[r]-radius), lin_whl[inmaze_time[0]:inmaze_time[1]]<=(reward_pos[r]+radius)))
        if t == 0:
            print(inmaze_time)
        times.append(t)
    return times #in number of frames, for each reward

def first_refrain(whl, behaviour, allshifts, reward_pos, ntrials=40, radius=10, output='duration'):
    #The function returns the time in seconds spent during the first passage of the unrewarded hole (refrain duration)
    #if it came before the contextual reward, otherwise it return NaN for that trial
    #please give the linear whl as input!
    
    def passage_time(on_hole):        
        start = np.argmax(on_hole)
        return np.argmax(np.invert(on_hole[start:])), start #in frames
    
        #get sequence of the rewards, from right to left
    if reward_pos[1]<reward_pos[2]:
        order = [['A', 'B'],[1,2]]
    else:
        order = [['B', 'A'],[2,1]]   
        
    refrain_duration = np.zeros(ntrials)
    should_refrain = np.zeros(ntrials)
    did_refrain = np.zeros(ntrials)
    start_end_frame = np.zeros((ntrials,2))
    
    for session in range(2,42):
        shift = (allshifts[session-2:session]/480).astype(int)
        trial_whl = whl[shift[0]:shift[1]]
        #define the situations where the animal had to refrain for the if statements
        if behaviour['Start'][session-2]=='R' and behaviour['Context'][session-2]==order[0][1]:
            should_refrain[session-2]=1
            if behaviour['Correct'][session-2]==1:
                did_refrain[session-2]=1
                on_hole = np.logical_and(trial_whl>reward_pos[order[1][0]]-radius,trial_whl<reward_pos[order[1][0]]+radius)
                while refrain_duration[session-2]<13: #in less than 13 frames speed would have been >80cm/s which is not possible
                    refrain_duration[session-2], start = passage_time(on_hole)
                    on_hole = on_hole[start+1:]
            else:
                refrain_duration[session-2]=None
        elif behaviour['Start'][session-2]=='L' and behaviour['Context'][session-2]==order[0][0]:
            should_refrain[session-2]=1
            if behaviour['Correct'][session-2]==1:
                did_refrain[session-2]=1
                on_hole = np.logical_and(trial_whl>reward_pos[order[1][1]]-radius,trial_whl<reward_pos[order[1][1]]+radius)
                while refrain_duration[session-2]<13:
                    refrain_duration[session-2], start = passage_time(on_hole)
                    on_hole = on_hole[start+1:]
            else:
                refrain_duration[session-2]=None
        # elif already got the contextual one but needs to refrain the other one before getting to U
        else:
            refrain_duration[session-2]=None
            
    if output=='duration':
        return refrain_duration, did_refrain, should_refrain #in frames. if you want in second divide by 50
    elif output=='start_end':
        return start_end_frame
    else:
        print('Please define a valid output type')
        return


def dig_start(lin_whl, speed, whl_shifts, reward_coord, behaviour, trial_nums, radius=12):
    #This function returns the estimated time where animal started to dig a reward hole for the first time in the trial.
    #This function will only give a value if the animal actually dug that reward in the trial. Otherwise it will return NaN.
    
    first_dig = np.zeros((len(trial_nums),3)) 
    for trial in trial_nums:
        filestart = whl_shifts[trial]
        fileend = whl_shifts[trial+1]
        
        #get time where he gets to rewarded holes
        rew_order=['U','A','B'] #as in the order of columns in the behaviour file
        for rew in range(3):
            
            if behaviour.iloc[trial][rew_order[rew]]==0: #did not dig this position in this trial
                first_dig[trial,rew]=np.nan
            
            else:
                time_on_rew = np.logical_and(lin_whl>(reward_coord[rew]-radius), lin_whl<(reward_coord[rew]+radius)) #times animal was around reward
                time_on_rew = np.logical_and(time_on_rew, speed<6) #filter for periods of low mobility, not just running past
                time_on_rew[:filestart]=False
                time_on_rew[fileend:]=False
                
                if np.sum(time_on_rew)==0: #no time on reward at low speed
                    first_dig[trial,rew]=np.nan
                
                else:
                    #if I want to make sure to take only the time the animal was at the reward for more than 0.5sec
                    rew_start = np.where(np.diff(time_on_rew)>0)[0] #get entry/exit times of reward zone
                    
                    for i in range(len(rew_start)):
                        mean_pos = np.mean(lin_whl[rew_start[i]:rew_start[i]+100]) #if on average was inside the reward zone for the next 1.5 second
                        at_rew = np.logical_and(mean_pos>(reward_coord[rew]-radius), mean_pos<(reward_coord[rew]+radius))
                        mean_speed = np.mean(speed[rew_start[i]:rew_start[i]+100]) #maybe remove this!!!!
                        if at_rew == True and mean_speed<10:
                            break
                    if i==(len(rew_start)-1) and at_rew == False:
                        first_dig[trial,rew]=np.nan
                    else:
                        true_entry = rew_start[i]
                        first_dig[trial,rew]=true_entry
                       
    return first_dig #order is U, A, B
    
    
def first_passage(lin_whl, speed, whl_shifts, coord, ntrials=40,radius=5):
    
    first_pass = np.zeros((ntrials,len(coord)))
    
    for trial in range(ntrials):
        filestart = whl_shifts[trial]
        fileend = whl_shifts[trial+1]
        sub_whl = lin_whl[filestart:fileend]
        for idx,c in enumerate(coord):
            first_pass[trial,idx]=filestart+np.argmax(np.logical_and(sub_whl>(c-radius), sub_whl<(c+radius)))    
    
    return first_pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    