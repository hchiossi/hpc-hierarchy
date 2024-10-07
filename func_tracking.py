#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code from paper Chiossi et al, 2024
Source: https://github.com/hchiossi/hpc-hierarchy
Functions related to the linearization of positional tracking data (whl)
"""

import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def LEDtocm(whl, pixelspermeter,clusterdelta):
    #convert from Axona Ltd whl output (6 column) to 2 columns containing x and y position, and converts pixels to cm values
    #pixels per meter can be found in .set file line "tracker_pixels_per_meter", usually 600, and cluster_delta also, usually 6
     
    #take the average of the two LED
    percm = pixelspermeter/100
    new_whl = np.zeros((whl.shape[0], 2))
    new_whl[:,0] = (whl[:,0]+whl[:,2])*clusterdelta/2/percm #xvalues
    new_whl[:,1] = (whl[:,1]+whl[:,3])*clusterdelta/2/percm #yvalues 
    
    #correct for positions were either one or both LEDs were not tracked
    onlysmall = np.logical_and(whl[:,2]==1023,whl[:,5]==1) #will have a 1 if only the big LED is not tracked    
    new_whl[onlysmall,:]=whl[onlysmall,0:2]*clusterdelta/percm #
    
    new_whl[whl[:,5]==0]= -1 #when entry is invalid according to igor's whl file
   
    return new_whl #for some reason jc233 needs this to have correct measures
    
    
def linearize_smaze_with_coord(whl, coord, overlap=10, smooth=True):
    #Broadcasts tracking to 1D
    #Rescaling to get maze of real size of 360 cm (80cm each - 10cm overlap in each corner)

    arm = [] #coordinates of the arms, from maze.coord file    
    for i in range(5):
        arm.append(Polygon([(coord[i][0],coord[i][2]),(coord[i][1],coord[i][2]),(coord[i][1],coord[i][3]),(coord[i][0],coord[i][3])]))
    
    arm_len = [coord[0][4],coord[1][4],coord[2][4],coord[3][4],coord[4][4]]
    
    lin_whl = -np.ones(whl.shape[0]) #casts -1 to every time bin and only will only replace those contained in the maze for real values
    for i in range(whl.shape[0]): #the subtractions are manual corrections so that there is no gap between arms
        point = Point(whl[i,0],whl[i,1])
        if arm[0].contains(point): #horizontal arm
            lin_whl[i]= (whl[i,0] - coord[0][0])*80/arm_len[0] #align to a zero start
        elif arm[1].contains(point): #vertical arm
            lin_whl[i]=(whl[i,1] - coord[1][2])*80/arm_len[1]  + 80 - overlap #add length of the other arms
        elif arm[3].contains(point): #better arm 3 before 2 because of the overlapping region
            lin_whl[i]=(whl[i,1] - coord[3][2])*80/arm_len[3] + 80*3 - overlap*3
        elif arm[2].contains(point):
            lin_whl[i]= ((-whl[i,0] + coord[2][0]))*80/arm_len[2]  + 80*2 - overlap*2 #rotate by 180 degrees to get the correct direction
        elif arm[4].contains(point):
            lin_whl[i]=(whl[i,0] - coord[4][0])*80/arm_len[4] + 80*4 - overlap*4
    
    if smooth: #assumes the animal moved at constant speed between two points if tracking was lost in between them
        idx = np.array(np.where(lin_whl>=0)).flatten() #get index of all frames when tracking was truly captured
 
        for i in range(np.size(idx)-1):
            #currently filling the whole gap with the average, later replace by a gaussian?
            gap = idx[i+1]-idx[i]
            if gap>1 and gap<300: # to fill gap of up to 3s but avoid doing this at the end of the trial
                lin_whl[idx[i]:idx[i+1]+1] = np.linspace(lin_whl[idx[i]],lin_whl[idx[i+1]],gap+1) #i and i+1 will be replaced by themselves
            
        sr = 5 #smoothing range
        for i in range(sr,whl.shape[0]-sr):
            t = lin_whl[i-sr:i+sr]
            t = t[t>=0]
            if len(t)>0 and lin_whl[i]>0:
                lin_whl[i] = np.mean(t)
            else:
            	lin_whl[i]=-1
        
    return lin_whl


def lin_vars (whl, coord):
    import func_basics as basics
    whl_cm = LEDtocm(whl,600,6) #Convert tracking to jozsef's format (2 columns and in cm)
    lin_whl = linearize_smaze_with_coord(whl_cm,coord, smooth=True) #filter artifacts and broadcasts to 1D
    speed = basics.lin_speed(lin_whl)
    speed[speed>100] = np.nan #remove artifact from when tracking jumped around
    occ = basics.lin_occ(lin_whl, speed, spf=3) #5cm bins!
    return lin_whl, speed, occ

