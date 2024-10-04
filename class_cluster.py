#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:42:39 2022

@author: helo
"""


import numpy as np
import func_basics as basics
from func_mazetime import lin_inmaze_time
import scipy.ndimage as nd
import pandas as pd

class PlaceField():
    def __init__(self):
        self.centres = None
        self.edges = None
        self.widths = None
        #self.ids = None

class ClusterUnit():
    """
    A single cluster unit (neuron) and its basic properties. Spike times (in number of frames) are aligned with whl (tracking) rate.
    
    Attributes
    ----------
    spiketimes : np.array
        Time for each spike, aligned with whl.
    meanrate : float
        Mean firing rate in Hz, calculated with the full recording (sleep + all trials).
    ratemaps : dict
        Linear rate map for each user-defined category. The keys of the dictionary are the categories and the values are 1D np.arrays.
    meanrate_percat : dict
        Mean firing rate calculated for the time spend in the maze in that category.The keys of the dictionary are the categories and the values are int.
    spatialinfo : dict
        Skaggs spatial information, in bits/sec.The keys of the dictionary are the categories and the values are float.
    sparsity : dict
        Calculate as < λ_x>**2/ < λ_x**2> where  λ_x is the average firing for each location. The keys of the dictionary are the categories and the values are float.
    stability : dict
        Mean correlation between the rate maps calculated in the two halfs of the trials in that category, repeated cross_val times). \
        The keys of the dictionary are the categories and the values are float.
    """
    
    def __init__(self,spiketimes, totaltime):
        """

        Parameters
        ----------
        spiketimes : ndarray or list
            Aligned with whl.
        totaltime : float or int
            In seconds so that firing rate is calculated in Hz.

        Returns
        -------
        None.

        """
        #firing attributes
        self.spiketimes = spiketimes
        self.meanrate = np.size(spiketimes)/totaltime #in the whole recording, including sleep
        self.ratemaps = {}
        self.meanrate_percat = {}
        self.ratemap_pertrial = []

        #spatial measures
        self.spatialinfo = {}
        self.sparsity = {}
        self.stability = {}
        self.coherence = {} 
    

    def calculate_mean_ratemap(self,lin_whl, speed, shifts, trial_nums, cat, filt_idx, occ_cat, speed_filter=3):
        """
        
        Calculates the rate map (place field) for each category given, using the trial numbers for that category.
        The placefields can be acessed at self.placefields.
        
        Parameters
        ----------
        lin_whl : 1D numpy array
            Linearized whl (tracking) values.
        speed : 1D numpy array
            Instataneous velocity, aligned with the whl, in cm/s as calculated in basics.lin_speed.
        shifts : list
            End frame of each file as in the session_shifts.txt file for each recording day, but aligned with the whl.
            To align it with the whl it is usually required to divide it by 480 (24kHz from the original recording / 50 Hz)
        trial_nums : list
            The trial numbers as the line of the trial in the behaviour spreadsheet.
            The algorithm assumes that the spreadsheet starts at 0 with learning trial 1, whilst the recording files it corresponds to file 02.
        cat : str
            The category under which the ratemap will be stored.

        Returns
        -------
        None.

        """
        
        spkl = np.intersect1d(self.spiketimes, filt_idx)
        self.ratemaps[cat] = basics.lin_rate(occ_cat,lin_whl,spkl,speed,spf=speed_filter,sigma_gauss=2)        
        self.calculate_meanrate_percat(cat, trial_nums, shifts, lin_whl)


    def calculate_meanrate_percat(self, category, trial_nums, shifts, lin_whl, whl_rate=50):
        """
        Updates self.meanrate_percat[category] to nspikes / time in the maze in that category. In Hz.

        Parameters
        ----------
        category : str
            Trial category for which this measure should be calculated.
        trial_nums : list
            The trial numbers as the line of the trial in the behaviour spreadsheet.
            The algorithm assumes that the spreadsheet starts at 0 with learning trial 1, whilst the recording files it corresponds to file 02.
        shifts : list
            End frame of each file as in the session_shifts.txt file for each recording day, but aligned with the whl.
        lin_whl : 1D numpy array
            Linearized whl (tracking) values.
        whl_rate : int, optional
            Rate of the whl tracking file, in Hz. The default is 50.

        Returns
        -------
        None.

        """
        totaltime = 0
        nspikes = 0
        for trial in trial_nums:
            inmaze = lin_inmaze_time(lin_whl, shifts[trial-2], shifts[trial-1])
            totaltime += inmaze[1]-inmaze[0]
            nspikes += np.sum(np.logical_and(self.spiketimes>inmaze[0], self.spiketimes<inmaze[1]))
        
        self.meanrate_percat[category] = nspikes/(totaltime/whl_rate)

    def calculate_placefields(self, peak_thres=2, per_trial=True, per_cat=True):
        
        import func_pfields as pf

        if per_cat:
            # Calculate pfields based on the mean map of the category   
            
            self.pfield_percat = PlaceField()
            self.pfield_percat.centres = {}
            self.pfield_percat.edges = {}
            self.pfield_percat.widths = {}

            for cat in self.ratemaps.keys():

                rate_cat = self.ratemaps[cat].copy()
                rate_cat[np.isnan(rate_cat)]=0
                peaks = pf.dist_peaks(rate_cat, thres=peak_thres) #, mean=self.meanrate)
                max_fields = np.sum(peaks)
                if max_fields>0:
                    self.pfield_percat.centres[cat] = np.where(peaks)[0] #it is calculated with np.diff so will give the index one below the real peak
                    self.pfield_percat.centres[cat][np.logical_and(0<self.pfield_percat.centres[cat],self.pfield_percat.centres[cat]<89)]+=1
                    self.pfield_percat.edges[cat] = pf.pfield_region(rate_cat,self.pfield_percat.centres[cat], thres=1)                            
                else:
                    self.pfield_percat.centres[cat] = np.empty(0)
                    self.pfield_percat.edges[cat] = np.empty(0)
                    self.pfield_percat.widths[cat] = np.empty(0)

        if per_trial:
            #Calculate place fields in each trial
            ntrials = self.ratemap_pertrial.shape[0]
            
            rate_pertrial = self.ratemap_pertrial.copy()
            rate_pertrial[np.isnan(rate_pertrial)]=0
            
            self.pfield_pertrial = PlaceField()            
            self.pfield_pertrial.centres = [[] for _ in range(ntrials)]
            self.pfield_pertrial.edges = [[] for _ in range(ntrials)]
            self.pfield_pertrial.widths = [[] for _ in range(ntrials)]

            #peaks = pf.dist_peaks(rate_pertrial, thres=peak_thres) #using std of entire day, instead of per trial
            for trial in range(ntrials):
                peaks = pf.dist_peaks(rate_pertrial[trial,:], thres=peak_thres) #using std per trial                     
                max_fields = np.sum(peaks)
                #max_fields = np.sum(peaks[trial,:])
                if max_fields>0:
                    self.pfield_pertrial.centres[trial] = np.where(peaks)[0] #it is calculated with np.diff so will give the index one below the real peak
                    self.pfield_pertrial.centres[trial][np.logical_and(0<self.pfield_pertrial.centres[trial], self.pfield_pertrial.centres[trial]<89)]+=1
                    self.pfield_pertrial.edges[trial] = pf.pfield_region(rate_pertrial[trial,:],self.pfield_pertrial.centres[trial], thres=1)                    
                    for f in range(max_fields): #calculate the width of each pfield
                        self.pfield_pertrial.widths[trial].append(self.pfield_pertrial.edges[trial][f][1]-self.pfield_pertrial.edges[trial][f][0])

 
    def calculate_spatialinfo(self, category):
        """
        Updates self.spatialinfo[category] to the value of spatial information of the ratemap in that category, as defined in Skaggs et al, 1993

        Parameters
        ----------
        category : str
            Category of trials for which spatial info should be calculated. Ratemaps for this category must already have been created.

        Returns
        -------
        None.

        """
        #Consider normalizing using Tort 2017 suggestion
        #This assumes uniform occupancy
        notNaN = ~np.isnan(self.ratemaps[category])
        notzero = ~(self.ratemaps[category]==0)
        true_bins = np.logical_and(notNaN, notzero)
        prob = 1/np.size(self.ratemaps[category]) #Considering uniform probability over space
        self.spatialinfo[category]= np.sum(prob*self.ratemaps[category][true_bins]*np.log2(self.ratemaps[category][true_bins]/np.mean(self.ratemaps[category][true_bins])))

    
    def calculate_sparsity(self, category):
        """
        Updates self.sparsity[category] to the value < λ_x>**2/ < λ_x**2> where  λ_x is the average firing for each location.

        Parameters
        ----------
        category : str
            Category of trials for which spatial info should be calculated. Ratemaps for this category must already have been created.

        Returns
        -------
        None.

        """
        notNaN = ~np.isnan(self.ratemaps[category])
        self.sparsity[category] = np.mean(self.ratemaps[category][notNaN])**2/ np.mean(self.ratemaps[category][notNaN]**2)
    

    def calculate_stability(self, lin_whl, speed, shifts, trial_nums, cat, speed_filter=3, move_dir=False, cross_val=5):
        """
        The trials in the input category are divided in two random halves. A ratemap is calculated for each half and the Pearson correlation coefficient
        is calculated between them. This process is repeated cross_val times and the mean is stored in self.stability[cat].
        If the firing rate is too low in a given category, one can get division by zero and therefore a NaN result. In general, this measure does not
        make much sense for low firing cells (less than 0.3Hz or so)

        Parameters
        ----------
        lin_whl : 1D numpy array
            Linearized whl (tracking) values.
        speed : 1D numpy array
            Instataneous velocity, aligned with the whl, in cm/s as calculated in basics.lin_speed.
        shifts : list
            End frame of each file as in the session_shifts.txt file for each recording day, but aligned with the whl.
            To align it with the whl it is usually required to divide it by 480 (24kHz from the original recording / 50 Hz)
        trial_nums : list
            The trial numbers as the line of the trial in the behaviour spreadsheet.
            The algorithm assumes that the spreadsheet starts at 0 with learning trial 1, whilst the recording files it corresponds to file 02.
        cat : str
            The category under which the stability will be calculated.
        speed_filter : int, optional
            Speed filter used for ratemap calculation. The default is 3.
        move_dir : int or bool, optional
            If set to -1 or 1 it will do the calculation in rate maps only using bins where the animal is moving to the left or the right, respectively. 
            If set to false, it will use both directions. The default is False.
        cross_val : int, optional
            Number of times the process is repeated, breaking the data in new halves. The result is the mean of these repetitions. More repetitions
            make it more accurate but also slow. The default is 5.

        Returns
        -------
        None.

        """

        trial_nums = trial_nums-2
        
        similarity = np.zeros(cross_val)
        cat_fields = self.pfield_percat.edges[cat]
        
        if not hasattr(self, "pfield_appearance_relative"):
            self.pfield_appearance_relative = {}
        
        #See if any place fields computed, otherwise there is no stability to be measured
        if len(cat_fields)==0: 
            self.stability[cat] = 0 
            return
        else:
            #check in the trial-by-trial maps the first that has a field matching the mean map in the category
            first_trial_withfield = []
            for edges in cat_fields:                            
                for t, trial in enumerate(trial_nums):
                    if np.logical_and(edges[0]<=self.pfield_pertrial.centres[trial],self.pfield_pertrial.centres[trial]<=edges[1]).any():
                        first_trial_withfield.append(t)
                        break
            #if no match was found, the map in the average was spurious but not above 2std on single trials
            if len(first_trial_withfield)==0:
                self.stability[cat]=0
                return

                
            self.pfield_appearance_relative[cat]=first_trial_withfield #trial relative # in category, when each pfield first appears            
            trials_afterappearance = trial_nums[np.min(first_trial_withfield):] #min to get the first trial the first field appears
            
            #if place field appears too late to be considered stable
            if np.min(first_trial_withfield) > 5: 
                self.stability[cat] = 0 
            
            #actual stability calculation
            # do the correlation between random halves of trials since the place fields appeared
            else: 
                for rep in range(cross_val):
                    halves = [np.random.choice(trials_afterappearance, int(len(trials_afterappearance)/2), replace=False)]
                    halves.append(np.array([num for num in trial_nums if num not in halves[0]]))
                    rate = [None]*2
                    for i, trial_half in enumerate(halves):
                        # _,_ ,occ_cat,filt_idx = basics.chunk_vars(lin_whl,speed,shifts,trial_half, move_dir, speed_filter) 
                        # spkl = np.intersect1d(self.spiketimes, filt_idx)
                        # rate[i] = basics.lin_rate(occ_cat,lin_whl,spkl,speed,spf=speed_filter,sigma_gauss=2)
                        rate[i]=np.mean(self.ratemap_pertrial[trial_half,:],axis=0)
                    notNaN = np.logical_and(~np.isnan(rate[0]), ~np.isnan(rate[1]))
                    similarity[rep] = np.corrcoef(rate[0][notNaN],rate[1][notNaN])[0,1]                    
                self.stability[cat]=np.mean(similarity)
    
    def calculate_coherence(self, category):
        
        # The coherence will always be high on smoothed rate maps...
        def fisher_z_transf(r):
            z = 0.5*np.log((1+r)/(1-r))
            return z

        mean_of_neighbours = [self.ratemaps[category][1]] #first item does not have a left neighbour
        mean_of_neighbours.extend([(self.ratemaps[category][pos-1]+self.ratemaps[category][pos+1])/2 for pos in range(1,len(self.ratemaps[category])-1)])
        mean_of_neighbours.append(self.ratemaps[category][-2]) #last item does not have a right neighbour
        pearson_r = np.ma.corrcoef(np.ma.masked_invalid(self.ratemaps[category]), np.ma.masked_invalid(self.ratemaps[category]))[0,1]
        self.coherence[category] = fisher_z_transf(pearson_r)
    
    

class ClusterPopulation():
    """
    A population of neurons from a given recording day. Each neuron is a ClusterUnit object.
    
    Assumptions: It assumes that cluster 0 contains artifacts and cluster 1 contains the noise in the clu file.
                 Trial files are assumed to start at 01 with sleep and first trial is file 02. This is important in functions that use session_shifts.
                 All spatial calculations assume a linear maze. Maze length and spatial bin size can be changed when generating ratemaps.
                 For 2D ratemaps, check func_basics.
                 Recording rates in Hz (for res, whl and session_shifts) are optional variables that might need to be changed.
                 These assumptions are also valid for the ClusterUnit class.
    
    Saving/Loading: to export the object use pickle.dump and to load, pickle.load
    """
    
    def __init__(self,clu,res, res_rate=20000, whl_rate=50, cell_types=None):
        
        self.nclusters = clu[0]-1
        self.res_rate = res_rate
        self.whl_rate = whl_rate
        self.ratemap_categories = set()
        self.unit = [None]*self.nclusters
        self.cell_types = cell_types
        
        #generate a clusterunit object per cluster, with spike times and mean_frate
        div = int(res_rate / whl_rate)
        for cellid in range(2, self.nclusters+2):
            self.unit[cellid-2]= ClusterUnit((res[clu[1:]==cellid]/div).astype(int), res[-1]/res_rate)

        
    def get_firing_rates(self):
        """
        

        Returns
        -------
        mean_frates : 1D numpy array
            Mean firing rate of each unit in the cluster population object, calculated as nspikes/length of the recording.

        """
        mean_frates = np.zeros(self.nclusters)
        for cellid in range(self.nclusters):
            mean_frates[cellid]=self.unit[cellid].meanrate
        return mean_frates


    def get_attribute_allcells(self,attribute, category):
        """
        Returns the value of the required numerical attribute for each unit in the cluster, in the defined category. For attributes that are more than a
        single value per cluterunit (e.g. rate map is a vector), use the specific get function.

        Parameters
        ----------
        attribute : str
            You can check all possible attributes in the ClusterUnit description.
        category : str
            One of the categories for which the ratemap was generated and the required measure/attribute was already calculated. To check for which categories
            there is a ratemap, look at self.ratemap_categories.

        Raises
        ------
        ValueError
            If ratemap for the requested category has not been generated.

        Returns
        -------
        attribute : 1D numpy array
            Array of length nclusters with the value of the requested attribute for each cluster.

        """
        if category not in self.ratemap_categories:
            raise ValueError("Category not in ratemap_categories. A ratemap has not yet been generated for this category. Use generate_ratemaps first")
        att = np.zeros(self.nclusters)
        for cellid in range(self.nclusters):
            att_allcat = getattr(self.unit[cellid], attribute)
            att[cellid] = att_allcat[category]
        return att
    
    
    def get_ratemap_allcells(self,category,mazelen=360, binsize=4):
        """
    
        Parameters
        ----------
        category : str
            Category for which you would like the rate map. The name should match one of the categories used in generate_ratemaps().
        mazelen : int, optional
            Length of the maze, in centimeters. The default is 360.
        binsize : int, optional
            Size of the spatial bins, in centimeters. This should match the value in basics.lin_rate. The default is 4.

        Raises
        ------
        ValueError
            The place map for the category required has not yet been generated. In this case, use generate_ratemaps first.

        Returns
        -------
        ratemap : 2D numpy array
            A numpy array of shape n units x m spatial bins. Each line is the rate map of that unit in the category requested.

        """
        if category not in self.ratemap_categories:
            raise ValueError("Category not in ratemap_categories. A ratemap has not yet been generated for this category. Use generate_ratemaps first")
        else:
            ratemap = np.zeros((self.nclusters,np.size(self.unit[0].ratemaps[category])))
            for cellid in range(self.nclusters):
                ratemap[cellid,:]=self.unit[cellid].ratemaps[category]
        return ratemap
    
    
    def generate_mean_ratemaps(self,lin_whl, speed, shifts, trial_nums_percat, move_dir=False, speed_filter=3):
        """
        
        Calculates and stores the place field for each unit in each trial category. Not generated at initialization of the population to save memory.
        It also automatically calculates mean firing rate in each category.

        Parameters
        ----------
        lin_whl : 1D numpy array
            Linearized whl (tracking) values.
        speed : 1D numpy array
            Instataneous velocity, aligned with the whl, in cm/s as calculated in basics.lin_speed.
        shifts : list
            End frame of each file as in the session_shifts.txt file for each recording day, but aligned with the whl.
            To align it with the whl it is usually required to divide it by 480 (24kHz from the original recording / 50 Hz)
        trial_nums_percat : dict of the form str:list
            Each entry of the dict is a category name and the corresponding trial numbers. 
            The trial number as the line of the trial in the behaviour spreadsheet.
            The algorithm assumes that the spreadsheet starts at 0 with learning trial 1, whilst the recording files it corresponds to file 02.

        Returns
        -------
        None.

        """

        for cat in trial_nums_percat.keys():
            if move_dir: mov = cat[0]
            else: mov = False
            trial_nums = trial_nums_percat[cat]
            _,_ ,occ_cat,filt_idx = basics.chunk_vars(lin_whl,speed,shifts,trial_nums,mov, speed_filter)
            for cellid in range(self.nclusters):
                print('Generating map ' + cat + ' for clu ' + str(cellid) + '/' + str(self.nclusters))
                self.unit[cellid].calculate_mean_ratemap(lin_whl, speed, shifts, trial_nums, cat, filt_idx, occ_cat, speed_filter)
        self.ratemap_categories.update(trial_nums_percat.keys())

    def generate_trial_ratemaps(self, lin_whl, speed, occ, shifts, speed_filter=3, ntrials=40, move_dir=False, trial_side=None):
        """
         Calculates and stores for each unit the ratemap in each trial. It is stores in self.ratemap_pertrial in the form of
         a matrix ntrials x position bins.

        Parameters
        ----------
        lin_whl : 1D numpy array
            Linearized whl (tracking) values.
        speed : 1D numpy array
            Instataneous velocity, aligned with the whl, in cm/s as calculated in basics.lin_speed.
        shifts : list
            End frame of each file as in the session_shifts.txt file for each recording day, but aligned with the whl.
            To align it with the whl it is usually required to divide it by 480 (24kHz from the original recording / 50 Hz)

        Returns
        -------
        None.

        """
        print('Generating ratemap for each trial')
        for cellid in range(self.nclusters): #first initialize for all cells
            self.unit[cellid].ratemap_pertrial = np.zeros((ntrials,occ.shape[0]))
            
        for trial in range(2,ntrials+2): #so I only calculate occupancy and indexes once for each trial and calculate ratemap for all cells
            #get rate map at each trial
            if move_dir: mov = trial_side[trial-2][0]
            else: mov = False
            _,_ ,occ_cat,filt_idx = basics.chunk_vars(lin_whl,speed, shifts,[trial], move_dir=mov,speed_filter=speed_filter)
            for cellid in range(self.nclusters):
                spkl = np.intersect1d(self.unit[cellid].spiketimes, filt_idx)
                self.unit[cellid].ratemap_pertrial[trial-2,:] = basics.lin_rate(occ_cat,lin_whl,spkl,speed,spf=speed_filter,sigma_gauss=2)
            

    
    def calculate_ratemap_measures(self,lin_whl, speed, occ, shifts, trial_nums_percat, speed_filter=3, move_dir=False, cross_val=5, replace=False):
        """
        For each cell it will calculate sparsity, stability and spatial information in all the trial categories requested. You can check which categories
        are stored in self.ratemap_categories. You can then access any of the measures by using the get_attribute function.

        Parameters
        ----------
        lin_whl : 1D numpy array
            Linearized whl (tracking) values.
        speed : 1D numpy array
            Instataneous velocity, aligned with the whl, in cm/s as calculated in basics.lin_speed.
        shifts : list
            End frame of each file as in the session_shifts.txt file for each recording day, but aligned with the whl.
            To align it with the whl it is usually required to divide it by 480 (24kHz from the original recording / 50 Hz)
        trial_nums_percat : dict of the form str:list
            Each entry of the dict is a category name and the corresponding trial numbers. 
            The trial number as the line of the trial in the behaviour spreadsheet.
            The algorithm assumes that the spreadsheet starts at 0 with learning trial 1, whilst the recording files it corresponds to file 02.
        
        Raises
        ------
        ValueError
            If measure does not match any of the options.

        Returns
        -------
        None.

        """
        for cat in trial_nums_percat.keys():
            if cat not in self.ratemap_categories:
                raise ValueError("Category not in ratemap_categories. A ratemap has not yet been generated for this category. Use generate_ratemaps first")
            print(f'Ratemap measures for category {cat}')
            for cellid in range(self.nclusters):
                if self.cell_types[cellid]=='p1':
                    print(f'Cell {cellid}/{self.nclusters-1}')
                    
                    #These computations take time
                    #only perform calculation if it has not yet been done for this category or you want to replace the previous calculation
                    if cat not in self.unit[cellid].sparsity.keys() or replace==True:
                        self.unit[cellid].calculate_sparsity(cat)
                    if cat not in self.unit[cellid].stability.keys() or replace==True:
                        self.unit[cellid].calculate_stability(lin_whl, speed, shifts, trial_nums_percat[cat], cat, speed_filter=3, move_dir=False, cross_val=5)
                    if cat not in self.unit[cellid].spatialinfo.keys() or replace==True:
                        self.unit[cellid].calculate_spatialinfo(cat)
                    # if cat not in self.unit[cellid].coherence.keys() or replace==True:
                    #     self.unit[cellid].calculate_coherence(cat)

    def generate_pfield_measures(self):
        print('Calculating place fields for pyramidal cells only')
        for cellid in range(self.nclusters):
            if self.cell_types[cellid]=='p1':
                print(f'Cell {cellid}/{self.nclusters-1}')
                self.unit[cellid].calculate_placefields()


    def get_placecells(self, category='any', frate_range = (0.25,6) , max_sparsity=0.5, min_info=0.8, min_stability = 0.8):
        """
        Boolean list of which units are place cells, given a set o criteria: firing rate, sparsity, spatial information and stability. It returns if
        it is a place cell in the desired category.

        Parameters
        ----------
        category : str, optional
            Trial category for which you want to know if cell is a place cell. If 'any' will return True if the cell fullfills the criteria in any of 
            the categories for which the measures were generated. If 'all' it will return only if it fullfills the criteria in all categories. The 
            default is 'any'.
        frate_range : tuple, optional
            Minimum and maximum firing rates (in Hz) to be considered place cell. The default is (0.25,6).
        max_sparsity : float, optional
            Maximum sparsity to be considered a place cell. The default is 0.5.
        min_info : float, optional
            Minimum spatial information of the rate map. The default is 0.8.
        min_stability : float, optional
            Minimum stability of the rate map within each trial category. The deafult is 0.8.

        Returns
        -------
        List of bool
            1D list of length nclusters describing if each unit fullfills the defined criteria.

        """

        isplace = np.zeros(self.nclusters, dtype=bool)
        for cellid in range(self.nclusters):

            ratemap_mean = {}
            for cat in self.ratemap_categories:
                cell_ratemap = self.unit[cellid].ratemaps[cat]
                ratemap_mean[cat]=np.mean(cell_ratemap[~np.isnan(cell_ratemap)])           
            
            cell_measures = pd.DataFrame(data={'Sparsity':self.unit[cellid].sparsity,'Stability':self.unit[cellid].stability, \
                                           'Rate':ratemap_mean, 'Spatial Info':self.unit[cellid].spatialinfo})          
            isplace_percat = ((cell_measures['Sparsity']<max_sparsity) & (cell_measures['Stability']>min_stability) & (cell_measures['Rate']>frate_range[0]) & \
                            (cell_measures['Rate']<frate_range[1]) & (cell_measures['Spatial Info']>min_info))

            if category == 'all':
                isplace[cellid] = isplace_percat.all() #get per category if true in all criteria and then see if all are true
            elif category == 'any':
                isplace[cellid] = isplace_percat.any()
            elif category in self.ratemap_categories:
                isplace[cellid] = isplace_percat.loc[category]
                   
        isplace = np.logical_and(isplace, self.cell_types=='p1')
        
        return isplace #boolean vector 
        
        
        
        
        