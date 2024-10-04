## basics: load clu and res, compute speed and heading, compute rate maps
## you can use this as a library or copy/paste the scripts you need
## notice that they are calculated for whls at 50Hz and 4cm spatial binning

import numpy as np
import scipy.ndimage as nd

### load clu and res - this is NOT suited for reading e.g. whl files, for that use np.loadtxt!
def read_integers(filename): # faster way of reading clu / res
    with open(filename) as f:
        return np.array([int(x) for x in f])

def load_files(animal, date, path, nsessions):
    
    if date == '050221': #issue specific to my dataset, can be removed otherwise
        start = '-02'
    else:
        start = '-01'

    path = path + animal + '/' + date + '/'
    basename= animal + '-'+ date
    clu = read_integers(path+basename+start+nsessions+'.clu')
    res = read_integers(path+basename + start+nsessions+'.res')
    whl = np.loadtxt(path+basename+'.whl')
    des = np.genfromtxt(path+basename+start+nsessions+'.des',dtype='str')
    allshifts = np.loadtxt(path+'session_shifts.txt')
    
    return clu, res, allshifts, whl, des

def load_coord(path,animal):
    maze = np.loadtxt(path+ animal + '/'+animal+'-maze.coord', delimiter=' ', skiprows=1)
    rewards = np.loadtxt(path+ animal + '/'+animal+'-reward.coord', delimiter=' ', skiprows=1)
    return maze, rewards

def lin_speed(whl,whl_rate=50):
    temp = np.zeros(whl.shape[0])
    speed = np.zeros(whl.shape[0])
    for i in range(5,whl.shape[0]-5):
        if whl[i] > 0 and whl[i+1] > 0:
            temp[i] = np.sqrt((whl[i] - whl[i+1])**2)
        else:
            temp[i]=-1
    # smooth using a moving average
    width=5
    for i in range(width,whl.shape[0]-width):
        t = temp[i-width:i+width]
        t = t[t>=0]
        if len(t)>0 and whl[i]>0:
            speed[i] = np.mean(t)
        else:
         	speed[i]=np.nan
    return speed*whl_rate

def lin_occ(whl, speed, spf, mazelen=360,whl_rate=50,posbin_size=4):
    maxbin=int(mazelen/posbin_size)
    whl = np.floor(whl[speed > spf] / posbin_size).astype(int)  # speed filter
    occ = np.zeros(maxbin)
    for i in range(whl.shape[0]):
        if(-1 < whl[i]<maxbin):
            occ[whl[i]] += 1
    return occ/ whl_rate

def lin_occ_chunk(whl_chunk, speed, spf, mazelen=360, whl_rate=50, posbin_size=4):
    maxbin=int(mazelen/posbin_size)
    whl = np.floor(whl_chunk[speed > spf] / posbin_size).astype(int)  # speed filter, divide by 4 due to 4cm bins
    occ = np.zeros(maxbin)
    for i in range(whl.shape[0]):
        if(-1 < whl[i]<maxbin):
            occ[whl[i]] += 1
    return occ/ whl_rate 

def lin_rate(occ, # occupancy map
			   whl, # positions x,y
			   spkl, # list of times, aligned with whl, when cell spiked
			   speed, # speed
			   spf, # speed filter
			   sigma_gauss=0, # sigma for smoothing
               posbin_size=4): #in cm 

    # sigma gauss is the amount of gaussian smoothing to apply - usually between 1 and 3
	# if you use this method then compute the occupancy with occmap_gauss ;)
    maxbin = (np.ceil(np.max(whl))/posbin_size).astype(int)
    rate = np.zeros(occ.shape) # spatial bins of 3cm?
    spkl = spkl[speed[spkl]>spf] # speed filter
    spkp = np.floor(whl[spkl] /posbin_size).astype(int) # positions where the spikes "happened"
    for i in range(spkp.shape[0]):
        if(-1<spkp[i]<maxbin):
            rate[spkp[i]] +=1
    # divide by occupancy
    rate[occ>0.05] = rate[occ>0.05] / occ[occ>0.05] # 0.05 means 50 ms occupancy - change this threshold if you want/need!
    # smoothing - watch out, if there are gaps in the behavior this will introduce small biases!
    if sigma_gauss > 0: 
        rate=nd.gaussian_filter(rate,sigma=sigma_gauss)
    
    rate[occ==0]=np.nan # delete places where occupancy is zero
    return rate

def lin_filter_pcell_SI(pop_rates,thres=0.5):
    ncells = pop_rates.shape[0]
    isplace = np.zeros(ncells).astype(bool)
    for cell in range(ncells):
        SI = np.mean(pop_rates[cell,:]*np.log2(pop_rates[cell,:] / np.mean(pop_rates[cell,:])))
        isplace[cell]=(SI>thres)
    return isplace


def lin_mov_direction(whl):
    #input is linear whl
    #-1 for starting left, 1 starting right, 0 for not moving
    temp = np.zeros(whl.shape[0])
    direction = np.zeros(whl.shape[0])
    for i in range(5,whl.shape[0]-5):
        if whl[i] > 0 and whl[i+1] > 0:
            temp[i] = whl[i+1] - whl[i]
        else:
            temp[i]=np.nan
    # smooth using a moving average
    for i in range(5,whl.shape[0]-5):
        t = temp[i-5:i+5]
        t = t[~np.isnan(t)]
        if len(t)>0:
            direction[i] = np.mean(t)
        else:
         	direction[i]=np.nan
    direction[direction<0]=-1
    direction[direction>0]=1    
    return direction

def chunk_vars(lin_whl, speed, shifts, trial_nums, move_dir=False, speed_filter=3):   
    
    from func_mazetime import lin_inmaze_time
    
    start_end = lin_inmaze_time(lin_whl, shifts[trial_nums[0]-2], shifts[trial_nums[0]-1])
    idx = np.r_[start_end[0]:start_end[1]+1]
    for trial in trial_nums[1:]:
        start_end = lin_inmaze_time(lin_whl, shifts[trial-2], shifts[trial-1])
        idx = np.concatenate((idx, np.r_[start_end[0]:start_end[1]+1])) #concatenate remaining trials
    
    if move_dir:
        mov = lin_mov_direction(lin_whl)   
        if move_dir=='L': dir_val = -1
        elif move_dir=='R': dir_val =1
        else: print('This direction does not exist')            
        mov_idx = np.where(mov==dir_val)[0]
        filt_idx = np.intersect1d(mov_idx, idx)
    else:
        filt_idx=idx       
   
    #get firing and position data only for that context
    whlcat = lin_whl[filt_idx]
    speedcat = speed[filt_idx]
    occ_cat = lin_occ_chunk(whlcat, speedcat, spf=speed_filter)
    
    return whlcat, speedcat, occ_cat, filt_idx

class RecordingSession():  
    def __init__(self,a, d, params):           
              
        path = params['path']
        rec_rate = params['rec_rate']
        whl_rate = params['whl_rate']
            
        self.animal = params['animals'][a]
        self.date = params['dates'][a][d]
        self.nsessions = params['nsessions_all'][a][d]
        self.ntrials = params['ntrials_all'][a][d]
        
        basename = self.animal + '-' + self.date #base name for all files of that specific recording day          
                
        self.maze_coord, self.reward_coord = load_coord(path, self.animal)                
        self.lwhl = np.loadtxt(f'{path}{self.animal}/{self.date}/{basename}.lwhl') #linearized position at every frame
        self.linspeed = np.loadtxt(f'{path}{self.animal}/{self.date}/{basename}.spd') #instataneous speed calculated for every frame
        self.movdir = lin_mov_direction(self.lwhl) #linerized movement direction, either -1 or 1 for each frame
        
        #all recorded files were merged before analysis. This file says at which frame it file(1 trial or sleep session) ended, cumulatively
        self.allshifts = np.loadtxt(f'{path}{self.animal}/{self.date}/session_shifts.txt')
        self.whl_shifts = (self.allshifts/(rec_rate/whl_rate)).astype(int) #convert to the rate of position sampling

