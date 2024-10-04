# hpc-hierarchy
Scripts for the publication:
Chiossi, H.S.C., Nardin, M., Tkacik G., Csicsvari, J., (2024) The effects of learning on the hippocampal representational hierarchy, under review. Preprint available at: https://doi.org/10.1101/2024.08.21.608911 

We apologise that not all the scripts are available yet, they will be published here by the time of its final publication.

Minimum required data files to run these scripts:

.res - spike counts, one line per time bin (ideally at a resolution where it is 0 or 1)
.clu - cluster ID for each spike, one line per time bin + 1st line is the total number of clusters
.des - classification of cluster into different cell types, one line per cluster
.whl - animal position over time
.csv - with behaviour information (see example)
.coord - maze coordinates on the video image and reward positions on the linearised maze

Function sets provided:

basics - basic functions to load files, calculate speed, movement direction, occupancy, etc. called by other functions.
behaviour - loading behaviour files and extract other information
decoder - global and conditional decoders
GLM - Generalized Linear Model, as described in the publication above
mazetime - for calculating behaviour measures, such as filtering ITI, time spent on rewards, etc
tracking - for removing tracking artifacts, linearisation, etc
