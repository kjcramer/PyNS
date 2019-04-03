# -*- coding: utf-8 -*-
"""
Postprocessing script to combine information from multiple "Output" text files
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")

#%% This example reads Output files from three simulations with different 
#   feed velocities 

velo = np.zeros(3)
airg = np.zeros(3)
m = np.zeros(3)
RR = np.zeros(3)
GOR = np.zeros(3)

velocities = ['025', '05', '1']
temp = '80'
ag = '05'
  
for counter, uin in enumerate(velocities):
    
  with open('./Demo_data/Output_N_{}_{}_{}_100000.txt'.format(temp,uin,ag)) as file:  
    data = file.read()
    
  ii = 59      
  
  velo[counter] = float(data[ii:ii+5])
  ii = ii + 6
  airg[counter] = float(data[ii:ii+6])
  ii = ii + 8
  m[counter] = float(data[ii:ii+9])
  ii = ii + 10
  RR[counter] = float(data[ii:ii+10])
  ii = ii + 11
  GOR[counter] = float(data[ii:ii+10]) 
 
# Saving option for further use in pgf plot    
#np.savetxt('N_{}_{}.dat'.format(temp,ag), np.transpose((velo, m, RR, GOR)), fmt='%1.4e',header='u_in m_evap RR GOR', comments='')

