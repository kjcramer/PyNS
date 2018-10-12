# -*- coding: utf-8 -*-
"""
Created on Mon Oct 08 11:20:58 2018

@author: kerstin.cramer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import sys
sys.path.append("../..")

# Standard Python modules
from pyns.standard import *

# PyNS modules
from pyns.constants          import *
from pyns.operators          import *
from pyns.discretization     import *

data=np.load('velocity_70_70000ts.npz')
# uc_air,vc_air,wc_air,uc_h2o,vc_h2o,wc_h2o,uc_fil,vc_fil,wc_fil,uc_col,vc_col,wc_col

u_air = data['arr_0']
v_air = data['arr_1'] 
w_air = data['arr_2']

u_h2o = data['arr_3']
v_h2o = data['arr_4']
w_h2o = data['arr_5']

u_fil = data['arr_6']
v_fil = data['arr_7']
w_fil = data['arr_8']

u_col = data['arr_9']
v_col = data['arr_10']
w_col = data['arr_11']

AIR = 0
H2O = 1
FIL = 2
COL = 3

#%% streamlines in air gap

z_pos = 40

y,x = np.mgrid[0:21:1,0:128:1]
#y,x = np.mgrid[0:0.125:0.00390625,0:1.25:0.0048828125]
uu = np.transpose(u_air[:,:,z_pos])
vv = np.transpose(v_air[:,:,z_pos])
ww = np.transpose(w_air[:,:,z_pos])
U=np.sqrt(uu*uu+vv*vv+ww*ww)

plt.figure()
plt.gca(aspect="equal")
#plt.contourf(x,y,np.transpose(obst[:,:,16]),[0.7,1.2],colors=('black'))
plt.streamplot(x,y,uu,vv, density=1.5,color=U, linewidth=1)
#plt.xticks([0,0.25,0.5,0.75,1,1.25],fontsize=18)
#plt.yticks([0,0.06,0.12],fontsize=18)
plt.ylabel('Y [m]',fontsize=20)
plt.xlabel('X [m]',fontsize=20)

#%%

plt.figure()
plt.contourf(np.transpose(u_air[:,:,z_pos]))

#%%

from evtk.hl import gridToVTK

# Node coordinates for both domains
xn = (nodes(0,   0.16, 128), nodes(0, 0.16,  128), nodes(0,       0.16, 128), nodes(0,       0.16, 128))
yn = (nodes(-0.0035, 0, 21), nodes(0, 0.0015,  9), nodes(-0.004, -0.0035, 3), nodes(-0.0055, -0.004, 9))
zn = (nodes(0,   0.1,   80), nodes(0, 0.1,    80), nodes(0,       0.1,   80), nodes(0,       0.1,   80))


gridToVTK("./velocity_air_gap", xn[AIR], yn[AIR], zn[AIR], cellData = {"u" : u_air, "v" : v_air, "w" : w_air})