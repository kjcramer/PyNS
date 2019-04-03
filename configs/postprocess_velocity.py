# -*- coding: utf-8 -*-
"""
Script to load velocity .npz file and convert it to paraview's .vtr format
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

# Standard Python modules
from pyns.standard import *

# PyNS modules
from pyns.constants          import *
from pyns.operators          import *
from pyns.discretization     import *

name = 'velocity_N_80_05_05_5ts'
airgap = 0.0005
data = np.load(name + '.npz')
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

#u_col = data['arr_9']
#v_col = data['arr_10']
#w_col = data['arr_11']

AIR = 0
H2O = 1
FIL = 2

#%% streamlines in air gap

z_pos = 25

# adjust number of cells!
## 8mm = 48, 2mm = 12, 0.5mm = 6

y,x = np.mgrid[0:6:1,0:56:1]
uu = np.transpose(u_air[:,:,z_pos])
vv = np.transpose(v_air[:,:,z_pos])
ww = np.transpose(w_air[:,:,z_pos])
U=np.sqrt(uu*uu+vv*vv+ww*ww)

plt.figure()
plt.gca(aspect="equal")
plt.streamplot(x,y,uu,vv, density=1.5,color=U, linewidth=1)
plt.ylabel('Y [m]',fontsize=20)
plt.xlabel('X [m]',fontsize=20)

#%%

plt.figure()
plt.contourf(np.transpose(u_air[:,:,z_pos]))

#%%

from pyevtk.hl import gridToVTK

# Node coordinates for domains
# need to be updated manually!
xn = (nodes(0,   0.07, 56), nodes(0, 0.07, 56), nodes(0,       0.07, 56))
yn = (nodes(-airgap, 0, 6), nodes(-airgap-0.01, -airgap,  48), nodes(0.0, 0.001, 6))
zn = (nodes(0,   0.07, 56), nodes(0, 0.07, 56), nodes(0,       0.07,  56))


gridToVTK("./"+ name, xn[AIR], yn[AIR], zn[AIR], cellData = {"u" : u_air, "v" : v_air, "w" : w_air})