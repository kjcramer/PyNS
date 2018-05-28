# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:00:32 2017

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

data=np.load('ws_temp_70.npz')

ts = data['arr_0']
t_air = data['arr_1']
t_h2o = data['arr_2']
t_fil = data['arr_3']
u_h2o = data['arr_4']
v_h2o = data['arr_5']
w_h2o = data['arr_6']
a_h2o = data['arr_7']
a_air = data['arr_8']
p_h2o = data['arr_9']
t_int_mem = data['arr_10']
t_int_film = data['arr_11']
m_out = data['arr_12']
m_j = data['arr_13']
m_pv = data['arr_14']
pv_air = data['arr_15']
pv_n = data['arr_16']
pv_s = data['arr_17']
u_air = data['arr_18']
v_air = data['arr_19']
w_air = data['arr_20']
xn = data['arr_21'] 
yn_air = data['arr_22']
yn_h2o = data['arr_23']
yn_fil = data['arr_24']
yn_col = data['arr_25']
zn = data['arr_26']
t_col = data['arr_27']
uf_col = data['arr_28']
vf_col = data['arr_29']
wf_col = data['arr_30']

AIR = 0
H2O = 1
FIL = 2
COL = 3

#xn = (nodes(-0.05,  0.05, 150), nodes(-0.05, 0.05, 150), nodes(-0.05,   0.05, 150))
#yn = (nodes(-0.004, 0,     10), nodes( 0,    0.02,  30), nodes(-0.005, -0.004,  3))
#zn = (nodes(-0.05,  0.05, 150), nodes(-0.05, 0.05, 150), nodes(-0.05,   0.05, 150))

z_pos = 75
    
xc = avg(xn[AIR])
yc = np.append(avg(yn_fil), avg(yn_air),axis=0)
#yc = np.append(yc, avg(yn_air),axis=0)
yc = np.append(yc, avg(yn_h2o),axis=0)
zc = avg(zn[AIR])

#%% vertical temperature profil

t_plot_s=np.append(t_fil[0,:,z_pos],t_air[0,:,z_pos],axis=1)
t_plot_s=np.append(t_plot_s, t_h2o[0,:,z_pos],axis=1)

t_plot_m=np.append(t_fil[75,:,z_pos],t_air[75,:,z_pos],axis=1)
t_plot_m=np.append(t_plot_m, t_h2o[75,:,z_pos],axis=1)

t_plot_e=np.append(t_fil[149,:,z_pos],t_air[149,:,z_pos],axis=1)
t_plot_e=np.append(t_plot_e, t_h2o[149,:,z_pos],axis=1)

plt.figure
plt.subplot(1,3,1)
plt.plot(t_plot_s,yc, linestyle='-', color='blue', linewidth=1.2)
plt.plot([18, 80],[0, 0], linestyle='--', color='black', linewidth=1)
plt.plot([18, 80],[-0.0035, -0.0035], linestyle='--', color='black', linewidth=1)
plt.plot([18, 80],[-0.004, -0.004], linestyle='--', color='black', linewidth=1)
plt.xlim([18, 80])
plt.ylabel('Y [m]',fontsize=20)
plt.xticks([20,40,60,80],fontsize=18)
plt.yticks([-0.004, -0.002, 0, 0.0015],fontsize=18)

plt.subplot(1,3,2)
plt.plot(t_plot_m,yc, linestyle='-', color='blue', linewidth=1.2)
plt.plot([18, 80],[0, 0], linestyle='--', color='black', linewidth=1)
plt.plot([18, 80],[-0.0035, -0.0035], linestyle='--', color='black', linewidth=1)
plt.plot([18, 80],[-0.004, -0.004], linestyle='--', color='black', linewidth=1)
plt.xlim([18, 80])
plt.xticks([20,40,60,80],fontsize=18)
plt.yticks([])
plt.xlabel('Temperature [°C]',fontsize=20)

plt.subplot(1,3,3)
plt.plot(t_plot_e,yc, linestyle='-', color='blue', linewidth=1.2)
plt.plot([18, 80],[0, 0], linestyle='--', color='black', linewidth=1)
plt.plot([18, 80],[-0.0035, -0.0035], linestyle='--', color='black', linewidth=1)
plt.plot([18, 80],[-0.004, -0.004], linestyle='--', color='black', linewidth=1)
plt.xlim([18, 80])
plt.xticks([20,40,60,80],fontsize=18)
plt.yticks([])

pylab.show

#%% axial membrane flux

plt.figure
plt.plot(xc+0.05,m_j[:,:1,75]/0.001/0.0006667*3600, linestyle='-', color='blue', linewidth=1.2)
plt.xlabel('X [m]',fontsize=20)
plt.ylabel('Membrane Mass Flux [kg/(m^2 h)]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
pylab.show

#%% mem interface contour plot

plt.figure
plt.contourf(xc+0.05,zc,np.transpose(t_int_mem[:,0,:]))
plt.colorbar()
plt.xlabel('X [m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('Z [m]',fontsize=20)
plt.title('Evaporation Interface Temperature [°C]',fontsize=18)
pylab.show

#%% inlet velocity profile

u_h2o_plot=u_h2o
u_h2o[u_h2o<0]=0

plt.figure
plt.contourf(zc,avg(yn[H2O]),u_h2o_plot[0,:,:])
plt.colorbar()
plt.xlabel('Z [m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('Y [m]',fontsize=20)
plt.title('Inlet Velocity Profile [m/s]',fontsize=20)
pylab.show
