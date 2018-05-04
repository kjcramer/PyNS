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

data=np.load('ws_temp_Membrane_R_3_1_4.npz')

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
xn = data['arr_18']
yn_air = data['arr_19']
yn_h2o = data['arr_20']
yn_fil = data['arr_21']
zn = data['arr_22']
uf_air = data['arr_23']
vf_air = data['arr_24']
wf_air = data['arr_25']
p_air = data['arr_26']

AIR = 0
H2O = 1
FIL = 2

#xn = (nodes(-0.05,  0.05, 150), nodes(-0.05, 0.05, 150), nodes(-0.05,   0.05, 150))
#yn = (nodes(-0.004, 0,     10), nodes( -0.02,    -0.004,  30), nodes(0, 0.005,  15))
#zn = (nodes( 0.0,   0.0006, 2), nodes( 0.0,  0.0006, 2), nodes( 0.0,    0.0006, 2))

z_pos = 1
    
xc = avg(xn[AIR])
yc = np.append(avg(yn_h2o), avg(yn_air),axis=0)
yc = np.append(yc, avg(yn_fil),axis=0)
zc = avg(zn[AIR])
np.mean(t_h2o[-1:,:,:])

#%% vertical temperature profil

t_plot_s=np.append(t_h2o[0,:,z_pos],t_air[0,:,z_pos])
t_plot_s=np.append(t_plot_s, t_fil[0,:,z_pos])

t_plot_m=np.append(t_h2o[75,:,z_pos],t_air[75,:,z_pos])
t_plot_m=np.append(t_plot_m, t_fil[75,:,z_pos])

t_plot_e=np.append(t_h2o[149,:,z_pos],t_air[149,:,z_pos])
t_plot_e=np.append(t_plot_e, t_fil[149,:,z_pos])

plt.figure
plt.subplot(1,3,1)
plt.plot(t_plot_s,yc, linestyle='-', color='blue', linewidth=1.2)
plt.plot([18, 80],[0, 0], linestyle='--', color='black', linewidth=1)
plt.plot([18, 80],[-0.004, -0.004], linestyle='--', color='black', linewidth=1)
plt.xlim([18, 80])
plt.ylabel('Y [m]',fontsize=20)
plt.xticks([20,40,60,80],fontsize=18)
plt.yticks([-0.005, 0, 0.01, 0.02],fontsize=18)

plt.subplot(1,3,2)
plt.plot(t_plot_m,yc, linestyle='-', color='blue', linewidth=1.2)
plt.plot([18, 80],[0, 0], linestyle='--', color='black', linewidth=1)
plt.plot([18, 80],[-0.004, -0.004], linestyle='--', color='black', linewidth=1)
plt.xlim([18, 80])
plt.xticks([20,40,60,80],fontsize=18)
plt.yticks([])
plt.xlabel('Temperature [°C]',fontsize=20)

plt.subplot(1,3,3)

plt.plot(t_plot_e,yc, linestyle='-', color='blue', linewidth=1.2)
plt.plot([18, 80],[0, 0], linestyle='--', color='black', linewidth=1)
plt.plot([18, 80],[-0.004, -0.004], linestyle='--', color='black', linewidth=1)
plt.xlim([18, 80])
plt.xticks([20,40,60,80],fontsize=18)
plt.yticks([])

pylab.show
np.mean(t_h2o[-1:,:,:])

#%% axial membrane flux

plt.figure
plt.plot(xc+0.05,m_j[:,0,1]/6.666E-4/6.666E-4*3600, linestyle='-', color='blue', linewidth=1.2)
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
plt.contourf(zc,avg(yn_h2o),u_h2o_plot[0,:,:])
plt.colorbar()
plt.xlabel('Z [m]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('Y [m]',fontsize=20)
plt.title('Inlet Velocity Profile [m/s]',fontsize=20)
pylab.show
np.mean(t_h2o[-1:,:,:])

#%% velocity contour plot

xc = avg(xn[AIR,1:])
yc = avg(yn_air[1:])

u_plot_3=np.sqrt(uf_air[:,1:,1:]*uf_air[:,1:,1:]+vf_air[1:,:,1:]*vf_air[1:,:,1:]+wf_air[1:,1:,:]*wf_air[1:,1:,:])
u_plot=u_plot_3[:,:,20]
u_plot=transpose(u_plot)

plt.figure
levels_t=linspace( u_plot.min(), u_plot.max(), 11)
norm_t=cm.colors.Normalize( vmax=u_plot.max(), vmin=u_plot.min() )
cax_u=plt.contourf(xc,yc,u_plot,levels_t,cmap="rainbow",norm=norm_t)
cbar_u=plt.colorbar(cax_u)
plt.title("Velocity [m/s]")
plt.xlabel("x [m]")
#plt.ylim([-1E1,1E1])
plt.ylabel("y [m]" )
np.mean(t_h2o[-1:,:,:])