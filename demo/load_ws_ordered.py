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

data=np.load('ws_70_99000.npz')
#(ts, xn, yn[AIR], yn[H2O], yn[FIL], yn[COL], zn, 
# t[AIR].val, uf[AIR].val,vf[AIR].val,wf[AIR].val, p_tot[AIR].val, p[AIR].val, a[AIR].val,  p_v[AIR].val, p_v[AIR].bnd[N].val, p_v[AIR].bnd[S].val, 
# t[H2O].val, uf[H2O].val,vf[H2O].val,wf[H2O].val, p_tot[H2O].val, p[H2O].val, a[H2O].val, 
# t[FIL].val, 
# t[COL].val, uf[COL].val, vf[COL].val, wf[COL].val, p_tot[COL].val, p[COL].val, 
# mem.t_int, mem.j, mem.pv, t_int,m_evap )

ts = data['arr_0']
xn = data['arr_1'] 
yn_air = data['arr_2']
yn_h2o = data['arr_3']
yn_fil = data['arr_4']
yn_col = data['arr_5']
zn = data['arr_6']

t_air = data['arr_7']
u_air = data['arr_8']
v_air = data['arr_9']
w_air = data['arr_10']
p_tot_air = data['arr_11']
p_air = data['arr_12']
a_air = data['arr_13']
pv_air = data['arr_14']
pv_n = data['arr_15']
pv_s = data['arr_16']

t_h2o = data['arr_17']
u_h2o = data['arr_18']
v_h2o = data['arr_19']
w_h2o = data['arr_20']
p_tot_h2o = data['arr_21']
p_h2o = data['arr_22']
a_h2o = data['arr_23']

t_fil = data['arr_24']

t_col = data['arr_25']
u_col = data['arr_26']
v_col = data['arr_27']
w_col = data['arr_28']
p_tot_col = data['arr_29']
p_col = data['arr_30']

t_int_mem = data['arr_31']
m_j = data['arr_32']
m_pv = data['arr_33']
t_int_film = data['arr_34']
m_out = data['arr_35']



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
plt.plot(xc[AIR]+0.05,m_j[:,:1,75]/0.001/0.0006667*3600, linestyle='-', color='blue', linewidth=1.2)
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

#%% contour plots

xc = avg(xn[AIR])
#yc = np.append(avg(yn[FIL]), avg(yn[AIR]),axis=0)
#yc = np.append(yc, avg(yn[H2O]),axis=0)

t_plot=np.append(t_fil[:,:,z_pos],t_air[:,:,z_pos],axis=1)
t_plot=np.append(t_plot, t_h2o[:,:,z_pos],axis=1)
t_plot=transpose(t_plot)
p_air = np.zeros(np.shape(t_air))
p_fil = np.zeros(np.shape(t_fil))
p_plot=np.append(p_fil[:,:,z_pos],p_air[:,:,z_pos],axis=1)
p_plot=np.append(p_plot, p_h2o[:,:,z_pos],axis=1)
p_plot=transpose(p_plot)
a_fil = np.zeros(np.shape(t_fil))
a_plot=np.append(a_fil[:,:,z_pos],a_air[:,:,z_pos],axis=1)
a_plot=np.append(a_plot, a_h2o[:,:,z_pos],axis=1)
a_plot=transpose(a_plot)

plt.figure
plt.subplot(3,2,1)
levels_t=linspace( t_plot.min(), t_plot.max(), 11)
norm_t=cm.colors.Normalize( vmax=t_plot.max(), vmin=t_plot.min() )
cax_u=plt.contourf(xc,yc,t_plot,levels_t,cmap="rainbow",norm=norm_t)
cbar_u=plt.colorbar(cax_u)
plt.title("Temperature [m/s]")
plt.xlabel("x [m]")
#plt.ylim([-1E1,1E1])
plt.ylabel("y [m]" )

plt.subplot(3,2,2)
matplotlib.rcParams["contour.negative_linestyle"] = "solid"
cax_p=plt.contourf(xc,yc,p_plot,cmap="rainbow")
cax_p2=plt.contour(xc,yc,p_plot,colors="k")
plt.clabel(cax_p2, fontsize=12, inline=1)
cbar_p = plt.colorbar(cax_p)
plt.title("Pressure Correction")
plt.xlabel("x [m]")
#plt.ylim([-1E1,1E1])
plt.ylabel("y [m]" )

plt.subplot(3,2,3)
cax_a=plt.contourf(xc,yc,a_plot,cmap="rainbow")
cbar_a=plt.colorbar(cax_a)
plt.title("Concentration")
plt.xlabel("x [m]")
#plt.ylim([-1E1,1E1])
plt.ylabel("y [m]" )