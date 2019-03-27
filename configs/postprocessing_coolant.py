# -*- coding: utf-8 -*-
"""
Postprocessing script for simulations with coolant:
Loads the .npz files

The plots are for normal configuration and need to be adjusted for reversed
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")

# Standard Python modules
from pyns.standard import *

# PyNS modules
from pyns.constants          import *
from pyns.operators          import *
from pyns.discretization     import *

# Identifier of the simulation to be loaded
name = 'ws_N_coolant_70_6_temp'

data=np.load(name + '.npz')
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

# z position for the following plots
z_pos = 40
    
xc = avg(xn[AIR])
yc = np.append(avg(yn_col), avg(yn_fil),axis=0)
yc = np.append(yc, avg(yn_air),axis=0)
yc = np.append(yc, avg(yn_h2o),axis=0)
zc = avg(zn[AIR])

#%% vertical temperature profil for normal configuration, reordering necessary for reversed

x_end= np.shape(xn)[1]-2
x_mid= np.int(np.round((np.shape(xn)[1]-2)/2))

yc = np.append(avg(yn_col),avg(yn_fil),axis=0)
yc = np.append(yc, -0.0035) # location of condensation interface
yc = np.append(yc, avg(yn_air),axis=0)
yc = np.append(yc, 0.0) # location of membrane
yc = np.append(yc, avg(yn_h2o),axis=0)

t_plot_s=np.append(t_col[0,:,z_pos],t_fil[0,:,z_pos])
t_plot_s=np.append(t_plot_s, t_int_film[0,:,z_pos])
t_plot_s=np.append(t_plot_s, t_air[0,:,z_pos])
t_plot_s=np.append(t_plot_s, t_int_mem[0,:,z_pos])
t_plot_s=np.append(t_plot_s, t_h2o[0,:,z_pos])

t_plot_m=np.append(t_col[x_mid,:,z_pos],t_fil[x_mid,:,z_pos])
t_plot_m=np.append(t_plot_m, t_int_film[x_mid,:,z_pos])
t_plot_m=np.append(t_plot_m, t_air[x_mid,:,z_pos])
t_plot_m=np.append(t_plot_m, t_int_mem[x_mid,:,z_pos])
t_plot_m=np.append(t_plot_m, t_h2o[x_mid,:,z_pos])

t_plot_e=np.append(t_col[x_end,:,z_pos],t_fil[x_end,:,z_pos])
t_plot_e=np.append(t_plot_e, t_int_film[x_end,:,z_pos])
t_plot_e=np.append(t_plot_e, t_air[x_end,:,z_pos])
t_plot_e=np.append(t_plot_e, t_int_mem[x_end,:,z_pos])
t_plot_e=np.append(t_plot_e, t_h2o[x_end,:,z_pos])

# Saving option for further use in pgf plot
#np.savetxt('vertical_temp_70.dat', np.transpose((yc, t_plot_s, t_plot_m, t_plot_e)), fmt='%1.4e',header='y ts tm te')

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
plt.xlabel('Temperature [C]',fontsize=20)

plt.subplot(1,3,3)
plt.plot(t_plot_e,yc, linestyle='-', color='blue', linewidth=1.2)
plt.plot([18, 80],[0, 0], linestyle='--', color='black', linewidth=1)
plt.plot([18, 80],[-0.0035, -0.0035], linestyle='--', color='black', linewidth=1)
plt.plot([18, 80],[-0.004, -0.004], linestyle='--', color='black', linewidth=1)
plt.xlim([18, 80])
plt.xticks([20,40,60,80],fontsize=18)
plt.yticks([])

#%% streamlines in air gap

y,x = np.mgrid[0:20:1,0:127:1]
#y,x = np.mgrid[0:0.125:0.00390625,0:1.25:0.0048828125]
uu = np.transpose(u_air[:,0:20,40])
vv = np.transpose(v_air[0:127,:,40])
ww = np.transpose(w_air[0:127,0:20,40])
U=np.sqrt(uu*uu+vv*vv+ww*ww)

plt.figure()
plt.gca(aspect="equal")
#plt.contourf(x,y,np.transpose(obst[:,:,16]),[0.7,1.2],colors=('black'))
plt.streamplot(x,y,uu,vv, color=U, linewidth=2)
#plt.xticks([0,0.25,0.5,0.75,1,1.25],fontsize=18)
#plt.yticks([0,0.06,0.12],fontsize=18)
plt.ylabel('Y [m]',fontsize=20)
plt.xlabel('X [m]',fontsize=20)

#%% axial membrane flux


# correct these values for each simulation!!!!
dz = 0.00125
dx = 0.00125

# Saving option for further use in pgf plot
#np.savetxt('axial_membrane_flux.dat', np.transpose((xc, m_j[:,0,40]/(dx*dz)*3600, -m_out[:,0,40]/(dx*dz)*3600, t_int_film[:,0,40], a_air[:,0,40])), fmt='%1.4e',header='x m_mem m_out t_int_film a_air')

plt.figure
plt.plot(xc,m_j[:,:,40]/(dx*dz)*3600, linestyle='-', color='blue', linewidth=1.2)
plt.xlabel('X [m]',fontsize=20)
plt.ylabel('Membrane Mass Flux [kg/(m^2 h)]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()

#%% mem interface contour plot, options for pgf plot are commented

from mpl_toolkits.axes_grid1 import make_axes_locatable

#import matplotlib as mpl
#mpl.use("pgf")
#pgf_with_rc_fonts = {
#    "pgf.texsystem": "pdflatex",
#    "font.family": "serif",
#    "font.serif": [],                   # use latex default serif font
#    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
#}
#mpl.rcParams.update(pgf_with_rc_fonts)

plt.figure(figsize=(4.5,2.5))
plt.gca(aspect="equal")
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.xticks((0.04, 0.08, 0.12))
plt.tight_layout()
ax = plt.gca()
im = plt.contourf(xc,zc,np.transpose(t_int_mem[:,0,:]),cmap='viridis')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)

# save options
#plt.savefig('t_int_mem_70.pdf')
#plt.savefig('t_int_mem_70.pgf')


#%% Temperature polarization coefficient
z_pos = 40

polc_t = (t_int_mem[:,0,z_pos] - t_air[:,20,z_pos])/(t_h2o[:,4,z_pos]-t_air[:,10,z_pos])

# Saving option for further use in pgf plot
#np.savetxt('temperature_polarization.dat', np.transpose((xc, polc_t)), fmt='%1.4e',header='x polc_t')

plt.figure()
plt.plot(polc_t[:])

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

#%% contour plots

xc = avg(xn[AIR])
yc = np.append(avg(yn_col),avg(yn_fil),axis=0)
yc = np.append(yc, avg(yn_air),axis=0)
yc = np.append(yc, avg(yn_h2o),axis=0)

t_plot=np.append(t_col[:,:,z_pos],t_fil[:,:,z_pos],axis=1)
t_plot=np.append(t_plot, t_air[:,:,z_pos],axis=1)
t_plot=np.append(t_plot, t_h2o[:,:,z_pos],axis=1)
t_plot=transpose(t_plot)
p_fil = np.zeros(np.shape(t_fil))
p_plot=np.append(p_col[:,:,z_pos],p_fil[:,:,z_pos],axis=1)
p_plot=np.append(p_plot, p_air[:,:,z_pos],axis=1)
p_plot=np.append(p_plot, p_h2o[:,:,z_pos],axis=1)
p_plot=transpose(p_plot)
a_fil = np.zeros(np.shape(t_fil))
a_col = np.zeros(np.shape(t_col))
a_plot=np.append(a_col[:,:,z_pos],a_fil[:,:,z_pos],axis=1)
a_plot=np.append(a_plot, a_air[:,:,z_pos],axis=1)
a_plot=np.append(a_plot, a_h2o[:,:,z_pos],axis=1)
a_plot=transpose(a_plot)
u_fil = np.zeros((np.shape(t_fil)[0]-1,np.shape(t_fil)[1],np.shape(t_fil)[2]))
u_plot = np.concatenate([u_col[:,:,z_pos],u_fil[:,:,z_pos],u_air[:,:,z_pos],u_h2o[:,:,z_pos]],axis=1)
u_plot = transpose(u_plot) 
p_tot_fil = np.zeros(np.shape(t_fil))
p_tot_plot=np.append(p_tot_col[:,:,z_pos],p_tot_fil[:,:,z_pos],axis=1)
p_tot_plot=np.append(p_tot_plot, p_tot_air[:,:,z_pos],axis=1)
p_tot_plot=np.append(p_tot_plot, p_tot_h2o[:,:,z_pos],axis=1)
p_tot_plot=transpose(p_tot_plot)

plt.figure
plt.subplot(3,2,1)
levels_t=linspace( t_plot.min(), t_plot.max(), 11)
norm_t=cm.colors.Normalize( vmax=t_plot.max(), vmin=t_plot.min() )
cax_u=plt.contourf(xc,yc,t_plot,levels_t,cmap="rainbow",norm=norm_t)
cbar_u=plt.colorbar(cax_u)
plt.title("Temperature")
plt.xlabel("x [m]")
plt.ylabel("y [m]" )

plt.subplot(3,2,2)
matplotlib.rcParams["contour.negative_linestyle"] = "solid"
cax_p=plt.contourf(xc,yc,p_plot,cmap="rainbow")
cax_p2=plt.contour(xc,yc,p_plot,colors="k")
plt.clabel(cax_p2, fontsize=12, inline=1)
cbar_p = plt.colorbar(cax_p)
plt.title("Pressure Correction")
plt.xlabel("x [m]")
plt.ylabel("y [m]" )

plt.subplot(3,2,3)
cax_a=plt.contourf(xc,yc,a_plot,cmap="rainbow")
cbar_a=plt.colorbar(cax_a)
plt.title("Concentration")
plt.xlabel("x [m]")
plt.ylabel("y [m]" )

plt.subplot(3,2,4)
matplotlib.rcParams["contour.negative_linestyle"] = "solid"
cax_p=plt.contourf(xc,yc,p_tot_plot,cmap="rainbow")
cax_p2=plt.contour(xc,yc,p_tot_plot,colors="k")
plt.clabel(cax_p2, fontsize=12, inline=1)
cbar_p = plt.colorbar(cax_p)
plt.title("Total Pressure")
plt.xlabel("x [m]")
plt.ylabel("y [m]" )

plt.subplot(3,2,6)
cax_u=plt.contourf(avg(xc),yc,u_plot,cmap="rainbow")
cbar_u=plt.colorbar(cax_u)
plt.title("Axial Velocity")
plt.xlabel("x [m]")
plt.ylabel("y [m]" )