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

data=np.load('ws_N_70_06_temp.npz')
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
zn = data['arr_5']

t_air = data['arr_6']
u_air = data['arr_7']
v_air = data['arr_8']
w_air = data['arr_9']
p_tot_air = data['arr_10']
p_air = data['arr_11']
a_air = data['arr_12']
pv_air = data['arr_13']
pv_n = data['arr_14']
pv_s = data['arr_15']

t_h2o = data['arr_16']
u_h2o = data['arr_17']
v_h2o = data['arr_18']
w_h2o = data['arr_19']
p_tot_h2o = data['arr_20']
p_h2o = data['arr_21']
a_h2o = data['arr_22']

t_fil = data['arr_23']

t_int_mem = data['arr_24']
m_j = data['arr_25']
m_pv = data['arr_26']
t_int_film = data['arr_27']
m_out = data['arr_28']

AIR = 0
H2O = 1
FIL = 2
COL = 3

#xn = (nodes(-0.05,  0.05, 150), nodes(-0.05, 0.05, 150), nodes(-0.05,   0.05, 150))
#yn = (nodes(-0.004, 0,     10), nodes( 0,    0.02,  30), nodes(-0.005, -0.004,  3))
#zn = (nodes(-0.05,  0.05, 150), nodes(-0.05, 0.05, 150), nodes(-0.05,   0.05, 150))

z_pos = 40
    
xc = avg(xn[AIR])
yc = np.append(avg(yn_fil), avg(yn_air),axis=0)
yc = np.append(yc, avg(yn_h2o),axis=0)
zc = avg(zn[AIR])

#%% vertical temperature profil

yc = np.append(avg(yn_fil),-0.0035,axis=0)
yc = np.append(yc, avg(yn_air),axis=0)
yc = np.append(yc, 0.0)
yc = np.append(yc, avg(yn_h2o),axis=0)

t_plot_s=np.append(t_fil[0,:,z_pos],t_int_film[0,:,z_pos])
t_plot_s=np.append(t_plot_s, t_air[0,:,z_pos])
t_plot_s=np.append(t_plot_s, t_int_mem[0,:,z_pos])
t_plot_s=np.append(t_plot_s, t_h2o[0,:,z_pos])

t_plot_m=np.append(t_fil[64,:,z_pos],t_int_film[64,:,z_pos])
t_plot_m=np.append(t_plot_m, t_air[64,:,z_pos])
t_plot_m=np.append(t_plot_m, t_int_mem[64,:,z_pos])
t_plot_m=np.append(t_plot_m, t_h2o[64,:,z_pos])

t_plot_e=np.append(t_fil[127,:,z_pos],t_int_film[127,:,z_pos])
t_plot_e=np.append(t_plot_e, t_air[127,:,z_pos])
t_plot_e=np.append(t_plot_e, t_int_mem[127,:,z_pos])
t_plot_e=np.append(t_plot_e, t_h2o[127,:,z_pos])

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

pylab.show

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

dz = 0.00125
dx = 0.00125

#np.savetxt('axial_membrane_flux.dat', np.transpose((xc, m_j[:,0,40]/(dx*dz)*3600, -m_out[:,0,40]/(dx*dz)*3600, t_int_film[:,0,40], a_air[:,0,40])), fmt='%1.4e',header='x m_mem m_out t_int_film a_air')

plt.figure
plt.plot(xc,m_j[:,:,40]/(dx*dz)*3600, linestyle='-', color='blue', linewidth=1.2)
plt.xlabel('X [m]',fontsize=20)
plt.ylabel('Membrane Mass Flux [kg/(m^2 h)]',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
pylab.show

#%% mem interface contour plot

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.use("pgf")
pgf_with_rc_fonts = {
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.serif": [],                   # use latex default serif font
    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
}
mpl.rcParams.update(pgf_with_rc_fonts)

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

#plt.title('Evaporation Interface Temperature [C]')
pylab.show
#plt.savefig('t_int_mem_70.pdf')
#plt.savefig('t_int_mem_70.pgf')


#%% Temperature polarization coefficient
z_pos = 40

polc_t = (t_int_mem[:,0,z_pos] - t_air[:,20,z_pos])/(t_h2o[:,4,z_pos]-t_air[:,10,z_pos])

#np.savetxt('temperature_polarization.dat', np.transpose((xc, polc_t)), fmt='%1.4e',header='x polc_t')

plt.figure()
plt.plot(polc_t[:])

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
yc = np.append(avg(yn_fil),avg(yn_air),axis=0)
yc = np.append(yc, avg(yn_h2o),axis=0)

t_plot=np.append(t_fil[:,:,z_pos],t_air[:,:,z_pos],axis=1)
t_plot=np.append(t_plot, t_h2o[:,:,z_pos],axis=1)
t_plot=transpose(t_plot)
p_fil = np.zeros(np.shape(t_fil))
p_plot=np.append(p_fil[:,:,z_pos], p_air[:,:,z_pos],axis=1)
p_plot=np.append(p_plot, p_h2o[:,:,z_pos],axis=1)
p_plot=transpose(p_plot)
a_fil = np.zeros(np.shape(t_fil))
a_plot=np.append(a_fil[:,:,z_pos], a_air[:,:,z_pos],axis=1)
a_plot=np.append(a_plot, a_h2o[:,:,z_pos],axis=1)
a_plot=transpose(a_plot)
u_fil = np.zeros((np.shape(t_fil)[0]-1,np.shape(t_fil)[1],np.shape(t_fil)[2]))
u_plot = np.concatenate([u_fil[:,:,z_pos],u_air[:,:,z_pos],u_h2o[:,:,z_pos]],axis=1)
u_plot = transpose(u_plot) 
p_tot_fil = np.zeros(np.shape(t_fil))
p_tot_plot=np.append(p_tot_fil[:,:,z_pos], p_tot_air[:,:,z_pos],axis=1)
p_tot_plot=np.append(p_tot_plot, p_tot_h2o[:,:,z_pos],axis=1)
p_tot_plot=transpose(p_tot_plot)

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

plt.subplot(3,2,4)
matplotlib.rcParams["contour.negative_linestyle"] = "solid"
cax_p=plt.contourf(xc,yc,p_tot_plot,cmap="rainbow")
cax_p2=plt.contour(xc,yc,p_tot_plot,colors="k")
plt.clabel(cax_p2, fontsize=12, inline=1)
cbar_p = plt.colorbar(cax_p)
plt.title("Total Pressure")
plt.xlabel("x [m]")
#plt.ylim([-1E1,1E1])
plt.ylabel("y [m]" )

plt.subplot(3,2,6)
cax_u=plt.contourf(avg(xc),yc,u_plot,cmap="rainbow")
cbar_u=plt.colorbar(cax_u)
plt.title("Axial Velocity")
plt.xlabel("x [m]")
#plt.ylim([-1E1,1E1])
plt.ylabel("y [m]" )