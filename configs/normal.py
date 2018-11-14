
#!/usr/bin/python

import sys
sys.path.append("../..")
import time

time_start = time.time()

# Standard Python modules
from pyns.standard import *

import matplotlib.pylab as pylab
import numpy as np
import matplotlib
from scipy.optimize import fsolve
from scipy.constants import R

# PyNS modules
from pyns.constants          import *
from pyns.operators          import *
from pyns.discretization     import *
from pyns.display            import plot, write
from pyns.physical           import properties
from pyns.physical.constants import G

# membrane aux functions
from pyns.demo.p_v_sat import *
from pyns.demo.calc_interface import *
from pyns.demo.latent_heat import *

plt.close("all")

#==========================================================================
#
# Define problem
#
#==========================================================================

AIR = 0
H2O = 1
FIL = 2

u_h_in = 0.025 # m/s
t_h_in = 80   # C
a_salt = 90.0 # g/l
t_c_in = 15   # C

# when setting the air gap thickness here
# MAKE SURE TO ADJUST THE NUMBER OF CELLS IN THE AIR GAP
# in line 62 accordingly!!!
airgap = 0.008 # m

name = 'N_' + str(t_h_in) + '_' + str(u_h_in).replace("0.", "") + '_' + str(airgap).replace("0.00", "")

# restart options
restart = False
restart_file = 'ws_' + name + '_temp.npz'

# Node coordinates for both domains
xn = (nodes(0,   0.07, 56), nodes(0, 0.07, 56), nodes(0,       0.07, 56))
yn = (nodes(-airgap, 0,32), nodes(0, 0.01, 26), nodes(- airgap - 0.0005, -airgap, 3))
zn = (nodes(0,   0.07, 56), nodes(0, 0.07, 56), nodes(0,       0.07,  56))

# Cell coordinates 
xc = (avg(xn[AIR]), avg(xn[H2O]), avg(xn[FIL]))
yc = (avg(yn[AIR]), avg(yn[H2O]), avg(yn[FIL]))
zc = (avg(zn[AIR]), avg(zn[H2O]), avg(zn[FIL]))

# Cell dimensions
cell = [cartesian_grid(xn[AIR],yn[AIR],zn[AIR]),  \
        cartesian_grid(xn[H2O],yn[H2O],zn[H2O]),  \
        cartesian_grid(xn[FIL],yn[FIL],zn[FIL])]

nx,ny,nz, dx,dy,dz = (cell[AIR][0], cell[H2O][0], cell[FIL][0]),  \
                     (cell[AIR][1], cell[H2O][1], cell[FIL][1]),  \
                     (cell[AIR][2], cell[H2O][2], cell[FIL][2]),  \
                     (cell[AIR][3], cell[H2O][3], cell[FIL][3]),  \
                     (cell[AIR][4], cell[H2O][4], cell[FIL][4]),  \
                     (cell[AIR][5], cell[H2O][5], cell[FIL][5])
rc,ru,rv,rw =        (cell[AIR][6], cell[H2O][6], cell[FIL][6]),  \
                     (cell[AIR][7], cell[H2O][7], cell[FIL][7]),  \
                     (cell[AIR][8], cell[H2O][8], cell[FIL][8]),  \
                     (cell[AIR][9], cell[H2O][9], cell[FIL][9])
					 
# Set physical properties temperature dependent:
prop = [properties.air(round((t_h_in+t_c_in)/2,-1),rc[AIR]), \
        properties.water(t_h_in,rc[H2O]),                    \
        properties.water(t_c_in,rc[FIL])]

rho, mu, cap, kappa = (prop[AIR][0], prop[H2O][0], prop[FIL][0]),  \
                      (prop[AIR][1], prop[H2O][1], prop[FIL][1]),  \
                      (prop[AIR][2], prop[H2O][2], prop[FIL][2]),  \
                      (prop[AIR][3], prop[H2O][3], prop[FIL][3])
                      
diff = (ones(rc[AIR])*3.5E-5, ones(rc[H2O])*1.99E-09)

h_d = [ 0.0, latent_heat(t_h_in), latent_heat(t_c_in)]

M_H2O  = 18E-3      # kg/mol
M_AIR  = 28E-3      # kg/mol
M_salt = 58.4428E-3 # kg/mol      

pi = np.pi

# for density interpolation
t_interp = [20,30,40,50,60,70,80]
rho_air = zeros(np.shape(t_interp))
rho_water = zeros(np.shape(t_interp))
for count,tem in enumerate(t_interp):
  rho_air[count]   = properties.air(tem,1,False)[0]
  rho_water[count] = properties.water(tem,1,False)[0]


# Membrane properties
membrane = namedtuple("membrane", "d kap eps tau r p t t_int pv j t_old t_int_old")
  # d is thickness in m, kap is thermal conductivity in W/mK
  # eps is porosity, tau is tortuosity
  # r is pore radius

# values duropore PDVF membrane
mem = membrane(65E-6,   \
                 0.25,    \
                 0.85,  \
                 1.5,      \
                 0.1E-6, \
                 zeros((nx[AIR],1,nz[AIR])), \
                 zeros((nx[AIR],1,nz[AIR])), \
                 zeros((nx[AIR],1,nz[AIR])), \
                 zeros((nx[AIR],1,nz[AIR])), \
                 zeros((nx[AIR],1,nz[AIR])), \
                 zeros((nx[AIR],1,nz[AIR])), \
                 zeros((nx[AIR],1,nz[AIR])))
                               
      
 
# Create unknowns; names, positions and sizes
uf    = [Unknown("face-u-vel",    X, ru[AIR], DIRICHLET),  \
         Unknown("face-u-vel",    X, ru[H2O], DIRICHLET),  \
         Unknown("face-u-vel",    X, ru[FIL], DIRICHLET)]
vf    = [Unknown("face-v-vel",    Y, rv[AIR], DIRICHLET),  \
         Unknown("face-v-vel",    Y, rv[H2O], DIRICHLET),  \
         Unknown("face-v-vel",    Y, rv[FIL], DIRICHLET)]
wf    = [Unknown("face-w-vel",    Z, rw[AIR], DIRICHLET),  \
         Unknown("face-w-vel",    Z, rw[H2O], DIRICHLET),  \
         Unknown("face-w-vel",    Z, rw[FIL], DIRICHLET)]
p     = [Unknown("pressure",      C, rc[AIR], NEUMANN),  \
         Unknown("pressure",      C, rc[H2O], NEUMANN),  \
         Unknown("pressure",      C, rc[FIL], NEUMANN)]
t     = [Unknown("temperature",   C, rc[AIR], NEUMANN),  \
         Unknown("temperature",   C, rc[H2O], NEUMANN),  \
         Unknown("temperature",   C, rc[FIL], NEUMANN)]
a     = [Unknown("concentration", C, rc[AIR], NEUMANN),  \
         Unknown("concentration", C, rc[H2O], NEUMANN),  \
         Unknown("concentration", C, rc[FIL], NEUMANN)]
p_tot = [Unknown("pressure-tot",  C, rc[AIR], NEUMANN),  \
         Unknown("pressure-tot",  C, rc[H2O], NEUMANN),  \
         Unknown("pressure-tot",  C, rc[FIL], NEUMANN)]

# just for air
p_v =[Unknown("vapor_pressure",C, rc[AIR], NEUMANN)]
M  = [Unknown("molar mass",    C, rc[AIR], NEUMANN)]

# initialize source terms
q_t = [zeros(rc[AIR]),
       zeros(rc[H2O]), 
       zeros(rc[FIL])]
q_a = [zeros(rc[AIR]),
       zeros(rc[H2O])]
dv  = [dx[AIR]*dy[AIR]*dz[AIR], 
       dx[H2O]*dy[H2O]*dz[H2O], 
       dx[FIL]*dy[FIL]*dz[FIL]]
       
# variables to temporarily store vf bnd values:
vf_h2o_S_tmp=zeros(vf[H2O].bnd[S].val.shape)

# Specify boundary conditions  
uf[H2O].bnd[W].val[:1,:,:] = u_h_in
uf[H2O].val[:,:,:] = u_h_in
uf[H2O].bnd[E].typ[:1,:,:] = OUTLET 
uf[H2O].bnd[E].val[:1,:,:] = u_h_in
#uf[AIR].bnd[W].typ[:1,:,:] = OUTLET     
  
t[AIR].bnd[S].typ[:,:1,:] = DIRICHLET  
t[AIR].bnd[S].val[:,:1,:] = t_c_in
t[AIR].bnd[N].typ[:,:1,:] = DIRICHLET  
t[AIR].bnd[N].val[:,:1,:] = t_h_in

t[H2O].bnd[W].typ[:1,:,:] = DIRICHLET
t[H2O].bnd[W].val[:1,:,:] = t_h_in
t[H2O].bnd[S].typ[:,:1,:] = DIRICHLET  
t[H2O].bnd[S].val[:,:1,:] = t_h_in
 
t[FIL].bnd[S].typ[:,:1,:] = DIRICHLET
t[FIL].bnd[S].val[:,:1,:] = t_c_in
t[FIL].bnd[N].typ[:,:1,:] = DIRICHLET  
t[FIL].bnd[N].val[:,:1,:] = t_c_in

mem.t_int[:,:,:] = t[H2O].bnd[S].val[:,:1,:] # temporary

a[H2O].bnd[W].typ[:1,:,:] = DIRICHLET
a[H2O].bnd[W].val[:1,:,:] = a_salt/rho[H2O][:1,:,:]

M[AIR].bnd[S].typ[:,:1,:] = DIRICHLET
M[AIR].bnd[S].val[:,:1,:] = M[AIR].val[:,:1,:]

p_v[AIR].bnd[S].typ[:,:,:] = DIRICHLET
p_v[AIR].bnd[S].val[:,:,:] = p_v_sat(t[FIL].bnd[N].val[:,:,:])
p_v[AIR].bnd[N].typ[:,:,:] = DIRICHLET
p_v[AIR].bnd[N].val[:,:,:] = p_v_sat(t[H2O].bnd[S].val[:,:,:])
p_v[AIR].bnd[S].typ[:,:,:] = DIRICHLET
p_v[AIR].bnd[S].val[:,:,:] = p_v_sat(t[FIL].bnd[N].val[:,:,:])

t[AIR].val[:,:,:] = np.reshape(np.linspace(t_c_in,t_h_in,num=t[AIR].val.shape[1]),[1,t[AIR].val.shape[1],1]) #round((t_h_in+t_c_in)/2,-1) #
t[H2O].val[:,:,:] = t_h_in
t[FIL].val[:,:,:] = t_c_in

a[AIR].val[:,:,:] = p_v_sat(t[AIR].val[:,:,:])*1E-5*M_H2O/M_AIR
M[AIR].val[:,:,:] = 1/((1-a[AIR].val[:,:,:])/M_AIR + a[AIR].val[:,:,:]/M_H2O)
a[H2O].val[:,:,:] = a_salt/rho[H2O][:,:,:]
 
for c in (AIR,H2O,FIL):
  adj_n_bnds(p[c])
  adj_n_bnds(t[c])
  adj_n_bnds(a[c])
  
# max values in domain:
t_max = 0.0
t_min = 100.0

for c in (W,T):
  t_max = np.amax([t_max, np.amax(t[H2O].bnd[c].val)])
  t_min = np.amin([t_min, np.amin(t[FIL].bnd[c].val)]) 
  
  # Time-stepping parameters
dt  =    0.0001  # time step
ndt =    70000  # number of time steps
dt_plot = ndt    # plot frequency
dt_save = 500
dt_save_ts = 10000
tss = 1

obst = [zeros(rc[AIR]), zeros(rc[H2O]),zeros(rc[FIL])]

# initialize tracking of change
change_t = zeros(ndt)
change_a = zeros(ndt)
change_p = zeros(ndt)

time_start=time.time()

#%%

if restart==True:
  
  data=np.load(restart_file)
    
  tss = data['arr_0']-1
  
  t[AIR].val[:,:,:] = data['arr_6']
  uf[AIR].val[:,:,:] = data['arr_7']
  vf[AIR].val[:,:,:] = data['arr_8']
  wf[AIR].val[:,:,:] = data['arr_9']
  p_tot[AIR].val[:,:,:] = data['arr_10']
  p[AIR].val[:,:,:] = data['arr_11']
  a[AIR].val[:,:,:] = data['arr_12']
  p_v[AIR].val[:,:,:] = data['arr_13']
  p_v[AIR].bnd[N].val[:,:,:] = data['arr_14']
  p_v[AIR].bnd[S].val[:,:,:] = data['arr_15']
  
  t[H2O].val[:,:,:] = data['arr_16']
  uf[H2O].val[:,:,:] = data['arr_17']
  vf[H2O].val[:,:,:] = data['arr_18']
  wf[H2O].val[:,:,:] = data['arr_19']
  p_tot[H2O].val[:,:,:] = data['arr_20']
  p[H2O].val[:,:,:] = data['arr_21']
  a[H2O].val[:,:,:] = data['arr_22']
  
  t[FIL].val[:,:,:] = data['arr_23']
  
  mem.t_int[:,:,:] = data['arr_24']
  mem.j[:,:,:] = data['arr_25']
  mem.pv[:,:,:] = data['arr_26']
  t_int = data['arr_27']
  m_evap = data['arr_28']
  
  print("data from file loaded")
  
  for c in (AIR,H2O,FIL):
    adj_n_bnds(p[c])
    adj_n_bnds(t[c])
    adj_n_bnds(a[c])

#%%

#==========================================================================
#
# Solution algorithm
#
#==========================================================================

#-----------
#
# Time loop 
#
#-----------
for ts in range(tss,ndt+1):
  
  write.time_step(ts)
 
  #------------------
  # Store old values
  #------------------
  for c in (AIR,H2O,FIL):
    a[c].old[:]  = a[c].val
    t[c].old[:]  = t[c].val
    p[c].old[:]  = p[c].val
    uf[c].old[:] = uf[c].val
    vf[c].old[:] = vf[c].val
    wf[c].old[:] = wf[c].val
  mem.t_old[:] = mem.t
  mem.t_int_old[:] = mem.t_int
  
                     
  rho[AIR][:,:,:] = np.interp(t[AIR].val, t_interp, rho_air)
  rho[H2O][:,:,:] = np.interp(t[H2O].val, t_interp, rho_water)
    
  #------------------------
  # Heat and Mass Transfer between Domains
  #------------------------  
  
  # AIR domain values: Partial vapor pressure & molar mass
  M[AIR].val[:,:,:] = 1/((1-a[AIR].val[:,:,:])/M_AIR + a[AIR].val[:,:,:]/M_H2O)  
  p_v[AIR].val[:,:,:] = a[AIR].val[:,:,:] *M[AIR].val[:,:,:]/M_H2O * (p_tot[AIR].val[:,:,:] +1E5) 
    
  # Interphase energy equation between AIR & FIL
  t_int, m_evap, t, p_v = calc_interface(t, a, p_v, p_tot, kappa, M, \
                            M_AIR, M_H2O, h_d, (dx,dy,dz), (AIR, FIL))  
  
  # upward (positive) velocity induced through evaporation (positive m_evap) 
  q_a[AIR][:,:1,:]  = m_evap[:,:1,:] / dv[AIR][:,:1,:] 
  
  # Membrane diffusion and energy equation between H2O & AIR
  mem, t, p_v = calc_membrane(t, a, p_v, p_tot, mem, kappa, diff, M, \
                    (M_AIR,M_H2O,M_salt), h_d, (dx,dy,dz), (AIR, H2O))
  
  # downward (negative) velocity induced through evaporation (positive mem_j)                
  vf[H2O].bnd[S].val[:,:1,:] = -mem.j[:,:1,:]/(rho[H2O][:,:1,:]*dx[H2O][:,:1,:]*dz[H2O][:,:1,:]) 
  q_a[AIR][:,-1:,:] = mem.j [:,:1,:] / dv[AIR][:,-1:,:] 
         
  #------------------------
  # Concentration
  #------------------------
  
  # correct for salt retention in feed water
  vf_h2o_S_tmp[:,:,:] = vf[H2O].bnd[S].val[:,:1,:]
  vf[H2O].bnd[S].val[:,:1,:] = 0.0
  
  # in case of v[H2O].bnd[S].val ~= 0 correct convection into membrane 
  for c in (AIR,H2O):
    calc_t(a[c], (uf[c],vf[c],wf[c]), rho[c], diff[c],  \
           dt, (dx[c],dy[c],dz[c]), 
           obstacle = obst[c],
           source = q_a[c])
           
  for c in (AIR,H2O):
    a[c].val[a[c].val < 0.0] = 0.0
    
  vf[H2O].bnd[S].val[:,:1,:] = vf_h2o_S_tmp[:,:,:]
    
  #------------------------
  # Temperature (enthalpy)
  #------------------------
  
  for c in (AIR,H2O,FIL):
    calc_t(t[c], (uf[c],vf[c],wf[c]), (rho[c]*cap[c]), kappa[c],  \
           dt, (dx[c],dy[c],dz[c]), 
           obstacle = obst[c],
           source = q_t[c])

  for c in (AIR,H2O,FIL):
    t[c].val[t[c].val > t_max] = t_max
    t[c].val[t[c].val < t_min] = t_min

  #-----------------------
  # Momentum conservation
  #-----------------------
  for c in (AIR,H2O):
    #g_u = -G * avg(X, rho[c])
    
    g_v = -G * avg(Y, rho[c])
  
    ef = zeros(ru[c]), g_v, zeros(rw[c])
    
    
    calc_uvw((uf[c],vf[c],wf[c]), (uf[c],vf[c],wf[c]), rho[c], mu[c],  \
             dt, (dx[c],dy[c],dz[c]), 
             obstacle = obst[c],
             pressure = p_tot[c],
             force = ef)
  
  #----------
  # Pressure
  #----------
  for c in (AIR,H2O):
    calc_p(p[c], (uf[c],vf[c],wf[c]), rho[c],  \
           dt, (dx[c],dy[c],dz[c]), 
           obstacle = obst[c])
  
    p_tot[c].val = p_tot[c].val + p[c].val
  
  #---------------------
  # Velocity correction
  #---------------------
  for c in (AIR,H2O):
    corr_uvw((uf[c],vf[c],wf[c]), p[c], rho[c],  \
             dt, (dx[c],dy[c],dz[c]), 
             obstacle = obst[c])
 
#  # Compute volume balance for checking 
#  for c in (AIR,H2O,COL):
#    err = vol_balance((uf[c],vf[c],wf[c]),  \
#                      (dx[c],dy[c],dz[c]), 
#                      obstacle = obst[c])
#    print("Maximum volume error after correction: %12.5e" % abs(err).max())

  # Check the CFL number too 
  for c in (AIR,H2O):
    cfl = cfl_max((uf[c],vf[c],wf[c]), dt, (dx[c],dy[c],dz[c]))
    # print("Maximum CFL number: %12.5e" % cfl)
    
  if ts % dt_save == 0:
      ws_tmp_name = 'ws_' + name + '_temp.npz'
      np.savez(ws_tmp_name, ts, xn, yn[AIR], yn[H2O], yn[FIL], zn, t[AIR].val, uf[AIR].val,vf[AIR].val,wf[AIR].val, p_tot[AIR].val, p[AIR].val, a[AIR].val,  p_v[AIR].val, p_v[AIR].bnd[N].val, p_v[AIR].bnd[S].val, t[H2O].val, uf[H2O].val,vf[H2O].val,wf[H2O].val,p_tot[H2O].val, p[H2O].val, a[H2O].val, t[FIL].val, mem.t_int, mem.j, mem.pv, t_int,m_evap )
      time_end = time.time()     
      print("Total time: %4.4e" % ((time_end-time_start)/3600))
      if ts % dt_save_ts ==0:
        ws_save_title = 'ws_' + name + '_' + str(ts) + 'ts.npz'
        np.savez(ws_save_title, ts, xn, yn[AIR], yn[H2O], yn[FIL], zn, t[AIR].val, uf[AIR].val,vf[AIR].val,wf[AIR].val, p_tot[AIR].val, p[AIR].val, a[AIR].val,  p_v[AIR].val, p_v[AIR].bnd[N].val, p_v[AIR].bnd[S].val, t[H2O].val, uf[H2O].val,vf[H2O].val,wf[H2O].val,p_tot[H2O].val, p[H2O].val, a[H2O].val, t[FIL].val, mem.t_int, mem.j, mem.pv, t_int,m_evap )
        text_id = 'Output_' + name + '_' + str(ts) + '.txt'
        text_file = open(text_id, "w")
        airgap_outfile = 0.0035
        massflow_outfile = np.sum(m_evap) \
                     /np.sum(dx[AIR][:,-1:,:]*dz[AIR][:,-1:,:])*3600 
        RR_outfile = (-np.sum(np.sum(m_evap)))/(u_h_in*np.mean(rho[H2O][:1,:,:])\
                     *np.sum(np.sum(dx[AIR][:,-1:,:]*dz[AIR][:,-1:,:])))
        GOR_outfile = RR_outfile * h_d[H2O]/(np.mean(cap[H2O][:1,:,:]) \
                     *(t_h_in - np.mean(t[H2O].val[-1:,:,:])))
        text_file.write("t_h_in u_h_in airgap m_evap RR GOR\n")
        text_file.write("{0:2.0f} {1:1.3f} {2:1.4f} {3:2.3e} {4:2.4e} {5:2.4e}".format \
          (t_h_in, u_h_in, airgap_outfile, massflow_outfile, RR_outfile, GOR_outfile))
        text_file.close()
    
  # Check relative change in domain:
  change_t[ts-1] = (np.absolute(t[H2O].val-t[H2O].old)).max()/t[H2O].old.max()
  change_t[ts-1] = max(change_t[ts-1],(np.absolute(t[AIR].val-t[AIR].old)).max()/t[AIR].old.max())
  change_t[ts-1] = max(change_t[ts-1],(np.absolute(t[FIL].val-t[FIL].old)).max()/t[FIL].old.max()) 
  
  change_a[ts-1] = (np.absolute(a[H2O].val-a[H2O].old)).max()/a[H2O].old.max()
  change_a[ts-1] = max(change_a[ts-1],(np.absolute(a[AIR].val-a[AIR].old)).max()/a[AIR].old.max())
  if p[H2O].old.max() > 0.0: 
    change_p[ts-1] = (np.absolute(p[H2O].val-p[H2O].old)).max()/p[H2O].old.max()
    

#==========================================================================
#
# Visualisation
#
#========================================================================== 

#%%
  if ts % dt_plot == 0:
    plt.close("all")
    
    z_pos = 10
    
    xc = avg(xn[AIR])
    yc = np.append(avg(yn[FIL]), avg(yn[AIR]),axis=0)
    yc = np.append(yc, avg(yn[H2O]),axis=0)
    
    t_plot=np.append(t[FIL].val[:,:,z_pos],t[AIR].val[:,:,z_pos],axis=1)   
    t_plot=np.append(t_plot, t[H2O].val[:,:,z_pos],axis=1)
    t_plot=transpose(t_plot)
    p_plot=np.append(p[FIL].val[:,:,z_pos],p[AIR].val[:,:,z_pos],axis=1)
    p_plot=np.append(p_plot, p[H2O].val[:,:,z_pos],axis=1)
    p_plot=transpose(p_plot)
    a_plot=np.append(a[FIL].val[:,:,z_pos],a[AIR].val[:,:,z_pos],axis=1)
    a_plot=np.append(a_plot, a[H2O].val[:,:,z_pos],axis=1)
    a_plot=transpose(a_plot)
    
    uc_air = avg_x(cat_x((uf[AIR].bnd[W].val[:1,:,:], uf[AIR].val, uf[AIR].bnd[E].val[:1,:,:])))
    uc_h2o = avg_x(cat_x((uf[H2O].bnd[W].val[:1,:,:], uf[H2O].val, uf[H2O].bnd[E].val[:1,:,:])))
    uc_fil = avg_x(cat_x((uf[FIL].bnd[W].val[:1,:,:], uf[FIL].val, uf[FIL].bnd[E].val[:1,:,:])))
    u_plot = np.concatenate([uc_fil[:,:,z_pos],uc_air[:,:,z_pos],uc_h2o[:,:,z_pos]],axis=1)
    u_plot = transpose(u_plot)    
    
    plt.figure
    plt.subplot(2,2,1)
    levels_t=linspace( t_plot.min(), t_plot.max(), 11)
    norm_t=cm.colors.Normalize( vmax=t_plot.max(), vmin=t_plot.min() )
    cax_t=plt.contourf(xc,yc,t_plot,levels_t,cmap="rainbow",norm=norm_t)
    cbar_t=plt.colorbar(cax_t)
    plt.title("Temperature")
    plt.xlabel("x [m]")
    #plt.ylim([-1E1,1E1])
    plt.ylabel("y [m]" )
    
    plt.subplot(2,2,2)
    matplotlib.rcParams["contour.negative_linestyle"] = "solid"
    cax_p=plt.contourf(xc,yc,p_plot,cmap="rainbow")
    cax_p2=plt.contour(xc,yc,p_plot,colors="k")
    plt.clabel(cax_p2, fontsize=12, inline=1)
    cbar_p = plt.colorbar(cax_p)
    plt.title("Pressure Correction")
    plt.xlabel("x [m]")
    #plt.ylim([-1E1,1E1])
    plt.ylabel("y [m]" )
    
    plt.subplot(2,2,3)
    cax_a=plt.contourf(xc,yc,a_plot,cmap="rainbow")
    cbar_a=plt.colorbar(cax_a)
    plt.title("Concentration")
    plt.xlabel("x [m]")
    #plt.ylim([-1E1,1E1])
    plt.ylabel("y [m]" )
    
    plt.subplot(2,2,4)
    cax_u=plt.contourf(xc,yc,u_plot,cmap="rainbow")
    cbar_u=plt.colorbar(cax_u)
    plt.title("Axial Velocity")
    plt.xlabel("x [m]")
    #plt.ylim([-1E1,1E1])
    plt.ylabel("y [m]" )
    
    #pylab.savefig('membrane.pdf')

    pylab.show()

    #for c in (AIR,H2O):
    #  plot_isolines(t[c].val, (uf[c],vf[c],wf[c]), (xn[c],yn[c],zn[c]), Z)
     # plot_isolines(p_tot[c], (uf[c],vf[c],wf[c]), (xn[c],yn[c],zn[c]), Z)
     
time_end = time.time()     
print("Total time: %4.4e" % ((time_end-time_start)/3600))

#%%

z_pos = 40

uc_air = avg_x(cat_x((uf[AIR].bnd[W].val[:1,:,:], uf[AIR].val, uf[AIR].bnd[E].val[:1,:,:])))
uc_h2o = avg_x(cat_x((uf[H2O].bnd[W].val[:1,:,:], uf[H2O].val, uf[H2O].bnd[E].val[:1,:,:])))
uc_fil = avg_x(cat_x((uf[FIL].bnd[W].val[:1,:,:], uf[FIL].val, uf[FIL].bnd[E].val[:1,:,:])))
u_plot = np.concatenate([uc_fil[:,:,z_pos],uc_air[:,:,z_pos],uc_h2o[:,:,z_pos]],axis=1)
u_plot = transpose(u_plot)

vc_air = avg_y(cat_y((vf[AIR].bnd[S].val[:,:1,:], vf[AIR].val, vf[AIR].bnd[N].val[:,:1,:])))
vc_h2o = avg_y(cat_y((vf[H2O].bnd[S].val[:,:1,:], vf[H2O].val, vf[H2O].bnd[N].val[:,:1,:])))
vc_fil = avg_y(cat_y((vf[FIL].bnd[S].val[:,:1,:], vf[FIL].val, vf[FIL].bnd[N].val[:,:1,:])))

wc_air = avg_z(cat_z((wf[AIR].bnd[B].val[:,:,:1], wf[AIR].val, wf[AIR].bnd[T].val[:,:,:1])))
wc_h2o = avg_z(cat_z((wf[H2O].bnd[B].val[:,:,:1], wf[H2O].val, wf[H2O].bnd[T].val[:,:,:1])))
wc_fil = avg_z(cat_z((wf[FIL].bnd[B].val[:,:,:1], wf[FIL].val, wf[FIL].bnd[T].val[:,:,:1])))

velo_save_title = 'velocity_' + name + '_' + str(ts) + 'ts.npz'
np.savez(velo_save_title,uc_air,vc_air,wc_air,uc_h2o,vc_h2o,wc_h2o,uc_fil,vc_fil,wc_fil)
