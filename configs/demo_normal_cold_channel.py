
#!/usr/bin/python

import sys
sys.path.append("../..")

import time
import numpy as np
from scipy.constants import R

# Standard Python modules
from pyns.standard import *

# PyNS modules
from pyns.constants          import *
from pyns.operators          import *
from pyns.discretization     import *
from pyns.display            import plot, write
from pyns.physical           import properties
from pyns.physical.constants import G

plt.close("all")

#==========================================================================
#
# Define problem
#
#==========================================================================

# Domain order
AIR = 0
H2O = 1
FIL = 2
COL = 3

# Operational parameters
u_h_in = 0.6 # m/s
t_h_in = 70   # C
a_salt = 90.0 # g/l
t_c_in = 20   # C

# Time-stepping parameters
dt  =    0.0001   # time step
ndt =    5 #70000 # number of time steps
dt_plot = ndt     # plot frequency
dt_save = 5 #500  # save frequency for latest results in temp file 
dt_save_ts = 1000 # save frequency for results associated to time step
tss = 1           # per default, start at time step 1

# identifier for the simulation
name = 'N_coolant_' + str(t_h_in) + '_' + str(u_h_in).replace("0.", "")

# restart options
restart = False
restart_file = 'ws_' + name + '_temp.npz'

# start timer
time_start=time.time()

#-----------
# Define geometry
#----------- 

# Node coordinates for domains
xn = (nodes(0,   0.16, 128), nodes(0, 0.16,  128), nodes(0,       0.16, 128), nodes(0,       0.16, 128))
yn = (nodes(-0.0035, 0, 21), nodes(0, 0.0015,  9), nodes(-0.004, -0.0035, 3), nodes(-0.0055, -0.004, 9))
zn = (nodes(0,   0.1,   80), nodes(0, 0.1,    80), nodes(0,       0.1,   80), nodes(0,       0.1,   80))

# Cell coordinates 
xc = (avg(xn[AIR]), avg(xn[H2O]), avg(xn[FIL]), avg(xn[COL]))
yc = (avg(yn[AIR]), avg(yn[H2O]), avg(yn[FIL]), avg(yn[COL]))
zc = (avg(zn[AIR]), avg(zn[H2O]), avg(zn[FIL]), avg(zn[COL]))

# Cell dimensions
cell = [cartesian_grid(xn[AIR],yn[AIR],zn[AIR]),  \
        cartesian_grid(xn[H2O],yn[H2O],zn[H2O]),  \
        cartesian_grid(xn[FIL],yn[FIL],zn[FIL]),  \
        cartesian_grid(xn[COL],yn[COL],zn[COL])]

nx,ny,nz, dx,dy,dz = (cell[AIR][0], cell[H2O][0], cell[FIL][0], cell[COL][0]),  \
                     (cell[AIR][1], cell[H2O][1], cell[FIL][1], cell[COL][1]),  \
                     (cell[AIR][2], cell[H2O][2], cell[FIL][2], cell[COL][2]),  \
                     (cell[AIR][3], cell[H2O][3], cell[FIL][3], cell[COL][3]),  \
                     (cell[AIR][4], cell[H2O][4], cell[FIL][4], cell[COL][4]),  \
                     (cell[AIR][5], cell[H2O][5], cell[FIL][5], cell[COL][5])
rc,ru,rv,rw =        (cell[AIR][6], cell[H2O][6], cell[FIL][6], cell[COL][6]),  \
                     (cell[AIR][7], cell[H2O][7], cell[FIL][7], cell[COL][7]),  \
                     (cell[AIR][8], cell[H2O][8], cell[FIL][8], cell[COL][8]),  \
                     (cell[AIR][9], cell[H2O][9], cell[FIL][9], cell[COL][9])
					 
# Set physical properties temperature dependent:
prop = [properties.air(round((t_h_in+t_c_in)/2,-1),rc[AIR]), \
        properties.water(t_h_in,rc[H2O]),                    \
        properties.water(t_c_in,rc[FIL]),                    \
        properties.water(t_c_in,rc[COL])]

rho, mu, cap, kappa = (prop[AIR][0], prop[H2O][0], prop[FIL][0], prop[COL][0]),  \
                      (prop[AIR][1], prop[H2O][1], prop[FIL][1], prop[COL][1]),  \
                      (prop[AIR][2], prop[H2O][2], prop[FIL][2], prop[COL][2]),  \
                      (prop[AIR][3], prop[H2O][3], prop[FIL][3], prop[COL][3])
                      
diff = (ones(rc[AIR])*3.5E-5, ones(rc[H2O])*1.99E-09)

h_d = [ 0.0, properties.latent_heat(t_h_in), properties.latent_heat(t_c_in), \
       properties.latent_heat(t_c_in) ]

M_H2O  = 18E-3      # kg/mol
M_AIR  = 28E-3      # kg/mol
M_salt = 58.4428E-3 # kg/mol      

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
mem = membrane(110E-6,   \
                 0.19,    \
                 0.75,  \
                 1.5,      \
                 0.225E-6, \
                 zeros((nx[AIR],1,nz[AIR])), \
                 zeros((nx[AIR],1,nz[AIR])), \
                 zeros((nx[AIR],1,nz[AIR])), \
                 zeros((nx[AIR],1,nz[AIR])), \
                 zeros((nx[AIR],1,nz[AIR])), \
                 zeros((nx[AIR],1,nz[AIR])), \
                 zeros((nx[AIR],1,nz[AIR])))
                 
# stainless steel plate properties:
d_plate = 2.0E-3 # m
kappa_plate = 20 # W/(mK)                
      

#-----------
# Solving variables 
#-----------
 
# Create unknowns; names, positions, sizes and default boundary condition
uf    = [Unknown("face-u-vel",    X, ru[AIR], DIRICHLET),  \
         Unknown("face-u-vel",    X, ru[H2O], DIRICHLET),  \
         Unknown("face-u-vel",    X, ru[FIL], DIRICHLET),  \
         Unknown("face-u-vel",    X, ru[COL], DIRICHLET)]
vf    = [Unknown("face-v-vel",    Y, rv[AIR], DIRICHLET),  \
         Unknown("face-v-vel",    Y, rv[H2O], DIRICHLET),  \
         Unknown("face-v-vel",    Y, rv[FIL], DIRICHLET),  \
         Unknown("face-v-vel",    Y, rv[COL], DIRICHLET)]
wf    = [Unknown("face-w-vel",    Z, rw[AIR], DIRICHLET),  \
         Unknown("face-w-vel",    Z, rw[H2O], DIRICHLET),  \
         Unknown("face-w-vel",    Z, rw[FIL], DIRICHLET),  \
         Unknown("face-w-vel",    Z, rw[COL], DIRICHLET)]
p     = [Unknown("pressure",      C, rc[AIR], NEUMANN),  \
         Unknown("pressure",      C, rc[H2O], NEUMANN),  \
         Unknown("pressure",      C, rc[FIL], NEUMANN),  \
         Unknown("pressure",      C, rc[COL], NEUMANN)]
t     = [Unknown("temperature",   C, rc[AIR], NEUMANN),  \
         Unknown("temperature",   C, rc[H2O], NEUMANN),  \
         Unknown("temperature",   C, rc[FIL], NEUMANN),  \
         Unknown("temperature",   C, rc[COL], NEUMANN)]
a     = [Unknown("concentration", C, rc[AIR], NEUMANN),  \
         Unknown("concentration", C, rc[H2O], NEUMANN),  \
         Unknown("concentration", C, rc[FIL], NEUMANN),  \
         Unknown("concentration", C, rc[COL], NEUMANN)]
p_tot = [Unknown("pressure-tot",  C, rc[AIR], NEUMANN),  \
         Unknown("pressure-tot",  C, rc[H2O], NEUMANN),  \
         Unknown("pressure-tot",  C, rc[FIL], NEUMANN),  \
         Unknown("pressure-tot",  C, rc[COL], NEUMANN)]

# just for air
p_v =[Unknown("vapor_pressure",C, rc[AIR], NEUMANN)]
M  = [Unknown("molar mass",    C, rc[AIR], NEUMANN)]

# initialize source terms
q_t = [zeros(rc[AIR]),
       zeros(rc[H2O]), 
       zeros(rc[FIL]), 
       zeros(rc[COL])]
q_a = [zeros(rc[AIR]),
       zeros(rc[H2O])]
dv  = [dx[AIR]*dy[AIR]*dz[AIR], 
       dx[H2O]*dy[H2O]*dz[H2O], 
       dx[FIL]*dy[FIL]*dz[FIL], 
       dx[COL]*dy[COL]*dz[COL]]
 

#-----------
# Some helping variables 
#-----------
      
# variables to temporarily store vf bnd values:
# for salt concentration calculation without advection into air
vf_h2o_S_tmp=zeros(vf[H2O].bnd[S].val.shape)

# variable for salt boundary condition
x_tot = zeros(vf[H2O].bnd[S].val.shape)
x_salt = zeros(a[H2O].bnd[S].val.shape)
x_tot = zeros(a[H2O].bnd[S].val.shape)
a_2 = zeros(a[H2O].bnd[S].val.shape)

# initialize tracking of change
change_t = zeros(ndt)
change_a = zeros(ndt)
change_p = zeros(ndt)

#-----------
# Specify boundary condition 
#-----------

for k in range(0,nz[H2O]):
  uf[H2O].bnd[W].val[:1,:,k] = par(u_h_in, yn[H2O])
  uf[H2O].val[:,:,k] = par(u_h_in, yn[H2O])
  uf[COL].bnd[E].val[:1,:,k] = par(-u_h_in, yn[COL])
  uf[COL].val[:,:,k] = par(-u_h_in, yn[COL])
uf[H2O].bnd[E].typ[:1,:,:] = OUTLET 
uf[H2O].bnd[E].val[:1,:,:] = u_h_in
uf[COL].bnd[W].typ[:1,:,:] = OUTLET 
uf[COL].bnd[W].val[:1,:,:] = -u_h_in 
  
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

t[COL].bnd[E].typ[:1,:,:] = DIRICHLET
t[COL].bnd[E].val[:1,:,:] = t_c_in
t[COL].bnd[N].typ[:,:1,:] = DIRICHLET  
t[COL].bnd[N].val[:,:1,:] = t_c_in

a[H2O].bnd[W].typ[:1,:,:] = DIRICHLET
a[H2O].bnd[W].val[:1,:,:] = a_salt/rho[H2O][:1,:,:]

M[AIR].bnd[S].typ[:,:1,:] = DIRICHLET
M[AIR].bnd[S].val[:,:1,:] = M[AIR].val[:,:1,:]

p_v[AIR].bnd[S].typ[:,:,:] = DIRICHLET
p_v[AIR].bnd[S].val[:,:,:] = properties.p_v_sat(t[FIL].bnd[N].val[:,:,:])
p_v[AIR].bnd[N].typ[:,:,:] = DIRICHLET
p_v[AIR].bnd[N].val[:,:,:] = properties.p_v_sat(t[H2O].bnd[S].val[:,:,:])
p_v[AIR].bnd[S].typ[:,:,:] = DIRICHLET
p_v[AIR].bnd[S].val[:,:,:] = properties.p_v_sat(t[FIL].bnd[N].val[:,:,:])

t[AIR].val[:,:,:] = np.reshape(np.linspace(t_c_in,t_h_in, \
                  num=t[AIR].val.shape[1]),[1,t[AIR].val.shape[1],1])
t[H2O].val[:,:,:] = t_h_in
t[FIL].val[:,:,:] = t_c_in
t[COL].val[:,:,:] = t_c_in

mem.t_int[:,:,:] = t[H2O].bnd[S].val[:,:1,:]
t_int = t_c_in * ones(np.shape(mem.t_int))

a[AIR].val[:,:,:] = properties.p_v_sat(t[AIR].val[:,:,:])*1E-5*M_H2O/M_AIR
M[AIR].val[:,:,:] = 1/((1-a[AIR].val[:,:,:])/M_AIR + a[AIR].val[:,:,:]/M_H2O)
a[H2O].val[:,:,:] = a_salt/rho[H2O][:,:,:]

# correct Neumann boundary conditions 
for c in (AIR,H2O,FIL,COL):
  adj_n_bnds(p[c])
  adj_n_bnds(t[c])
  adj_n_bnds(a[c])
  
# max values in domain:
t_max = 0.0
t_min = 100.0

# check for extreme temperatures to set limits
for c in (W,E,S,N,T,B):
  t_max = np.amax([t_max, np.amax(t[H2O].bnd[c].val)])
  t_min = np.amin([t_min, np.amin(t[COL].bnd[c].val)]) 

#-----------
# Specify structures in the domains
#-----------
obst = [zeros(rc[AIR]), zeros(rc[H2O]),zeros(rc[FIL]),zeros(rc[COL])]


#%%
#-----------
# load previous data from file if restart is set
#-----------

if restart==True:
  
  data=np.load(restart_file)
    
  tss = data['arr_0']-1
  
  t[AIR].val[:,:,:] = data['arr_7']
  uf[AIR].val[:,:,:] = data['arr_8']
  vf[AIR].val[:,:,:] = data['arr_9']
  wf[AIR].val[:,:,:] = data['arr_10']
  p_tot[AIR].val[:,:,:] = data['arr_11']
  p[AIR].val[:,:,:] = data['arr_12']
  a[AIR].val[:,:,:] = data['arr_13']
  p_v[AIR].val[:,:,:] = data['arr_14']
  p_v[AIR].bnd[N].val[:,:,:] = data['arr_15']
  p_v[AIR].bnd[S].val[:,:,:] = data['arr_16']
  
  t[H2O].val[:,:,:] = data['arr_17']
  uf[H2O].val[:,:,:] = data['arr_18']
  vf[H2O].val[:,:,:] = data['arr_19']
  wf[H2O].val[:,:,:] = data['arr_20']
  p_tot[H2O].val[:,:,:] = data['arr_21']
  p[H2O].val[:,:,:] = data['arr_22']
  a[H2O].val[:,:,:] = data['arr_23']
  
  t[FIL].val[:,:,:] = data['arr_24']
  
  t[COL].val[:,:,:] = data['arr_25']
  uf[COL].val[:,:,:] = data['arr_26']
  vf[COL].val[:,:,:] = data['arr_27']
  wf[COL].val[:,:,:] = data['arr_28']
  p_tot[COL].val[:,:,:] = data['arr_29']
  p[COL].val[:,:,:] = data['arr_30']
  
  mem.t_int[:,:,:] = data['arr_31']
  mem.j[:,:,:] = data['arr_32']
  mem.pv[:,:,:] = data['arr_33']
  t_int = data['arr_34']
  m_evap = data['arr_35']
  
  print("data from file loaded")
  
  # correct Neumann boundary conditions
  for c in (AIR,H2O,FIL,COL):
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
  for c in (AIR,H2O,FIL,COL):
    a[c].old[:]  = a[c].val
    t[c].old[:]  = t[c].val
    p[c].old[:]  = p[c].val
    uf[c].old[:] = uf[c].val
    vf[c].old[:] = vf[c].val
    wf[c].old[:] = wf[c].val
  mem.t_old[:] = mem.t
  mem.t_int_old[:] = mem.t_int
  
  # interpolate temperature and salt dependent density                   
  rho[AIR][:,:,:] = np.interp(t[AIR].val, t_interp, rho_air)
  rho[H2O][:,:,:] = np.interp(t[H2O].val, t_interp, rho_water)
  rho[H2O][:,:,:] = properties.rho_salt(a[H2O].val[:,:,:],t[H2O].val[:,:,:],rho[H2O][:,:,:])
    
  #------------------------
  # Heat and Mass Transfer between Domains
  #------------------------  
  
  # AIR domain values: Partial vapor pressure & molar mass
  M[AIR].val[:,:,:] = 1/((1-a[AIR].val[:,:,:])/M_AIR + a[AIR].val[:,:,:]/M_H2O)  
  p_v[AIR].val[:,:,:] = a[AIR].val[:,:,:] *M[AIR].val[:,:,:]/M_H2O * (p_tot[AIR].val[:,:,:] +1E5) 
    
  # Interface energy equation between AIR & FIL
  t_int, m_evap, t, p_v = calc_interface(t, a, p_v, p_tot, kappa, M, \
                            (M_AIR, M_H2O, M_salt), h_d, (dx,dy,dz), (AIR, FIL), t_int)  
  
  # vapor source due to evaporation into AIR domain 
  q_a[AIR][:,:1,:]  = m_evap[:,:1,:] / dv[AIR][:,:1,:]
   
  # Membrane diffusion and energy equation between H2O & AIR
  mem, t, p_v = calc_membrane(t, a, p_v, p_tot, mem, kappa, diff, M, \
                    (M_AIR,M_H2O,M_salt), h_d, (dx,dy,dz), (AIR, H2O))
  
  # downward (negative) velocity induced through evaporation (positive mem_j)
  # makes simulation instable at start 
  #  -> only enable if ~1000 time steps have been solved                 
  #vf[H2O].bnd[S].val[:,:1,:] = -mem.j[:,:1,:]/(rho[H2O][:,:1,:]*dx[H2O][:,:1,:]*dz[H2O][:,:1,:])
  
  # vapor source due to evaporation into AIR domain 
  q_a[AIR][:,-1:,:] = mem.j [:,:1,:] / dv[AIR][:,-1:,:] 
  
  # salt concentration boundary condition
  x_tot[:,:1,:] = rho[H2O][:,:1,:] * dv[H2O][:,:1,:] # total mass of salt/water mix in cell
  x_salt[:,:1,:] = a[H2O].val[:,:1,:] * rho[H2O][:,:1,:] * dv[H2O][:,:1,:] # mass of salt
  a_2[:,:1,:] = x_salt / (x_tot - mem.j[:,:1,:]*dt) # salt concentration after evaporation
  q_a[H2O][:,:1,:] = rho[H2O][:,:1,:]/dt*(a_2[:,:1,:] - a[H2O].val[:,:1,:])
          
  # Heat transfer between FIL & COL d_plate=2mm, kappa_stainless steel=20W/(mK)
  tot_res_plate = dy[FIL][:,:1,:]/(2*kappa[FIL][:,:1,:]) + d_plate/kappa_plate \
                + dy[COL][:,-1:,:]/(2*kappa[COL][:,-1:,:])
  t[FIL].bnd[S].val[:,:1,:] = t[FIL].val[:,:1,:] \
                - dy[FIL][:,:1,:]/(2*kappa[FIL][:,:1,:] * tot_res_plate) \
                * (t[FIL].val[:,:1,:] - t[COL].val[:,-1:,:])          
  t[COL].bnd[N].val[:,:1,:] = t[COL].val[:,-1:,:] \
                + dy[COL][:,-1:,:]/(2*kappa[COL][:,-1:,:] * tot_res_plate) \
                * (t[FIL].val[:,:1,:] - t[COL].val[:,-1:,:])
         
  #------------------------
  # Concentration
  #------------------------
  
  # correct for salt retention in feed water
  vf_h2o_S_tmp[:,:,:] = vf[H2O].bnd[S].val[:,:1,:]
  vf[H2O].bnd[S].val[:,:1,:] = 0.0
  
  # discretize and solve concentration profile
  for c in (AIR,H2O):
    calc_t(a[c], (uf[c],vf[c],wf[c]), rho[c], diff[c],  \
           dt, (dx[c],dy[c],dz[c]), 
           obstacle = obst[c],
           source = q_a[c])
  
  # no negative concentrations       
  for c in (AIR,H2O):
    a[c].val[a[c].val < 0.0] = 0.0
    
  vf[H2O].bnd[S].val[:,:1,:] = vf_h2o_S_tmp[:,:,:]
    
  #------------------------
  # Temperature (enthalpy)
  #------------------------
  
  for c in (AIR,H2O,FIL,COL):
    calc_t(t[c], (uf[c],vf[c],wf[c]), (rho[c]*cap[c]), kappa[c],  \
           dt, (dx[c],dy[c],dz[c]), 
           obstacle = obst[c],
           source = q_t[c])

  for c in (AIR,H2O,FIL,COL):
    t[c].val[t[c].val > t_max] = t_max
    t[c].val[t[c].val < t_min] = t_min

  #-----------------------
  # Momentum conservation
  #-----------------------
  for c in (AIR,H2O,COL):

    # direction of gravity
    g_u = -G * avg(X, rho[c])
    #g_v = -G * avg(Y, rho[c])
  
    # external forces like gravity
    ef = g_u, zeros(rv[c]), zeros(rw[c])
    
    
    calc_uvw((uf[c],vf[c],wf[c]), (uf[c],vf[c],wf[c]), rho[c], mu[c],  \
             dt, (dx[c],dy[c],dz[c]), 
             obstacle = obst[c],
             pressure = p_tot[c],
             force = ef)
  
  #----------
  # Pressure
  #----------
  for c in (AIR,H2O,COL):
    calc_p(p[c], (uf[c],vf[c],wf[c]), rho[c],  \
           dt, (dx[c],dy[c],dz[c]), 
           obstacle = obst[c])
  
    p_tot[c].val = p_tot[c].val + p[c].val
  
  #---------------------
  # Velocity correction
  #---------------------
  for c in (AIR,H2O,COL):
    corr_uvw((uf[c],vf[c],wf[c]), p[c], rho[c],  \
             dt, (dx[c],dy[c],dz[c]), 
             obstacle = obst[c])

  # Check the CFL number too 
  for c in (AIR,H2O,COL):
    cfl = cfl_max((uf[c],vf[c],wf[c]), dt, (dx[c],dy[c],dz[c]))

  #---------------------
  # Saving of results
  #---------------------
    
  if ts % dt_save == 0:
      ws_tmp_name = 'ws_' + name + '_temp.npz'
      np.savez(ws_tmp_name, ts, xn, yn[AIR], yn[H2O], yn[FIL], yn[COL], zn, t[AIR].val, uf[AIR].val,vf[AIR].val,wf[AIR].val, p_tot[AIR].val, p[AIR].val, a[AIR].val,  p_v[AIR].val, p_v[AIR].bnd[N].val, p_v[AIR].bnd[S].val, t[H2O].val, uf[H2O].val,vf[H2O].val,wf[H2O].val,p_tot[H2O].val, p[H2O].val, a[H2O].val, t[FIL].val, t[COL].val, uf[COL].val, vf[COL].val, wf[COL].val, p_tot[COL].val, p[COL].val, mem.t_int, mem.j, mem.pv, t_int,m_evap )
      time_end = time.time()     
      print("Total time: %4.4e" % ((time_end-time_start)/3600))
      if ts % dt_save_ts ==0:
        ws_save_title = 'ws_' + name + '_' + str(ts) + 'ts.npz'
        np.savez(ws_save_title, ts, xn, yn[AIR], yn[H2O], yn[FIL], yn[COL], zn, t[AIR].val, uf[AIR].val,vf[AIR].val,wf[AIR].val, p_tot[AIR].val, p[AIR].val, a[AIR].val,  p_v[AIR].val, p_v[AIR].bnd[N].val, p_v[AIR].bnd[S].val, t[H2O].val, uf[H2O].val,vf[H2O].val,wf[H2O].val,p_tot[H2O].val, p[H2O].val, a[H2O].val, t[FIL].val, t[COL].val, uf[COL].val, vf[COL].val, wf[COL].val, p_tot[COL].val, p[COL].val, mem.t_int, mem.j, mem.pv, t_int,m_evap )
        text_id = 'Output_' + name + '_' + str(ts) + '.txt'
        text_file = open(text_id, "w")
        airgap_outfile = 0.0035
        massflow_outfile = np.sum(m_evap) \
                     /np.sum(dx[AIR][:,-1:,:]*dz[AIR][:,-1:,:])*3600 
        RR_outfile = (-np.sum(m_evap))/(u_h_in*np.mean(rho[H2O][:1,:,:])\
                      *np.sum(dy[H2O][:1,:,:]*dz[H2O][:1,:,:]))
        dT_H2O = t_h_in - np.mean(t[H2O].val[-1:,:,:])
        GOR_outfile = RR_outfile * h_d[H2O]/(np.mean(cap[H2O][:1,:,:])*dT_H2O)
        text_file.write("t_h_in u_h_in airgap m_evap RR GOR\n")
        text_file.write("{0:2.0f} {1:1.3f} {2:1.4f} {3:2.3e} {4:2.4e} {5:2.4e}".format \
          (t_h_in, u_h_in, airgap_outfile, massflow_outfile, RR_outfile, GOR_outfile))
        text_file.close()
    
  # Check relative change in domain:
  change_t[ts-1] = (np.absolute(t[H2O].val-t[H2O].old)).max()/t[H2O].old.max()
  change_t[ts-1] = max(change_t[ts-1],(np.absolute(t[AIR].val-t[AIR].old)).max()/t[AIR].old.max())
  change_t[ts-1] = max(change_t[ts-1],(np.absolute(t[FIL].val-t[FIL].old)).max()/t[FIL].old.max())
  change_t[ts-1] = max(change_t[ts-1],(np.absolute(t[COL].val-t[COL].old)).max()/t[COL].old.max())  
  
  change_a[ts-1] = (np.absolute(a[H2O].val-a[H2O].old)).max()/a[H2O].old.max()
  change_a[ts-1] = max(change_a[ts-1],(np.absolute(a[AIR].val-a[AIR].old)).max()/a[AIR].old.max())
  if p[H2O].old.max() > 0.0: 
    change_p[ts-1] = (np.absolute(p[H2O].val-p[H2O].old)).max()/p[H2O].old.max()
    

#==========================================================================
#
# Visualisation
#
#========================================================================== 

#%%save velocity profil at the end of the simulation
    
  if ts % dt_plot == 0:

    z_pos = 40
    
    uc_air = avg_x(cat_x((uf[AIR].bnd[W].val[:1,:,:], uf[AIR].val, uf[AIR].bnd[E].val[:1,:,:])))
    uc_h2o = avg_x(cat_x((uf[H2O].bnd[W].val[:1,:,:], uf[H2O].val, uf[H2O].bnd[E].val[:1,:,:])))
    uc_fil = avg_x(cat_x((uf[FIL].bnd[W].val[:1,:,:], uf[FIL].val, uf[FIL].bnd[E].val[:1,:,:])))
    uc_col = avg_x(cat_x((uf[COL].bnd[W].val[:1,:,:], uf[COL].val, uf[COL].bnd[E].val[:1,:,:])))
    
    
    vc_air = avg_y(cat_y((vf[AIR].bnd[S].val[:,:1,:], vf[AIR].val, vf[AIR].bnd[N].val[:,:1,:])))
    vc_h2o = avg_y(cat_y((vf[H2O].bnd[S].val[:,:1,:], vf[H2O].val, vf[H2O].bnd[N].val[:,:1,:])))
    vc_fil = avg_y(cat_y((vf[FIL].bnd[S].val[:,:1,:], vf[FIL].val, vf[FIL].bnd[N].val[:,:1,:])))
    vc_col = avg_y(cat_y((vf[COL].bnd[S].val[:,:1,:], vf[COL].val, vf[COL].bnd[N].val[:,:1,:])))
    
    
    wc_air = avg_z(cat_z((wf[AIR].bnd[B].val[:,:,:1], wf[AIR].val, wf[AIR].bnd[T].val[:,:,:1])))
    wc_h2o = avg_z(cat_z((wf[H2O].bnd[B].val[:,:,:1], wf[H2O].val, wf[H2O].bnd[T].val[:,:,:1])))
    wc_fil = avg_z(cat_z((wf[FIL].bnd[B].val[:,:,:1], wf[FIL].val, wf[FIL].bnd[T].val[:,:,:1])))
    wc_col = avg_z(cat_z((wf[COL].bnd[B].val[:,:,:1], wf[COL].val, wf[COL].bnd[T].val[:,:,:1])))
    
    velo_save_title = 'velocity_' + name + '_' + str(ts) + 'ts.npz'
    np.savez(velo_save_title,uc_air,vc_air,wc_air,uc_h2o,vc_h2o,wc_h2o,uc_fil,vc_fil,wc_fil,uc_col,vc_col,wc_col)
    
    #%%
    plt.ion()
    
    plt.close("all")
    
    z_pos = 10
    
    xc = avg(xn[AIR])
    yc = np.append(avg(yn[COL]), avg(yn[FIL]),axis=0)
    yc = np.append(yc, avg(yn[AIR]),axis=0)
    yc = np.append(yc, avg(yn[H2O]),axis=0)
    
    t_plot=np.append(t[COL].val[:,:,z_pos],t[FIL].val[:,:,z_pos],axis=1)
    t_plot=np.append(t_plot, t[AIR].val[:,:,z_pos],axis=1)    
    t_plot=np.append(t_plot, t[H2O].val[:,:,z_pos],axis=1)
    t_plot=transpose(t_plot)
    p_plot=np.append(p[COL].val[:,:,z_pos],p[FIL].val[:,:,z_pos],axis=1)
    p_plot=np.append(p_plot, p[AIR].val[:,:,z_pos],axis=1)
    p_plot=np.append(p_plot, p[H2O].val[:,:,z_pos],axis=1)
    p_plot=transpose(p_plot)
    a_plot=np.append(a[COL].val[:,:,z_pos],a[FIL].val[:,:,z_pos],axis=1)
    a_plot=np.append(a_plot, a[AIR].val[:,:,z_pos],axis=1)
    a_plot=np.append(a_plot, a[H2O].val[:,:,z_pos],axis=1)
    a_plot=transpose(a_plot)
    
    uc_air = avg_x(cat_x((uf[AIR].bnd[W].val[:1,:,:], uf[AIR].val, uf[AIR].bnd[E].val[:1,:,:])))
    uc_h2o = avg_x(cat_x((uf[H2O].bnd[W].val[:1,:,:], uf[H2O].val, uf[H2O].bnd[E].val[:1,:,:])))
    uc_fil = avg_x(cat_x((uf[FIL].bnd[W].val[:1,:,:], uf[FIL].val, uf[FIL].bnd[E].val[:1,:,:])))
    uc_col = avg_x(cat_x((uf[COL].bnd[W].val[:1,:,:], uf[COL].val, uf[COL].bnd[E].val[:1,:,:])))
    u_plot = np.concatenate([uc_col[:,:,z_pos],uc_fil[:,:,z_pos],uc_air[:,:,z_pos],uc_h2o[:,:,z_pos]],axis=1)
    u_plot = transpose(u_plot)    
    
    plt.figure
    plt.subplot(2,2,1)
    levels_t=linspace( t_plot.min(), t_plot.max(), 11)
    norm_t=cm.colors.Normalize( vmax=t_plot.max(), vmin=t_plot.min() )
    cax_t=plt.contourf(xc,yc,t_plot,levels_t,cmap="rainbow",norm=norm_t)
    cbar_t=plt.colorbar(cax_t)
    plt.title("Temperature")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]" )
    
    plt.subplot(2,2,2)
    matplotlib.rcParams["contour.negative_linestyle"] = "solid"
    cax_p=plt.contourf(xc,yc,p_plot,cmap="rainbow")
    cax_p2=plt.contour(xc,yc,p_plot,colors="k")
    plt.clabel(cax_p2, fontsize=12, inline=1)
    cbar_p = plt.colorbar(cax_p)
    plt.title("Pressure Correction")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]" )
    
    plt.subplot(2,2,3)
    cax_a=plt.contourf(xc,yc,a_plot,cmap="rainbow")
    cbar_a=plt.colorbar(cax_a)
    plt.title("Concentration")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]" )
    
    plt.subplot(2,2,4)
    cax_u=plt.contourf(xc,yc,u_plot,cmap="rainbow")
    cbar_u=plt.colorbar(cax_u)
    plt.title("Axial Velocity")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]" )
     
time_end = time.time()     
print("Total time: %4.4e h" % ((time_end-time_start)/3600))
