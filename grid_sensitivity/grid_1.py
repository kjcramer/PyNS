
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
COL = 3

u_h_in = 0.2 # m/s
t_h_in = 70   # C
a_salt = 90.0 # g/l
t_c_in = 20   # C

# Node coordinates for both domains
xn = (nodes(0,   0.16, 240), nodes(0, 0.16,  240), nodes(0,       0.16, 240), nodes(0,       0.16, 240))
yn = (nodes(-0.0035, 0, 21), nodes(0, 0.0015, 9), nodes(-0.004, -0.0035, 3), nodes(-0.0055, -0.004, 9))
zn = (nodes(0,   0.1,  150), nodes(0, 0.1,   150), nodes(0,       0.1,  150), nodes(0,       0.1,  150))

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

# Set physical properties					 
#prop = [properties.air(round((t_h_in+t_c_in)/2,-1),rc[AIR]),    \
#        properties.water(t_h_in,rc[H2O]),  \
#        properties.water(t_c_in,rc[FIL])]
					 
# Set physical properties temperature dependent:
prop = [properties.air(round((t_h_in+t_c_in)/2,-1),rc[AIR]), \
        properties.water(t_h_in,rc[H2O]),                    \
        properties.water(t_c_in,rc[FIL]),                    \
        properties.water(t_c_in,rc[COL])]

rho, mu, cap, kappa = (prop[AIR][0], prop[H2O][0], prop[FIL][0], prop[COL][0]),  \
                      (prop[AIR][1], prop[H2O][1], prop[FIL][1], prop[COL][1]),  \
                      (prop[AIR][2], prop[H2O][2], prop[FIL][2], prop[COL][2]),  \
                      (prop[AIR][3], prop[H2O][3], prop[FIL][3], prop[COL][3])
                      
diff = (ones(rc[AIR])*4.0E-4, ones(rc[H2O])*1.99E-09)

h_d = [ 0.0, latent_heat(t_h_in), latent_heat(t_c_in), latent_heat(t_c_in) ]
#h_d = [ 0.0, 2359E3, 2359E3]

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
kappa_plate = 16 # W/(mK)                
      
 
# Create unknowns; names, positions and sizes
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
       
# variables to temporarily store vf bnd values:
vf_air_S_tmp=zeros(vf[AIR].bnd[S].val.shape)
vf_h2o_S_tmp=zeros(vf[H2O].bnd[S].val.shape)

# Specify boundary conditions

for k in range(0,nz[H2O]):
  uf[H2O].bnd[W].val[:1,:,k] = par(u_h_in, yn[H2O])
  uf[H2O].val[:,:,k] = par(u_h_in, yn[H2O])
  uf[COL].bnd[E].val[:1,:,k] = par(-u_h_in, yn[COL])
  uf[COL].val[:,:,k] = par(-u_h_in, yn[COL])
uf[H2O].bnd[E].typ[:1,:,:] = OUTLET 
uf[H2O].bnd[E].val[:1,:,:] = u_h_in
uf[COL].bnd[W].typ[:1,:,:] = OUTLET 
uf[COL].bnd[W].val[:1,:,:] = -u_h_in
#uf[AIR].bnd[W].typ[:1,:,:] = OUTLET

for c in (AIR,H2O,COL):
  for j in (B,T):
    uf[c].bnd[j].typ[:] = NEUMANN     
    vf[c].bnd[j].typ[:] = NEUMANN     
    wf[c].bnd[j].typ[:] = NEUMANN     
  
t[AIR].bnd[S].typ[:,:1,:] = DIRICHLET  
t[AIR].bnd[S].val[:,:1,:] = t_c_in
t[AIR].bnd[N].typ[:,:1,:] = DIRICHLET  
t[AIR].bnd[N].val[:,:1,:] = 60

t[H2O].bnd[W].typ[:1,:,:] = DIRICHLET
t[H2O].bnd[W].val[:1,:,:] = t_h_in
t[H2O].bnd[S].typ[:,:1,:] = DIRICHLET  
t[H2O].bnd[S].val[:,:1,:] = 60
 
t[FIL].bnd[S].typ[:,:1,:] = DIRICHLET
t[FIL].bnd[S].val[:,:1,:] = t_c_in
t[FIL].bnd[N].typ[:,:1,:] = DIRICHLET  
t[FIL].bnd[N].val[:,:1,:] = t_c_in

t[COL].bnd[E].typ[:1,:,:] = DIRICHLET
t[COL].bnd[E].val[:1,:,:] = t_c_in
t[COL].bnd[N].typ[:,:1,:] = DIRICHLET  
t[COL].bnd[N].val[:,:1,:] = t_c_in

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
t[H2O].val[:,:,:] = 70
t[FIL].val[:,:,:] = t_c_in
t[COL].val[:,:,:] = t_c_in

a[AIR].val[:,:,:] = p_v_sat(t[AIR].val[:,:,:])*1E-5*M_H2O/M_AIR
M[AIR].val[:,:,:] = 1/((1-a[AIR].val[:,:,:])/M_AIR + a[AIR].val[:,:,:]/M_H2O)
a[H2O].val[:,:,:] = a_salt/rho[H2O][:,:,:]
 
for c in (AIR,H2O,FIL,COL):
  adj_n_bnds(p[c])
  adj_n_bnds(t[c])
  adj_n_bnds(a[c])
  
# max values in domain:
t_max = 0.0
t_min = 100.0

for c in (W,T):
  t_max = np.amax([t_max, np.amax(t[H2O].bnd[c].val)])
  t_min = np.amin([t_min, np.amin(t[COL].bnd[c].val)]) 
  
  # Time-stepping parameters
dt  =    0.0002  # time step
ndt =    1  # number of time steps
dt_plot = ndt    # plot frequency
dt_save = 100
dt_save_ts = 50

obst = [zeros(rc[AIR]), zeros(rc[H2O]),zeros(rc[FIL]),zeros(rc[COL])]

# initialize tracking of change
change_t = zeros(ndt)
change_a = zeros(ndt)
change_p = zeros(ndt)

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
for ts in range(1,ndt+1):
  
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
  
#  #calculate rho
#  t_min_rho = 20
#  t_max_rho = 80
#  rho_min = 1.205
#  rho_max = 1.000
#  rho[AIR][:,:,:] = (t[AIR].val[:,:,:] - t_min_rho)/(t_max_rho - t_min_rho) * \
#                     (rho_max - rho_min) + rho_min
                     
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
  #vf[FIL].bnd[N].val[:,:1,:] = m_evap[:,:1,:]/(rho[FIL][:,-1:,:]*dx[FIL][:,-1:,:]*dz[FIL][:,-1:,:]) 
  #vf[AIR].bnd[S].val[:,:1,:] = m_evap[:,:1,:]/(rho[AIR][:,:1,:]*dx[AIR][:,:1,:]*dz[AIR][:,:1,:])
   
  # Membrane diffusion and energy equation between H2O & AIR
  mem, t, p_v = calc_membrane(t, a, p_v, p_tot, mem, kappa, diff, M, \
                    (M_AIR,M_H2O,M_salt), h_d, (dx,dy,dz), (AIR, H2O))
  
  # downward (negative) velocity induced through evaporation (positive mem_j)                
  vf[H2O].bnd[S].val[:,:1,:] = -mem.j[:,:1,:]/(rho[H2O][:,:1,:]*dx[H2O][:,:1,:]*dz[H2O][:,:1,:]) 
  #vf[AIR].bnd[N].val[:,:1,:] = -mem.j[:,:1,:]/(rho[AIR][:,-1:,:]*dx[AIR][:,-1:,:]*dz[AIR][:,-1:,:])
  q_a[AIR][:,-1:,:] = mem.j [:,:1,:] / dv[AIR][:,-1:,:] 
          
  # Heat transfer between FIL & COL d_plate=2mm, kappa_stainless steel=20W/(mK)
  tot_res_plate = dy[FIL][:,:1,:]/(2*kappa[FIL][:,:1,:]) + d_plate/kappa_plate \
                + dy[COL][:,-1:,:]/(2*kappa[COL][:,-1:,:])
  t[FIL].bnd[S].val[:,:1,:] = t[FIL].val[:,:1,:] \
                - dy[FIL][:,:1,:]/(2*kappa[FIL][:,:1,:] + tot_res_plate) \
                * (t[FIL].val[:,:1,:] - t[COL].val[:,-1:,:])          
  t[COL].bnd[N].val[:,:1,:] = t[COL].val[:,-1:,:] \
                + dy[COL][:,-1:,:]/(2*kappa[COL][:,-1:,:] + tot_res_plate) \
                * (t[FIL].val[:,:1,:] - t[COL].val[:,-1:,:])
         
  #------------------------
  # Concentration
  #------------------------
  
  # correct for salt retention in feed water
  vf_h2o_S_tmp[:,:,:] = vf[H2O].bnd[S].val[:,:1,:]
  vf[H2O].bnd[S].val[:,:1,:] = 0.0
  
  vf_air_S_tmp[:,:,:] = vf[AIR].bnd[S].val[:,:1,:]
  vf[AIR].bnd[S].val[:,:1,:] = 0.0
  
  # in case of v[H2O].bnd[S].val ~= 0 correct convection into membrane 
  for c in (AIR,H2O):
    calc_t(a[c], (uf[c],vf[c],wf[c]), rho[c], diff[c],  \
           dt, (dx[c],dy[c],dz[c]), 
           obstacle = obst[c],
           source = q_a[c])
           
  for c in (AIR,H2O):
    a[c].val[a[c].val < 0.0] = 0.0
    
  vf[H2O].bnd[S].val[:,:1,:] = vf_h2o_S_tmp[:,:,:]
  vf[AIR].bnd[S].val[:,:1,:] = vf_air_S_tmp[:,:,:]
    
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
    g_u = -G * avg(X, rho[c])
  
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
 
#  # Compute volume balance for checking 
#  for c in (AIR,H2O,COL):
#    err = vol_balance((uf[c],vf[c],wf[c]),  \
#                      (dx[c],dy[c],dz[c]), 
#                      obstacle = obst[c])
#    print("Maximum volume error after correction: %12.5e" % abs(err).max())

  # Check the CFL number too 
  for c in (AIR,H2O,COL):
    cfl = cfl_max((uf[c],vf[c],wf[c]), dt, (dx[c],dy[c],dz[c]))
    # print("Maximum CFL number: %12.5e" % cfl)
    
  if ts % dt_save == 0:
      np.savez('ws_temp.npz', ts, t[AIR].val, t[H2O].val, t[FIL].val,uf[H2O].val,vf[H2O].val,wf[H2O].val,a[H2O].val,a[AIR].val,p[H2O].val,mem.t_int, t_int,m_evap, mem.j, mem.pv,p_v[AIR].val, p_v[AIR].bnd[N].val, p_v[AIR].bnd[S].val, uf[AIR].val,vf[AIR].val,wf[AIR].val, xn, yn[AIR], yn[H2O], yn[FIL], zn   )
      if ts % dt_save_ts ==0:
        ws_save_title = 'ws_' + str(ts) + 'ts.npz'
        np.savez(ws_save_title, ts, t[AIR].val, t[H2O].val, t[FIL].val,uf[H2O].val,vf[H2O].val,wf[H2O].val,a[H2O].val,a[AIR].val,p[H2O].val,mem.t_int, t_int,m_evap, mem.j, mem.pv,p_v[AIR].val, p_v[AIR].bnd[N].val, p_v[AIR].bnd[S].val, uf[AIR].val,vf[AIR].val,wf[AIR].val, xn, yn[AIR], yn[H2O], yn[FIL], zn   )
      
    
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

#%%
  if ts % dt_plot == 0:
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
