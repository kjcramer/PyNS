"""
Calculates interface temperature from partial vapor pressure and mass flux.
"""

# Standard Python modules
from pyns.standard import *

# PyNS modules
from pyns.constants import *
from pyns.operators import *
from pyns.display   import write

import numpy as np
from pyns.demo.p_v_sat import *
from scipy.optimize import fsolve

# =============================================================================
def calc_interface(t, a, p_v, p_tot, kappa, M, M_AIR, M_H2O, h_d, dxyz, dom):
# -----------------------------------------------------------------------------
    """
    Args:
      t: ...... Object of the type "Unknown", holding the temperature.
      a: ...... Object of the type "Unknown", holding the concentration.
      p_v: .... Object of the type "Unknown", holding the partial vapour pressure in air.
      p_tot: .. Object holding the total pressure.
      kappa: .. Three-dimensional array holding heat conductivity for all cells.
      M: ...... Three-dimensional array holding molar mass for all air cells
      M_AIR/H2O:Molar mass of pure air / water 
      h_d: .... Latent heat of evaporation array
      dxyz: ... Tuple holding cell dimensions in "x", "y" and "z" directions.
                Each cell dimension is a three-dimensional array.
      dom: .... Array of bordering domains, [vapor liquid]

    Returns:
      m_evap: . Evaporation mass flux, if negative then condensation
      t_int: .. Array of interface temperature
      t: ...... Object of the type "Unknown", holding the temperature.
      p_v: .... Object of the type "Unknown", holding the partial vapour pressure in air.
    """
    
    # define identifier for liquid & vapor phase
    VAP = 0
    LIQ = 1   
    
    # Unpack tuple(s)
    dx, dy, dz = dxyz 
    
    # Update boundary values of air 
    M[dom[VAP]].bnd[S].val[:,:,:] = np.power(((1-a[dom[VAP]].bnd[S].val[:,:,:])/M_AIR \
                                  + a[dom[VAP]].bnd[S].val[:,:,:]/M_H2O),(-1))
                                  
    p_v[dom[VAP]].bnd[S].val[:,:,:] = a[dom[VAP]].bnd[S].val[:,:,:]  \
                                  *M[dom[VAP]].bnd[S].val[:,:,:]/M_H2O *  \
                                  (p_tot[dom[VAP]].val[:,:1,:] +1E5)
    
    # calculate interface temperature for saturated air 
    t_int = t_sat(p_v[dom[VAP]].bnd[S].val[:,:,:])
    print("t_int = " + "%3.4f" %np.mean(t_int))
    
    # update temperature boundary values
    t[dom[VAP]].bnd[S].val[:,:,:] = t_int
    t[dom[LIQ]].bnd[N].val[:,:,:] = t_int
    
    # calculate phase change mass flux: positive for evaporation, negative for condensation
    m_evap = (2* kappa[dom[VAP]][:,:1,:] / dy[dom[VAP]][:,:1,:] *    \
                     (t[dom[VAP]].val[:,:1,:] - t_int)    \
            + 2*kappa[dom[LIQ]][:,-1:,:] / dy[dom[LIQ]][:,-1:,:] *  \
                    (t[dom[LIQ]].val[:,-1:,:] - t_int))   \
            * dx[dom[VAP]][:,:1,:] * dz[dom[VAP]][:,:1,:] / h_d[dom[LIQ]]  # kg/s
    
    print("m_evap = " + "%3.4e" %np.mean(m_evap)) 

    return t_int, m_evap, t, p_v # end of function
  
  
# =============================================================================
def calc_interface2(t, a, p_v, p_tot, kappa, M, M_AIR, M_H2O, h_d, dxyz, dom, t_int_old):
# -----------------------------------------------------------------------------
    """
    Args:
      t: ...... Object of the type "Unknown", holding the temperature.
      a: ...... Object of the type "Unknown", holding the concentration.
      p_v: .... Object of the type "Unknown", holding the partial vapour pressure in air.
      p_tot: .. Object holding the total pressure.
      kappa: .. Three-dimensional array holding heat conductivity for all cells.
      M: ...... Three-dimensional array holding molar mass for all air cells
      M_AIR/H2O:Molar mass of pure air / water 
      h_d: .... Latent heat of evaporation array
      dxyz: ... Tuple holding cell dimensions in "x", "y" and "z" directions.
                Each cell dimension is a three-dimensional array.
      dom: .... Array of bordering domains, [vapor liquid]

    Returns:
      m_evap: . Evaporation mass flux, if negative then condensation
      t_int: .. Array of interface temperature
      t: ...... Object of the type "Unknown", holding the temperature.
      p_v: .... Object of the type "Unknown", holding the partial vapour pressure in air.
    """
    
    # define identifier for liquid & vapor phase
    VAP = 0
    LIQ = 1   
    
    # Unpack tuple(s)
    dx, dy, dz = dxyz 
    
    # Update boundary values of air 
    M[dom[VAP]].bnd[S].val[:,:,:] = np.power(((1-a[dom[VAP]].bnd[S].val[:,:,:])/M_AIR \
                                  + a[dom[VAP]].bnd[S].val[:,:,:]/M_H2O),(-1))
                                  
    p_v[dom[VAP]].bnd[S].val[:,:,:] = a[dom[VAP]].bnd[S].val[:,:,:]  \
                                  *M[dom[VAP]].bnd[S].val[:,:,:]/M_H2O *  \
                                  (p_tot[dom[VAP]].val[:,:1,:] +1E5)
    
    # calculate interface temperature for saturated air 
    t_int = t_int_old *0.7 + 0.3 * t_sat(p_v[dom[VAP]].bnd[S].val[:,:,:])
    print("t_int = " + "%3.4f" %np.mean(t_int))
    
    # update temperature boundary values
    t[dom[VAP]].bnd[S].val[:,:,:] = t_int
    t[dom[LIQ]].bnd[N].val[:,:,:] = t_int
    
    # calculate phase change mass flux: positive for evaporation, negative for condensation
    m_evap = (2* kappa[dom[VAP]][:,:1,:] / dy[dom[VAP]][:,:1,:] *    \
                     (t[dom[VAP]].val[:,:1,:] - t_int)    \
            + 2*kappa[dom[LIQ]][:,-1:,:] / dy[dom[LIQ]][:,-1:,:] *  \
                    (t[dom[LIQ]].val[:,-1:,:] - t_int))   \
            * dx[dom[VAP]][:,:1,:] * dz[dom[VAP]][:,:1,:] / h_d[dom[LIQ]]  # kg/s
    
    print("m_evap = " + "%3.4e" %np.mean(m_evap)) 

    return t_int, m_evap, t, p_v # end of function
    
# =============================================================================
def calc_membrane(t, a, p_v, p_tot, mem, kappa, diff, M, M_input, h_d, dxyz, dom):
# -----------------------------------------------------------------------------
    """
    Args:
      t: ...... Object of the type "Unknown", holding the temperature.
      a: ...... Object of the type "Unknown", holding the concentration.
      p_v: .... Object of the type "Unknown", holding the partial vapour pressure in air.
      p_tot: .. Object holding the total pressure.
      mem: .... Tuple holding membrane realted variables and parameters
      kappa: .. Three-dimensional array holding heat conductivity for all cells.
      diff: ... Three-dimensional array holding the diffusivity for all cells.
      M: ...... Three-dimensional array holding molar mass for all air cells
      M_input:  Tuple holding molar masses of air, water and salt
      h_d: .... Latent heat of evaporation array
      dxyz: ... Tuple holding cell dimensions in "x", "y" and "z" directions.
                Each cell dimension is a three-dimensional array.
      dom: .... Array of bordering domains, [vapor liquid]

    Returns:
      m_evap: . Evaporation mass flux, if negative then condensation
      t_int: .. Array of interface temperature
      t: ...... Object of the type "Unknown", holding the temperature.
      p_v: .... Object of the type "Unknown", holding the partial vapour pressure in air.
    """
    
    # define identifier for liquid & vapor phase
    VAP = 0
    LIQ = 1   
    
    R = 8.314
    pi = 3.1415
    
    # Unpack tuple(s)
    dx, dy, dz = dxyz 
    M_AIR, M_H2O, M_salt = M_input

    kappa_mem = mem.kap*(1-mem.eps) + kappa[dom[VAP]][:,-1:,:]*mem.eps
    
    # Compute new fluid variables in the membrane
    mem.t[:,:1,:] = mem.t + 273.15;
    
    mem.p[:,:1,:] = (p_tot[dom[LIQ]].val[:,:1,:] \
                   + p_tot[dom[VAP]].val[:,-1:,:]) /2.0 + 1E5
                   
    mem.pv[:,:1,:] = (p_v[dom[VAP]].bnd[N].val[:,:1,:] \
                    + p_v[dom[VAP]].val[:,-1:,:]) /2.0
    
    # Diffusion Coefficients
    C_K = 2.0*mem.eps*mem.r/(3.0*mem.tau*mem.d)  \
              *np.power(8.0*M_H2O/(mem.t*R*pi),0.5)
              
    C_M = mem.eps*mem.p*diff[dom[VAP]][:,-1:,:]  \
         /(mem.d*mem.tau*R*mem.t*(mem.p-mem.pv))
         
    C_T = 1.0/(1.0/C_K + 1.0/C_M)
    
    # Jump condition at membrane -> calculation of membrane interface temperature
    lhs_lin_mem = (2.0*kappa[dom[LIQ]][:,:1,:]/dy[dom[LIQ]][:,:1,:] \
       + 1.0/(dy[dom[VAP]][:,-1:,:]/(2.0*kappa[dom[VAP]][:,-1:,:]) + mem.d/kappa_mem)) \
       * mem.eps * dx[dom[VAP]][:,-1:,:] * dz[dom[VAP]][:,-1:,:] / h_d[dom[LIQ]]
       
    lhs_fun_mem = C_T*dx[dom[VAP]][:,-1:,:]*dz[dom[VAP]][:,-1:,:]
    
    rhs_mem = C_T*dx[dom[VAP]][:,-1:,:]*dz[dom[VAP]][:,-1:,:]*p_v[dom[VAP]].val[:,-1:,:] \
      + mem.eps*dx[dom[VAP]][:,-1:,:]*dz[dom[VAP]][:,-1:,:]/h_d[dom[LIQ]] \
      * ( 2.0*kappa[dom[LIQ]][:,:1,:]/dy[dom[LIQ]][:,:1,:]*t[dom[LIQ]].val[:,:1,:] \
        + 1.0/(dy[dom[VAP]][:,-1:,:]/(2.0*kappa[dom[VAP]][:,-1:,:])+mem.d/kappa_mem) \
        * t[dom[VAP]].val[:,-1:,:])
        
    [nx,ny,nz] = t[dom[VAP]].val.shape
    
    for ii in range(0,nx):
      for kk in range(0,nz):
        jump_cond_mem = lambda t: lhs_lin_mem[ii,:1,kk]*t + lhs_fun_mem[ii,:1,kk] \
          * p_v_sat_salt(t, a[dom[LIQ]].val[ii,:1,kk], M_salt, M_H2O) - rhs_mem[ii,:1,kk]
        mem.t_int[ii,:1,kk] = fsolve(jump_cond_mem, t[dom[LIQ]].val[ii,:1,kk])
    
    print("mem.t_int = " + "%3.4f" %np.mean(mem.t_int))
    
    
    # update boundary conditions & membrane flux
    # Liquid domain boundary condition
    t[dom[LIQ]].bnd[S].val[:,:1,:] = mem.t_int
                  
    # Vapor domain boundary condition    
    const_mem_2 = 2*kappa[dom[VAP]][:,-1:,:]*mem.d  \
                  /kappa_mem/dy[dom[VAP]][:,-1:,:];
              
    t[dom[VAP]].bnd[N].val[:,:1,:] = (mem.t_int + const_mem_2 \
                                    *t[dom[VAP]].val[:,-1:,:])/(1+const_mem_2)
    
    # membrane temperature                                
    mem.t[:,:1,:] = (t[dom[VAP]].bnd[N].val[:,:1,:] + mem.t_int)/2.0
    
    # saturated vapor pressure at liquid domain boundary                             
    p_v[dom[VAP]].bnd[N].val[:,:,:]= p_v_sat_salt(mem.t_int, \
                                    a[dom[LIQ]].val[:,:1,:], M_salt, M_H2O)
    
    # membrane vapor flux                            
    mem.j[:,:,:] = C_T[:,:,:] *dx[dom[VAP]][:,-1:,:]*dz[dom[VAP]][:,-1:,:]  \
                   *(p_v[dom[VAP]].bnd[N].val[:,:,:]-p_v[dom[VAP]].val[:,-1:,:])

    return mem, t, p_v # end of function