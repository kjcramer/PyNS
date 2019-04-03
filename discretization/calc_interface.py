"""
Calculates mass flux and interface temperature from partial vapor pressure
at the "free" interface.
Again, positive m_evap means evaporation 
"""

import numpy as np

# Standard Python modules
from pyns.standard import *

# PyNS modules
from pyns.constants import *
from pyns.operators import *
from pyns.display   import write
from pyns.physical  import properties
  
# =============================================================================
def calc_interface(t, a, p_v, p_tot, kappa, M, M_input, h_d, dxyz, dom, t_int_old, alpha=1):
# -----------------------------------------------------------------------------
    """
    Args:
      t: ...... Object of the type "Unknown", holding the temperature.
      a: ...... Object of the type "Unknown", holding the concentration.
      p_v: .... Object of the type "Unknown", holding the partial vapour pressure in air.
      p_tot: .. Object holding the total pressure.
      kappa: .. Three-dimensional array holding heat conductivity for all cells.
      M: ...... Three-dimensional array holding molar mass for all air cells
      M_input:  Tuple holding molar masses of air, water and salt
      h_d: .... Latent heat of evaporation array
      dxyz: ... Tuple holding cell dimensions in "x", "y" and "z" directions.
                Each cell dimension is a three-dimensional array.
      dom: .... Array of bordering domains, [vapor liquid]
      t_int_old Interface temperature array from the previous iteration
      alpha ... Underrelaxation factor

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
    M_AIR, M_H2O, M_salt = M_input
    
    # Update boundary values of air 
    M[dom[VAP]].bnd[S].val[:,:,:] = np.power(((1-a[dom[VAP]].bnd[S].val[:,:,:])/M_AIR \
                                  + a[dom[VAP]].bnd[S].val[:,:,:]/M_H2O),(-1))
                                  
    p_v[dom[VAP]].bnd[S].val[:,:,:] = a[dom[VAP]].bnd[S].val[:,:,:]  \
                                  *M[dom[VAP]].bnd[S].val[:,:,:]/M_H2O *  \
                                  (p_tot[dom[VAP]].val[:,:1,:] +1E5)
    
    # calculate interface temperature for saturated air 
    t_int = t_int_old * (1-alpha) + alpha \
          * properties.t_sat_salt(p_v[dom[VAP]].bnd[S].val[:,:,:],a[dom[LIQ]].val[:,-1:,:],M_salt,M_H2O)
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
    
