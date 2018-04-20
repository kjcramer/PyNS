# Standard Python modules
from pyns.standard import *


#==========================================================================
def water(t_in,rc,prin=True):
#--------------------------------------------------------------------------
# Returns physical properties for water for given resolution 'rc'
#
# All values from: 
#   http://www.engineeringtoolbox.com/water-properties-d_1508.html 
# kappa values from:
#   https://www.nist.gov/document-1703
#--------------------------------------------------------------------------

  # Create and fill matrices for all properties
  
  if   (int(t_in) == 20):
    rho   = ones(rc) *  998.3       # density              [kg/m^3]
    mu    = ones(rc) *    1.000E-03  # viscosity            [Pa s]
    cp    = ones(rc) * 4182         # thermal capacity     [J/kg/K]
    kappa = ones(rc) *    0.6009     # thermal conductivity [W/m/K]
  elif   (int(t_in) == 30):
    rho   = ones(rc) *  995.7
    mu    = ones(rc) *    0.798E-03
    cp    = ones(rc) * 4178
    kappa = ones(rc) *    0.6176
  elif   (int(t_in) == 40):
    rho   = ones(rc) *  992.3
    mu    = ones(rc) *    0.653E-03
    cp    = ones(rc) * 4179
    kappa = ones(rc) *    0.6322
  elif   (int(t_in) == 50):
    rho   = ones(rc) *  988.0
    mu    = ones(rc) *    0.547E-03
    cp    = ones(rc) * 4182
    kappa = ones(rc) *    0.6445
  elif   (int(t_in) == 60):
    rho   = ones(rc) *  983.0
    mu    = ones(rc) *    0.466E-3
    cp    = ones(rc) * 4185
    kappa = ones(rc) *    0.6546
  elif   (int(t_in) == 70):
    rho   = ones(rc) *  978.0
    mu    = ones(rc) *    0.404E-03
    cp    = ones(rc) * 4191
    kappa = ones(rc) *    0.6624
  elif   (int(t_in) == 80):
    rho   = ones(rc) *  972.0
    mu    = ones(rc) *    0.355E-03
    cp    = ones(rc) * 4198
    kappa = ones(rc) *    0.6680
  else: # values at 60C
    rho   = ones(rc) *  983.0
    mu    = ones(rc) *    0.466E-3
    cp    = ones(rc) * 4185
    kappa = ones(rc) *    0.6546
  
  if prin == True: 
    print('water properties at ' + '%2.0f' %t_in + 'C')
    
  return rho, mu, cp, kappa  # end of function