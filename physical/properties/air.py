# Standard Python modules
from pyns.standard import *


#==========================================================================
def air(t_in,rc):
#--------------------------------------------------------------------------
# Returns physical properties of air for given resolution 'rc'
#
# All values from: 
#   http://www.engineeringtoolbox.com/air-properties-d_156.html 
#--------------------------------------------------------------------------

  # Create and fill matrice for all properties
  
  if   (int(t_in) == 30): # interpolated!
    rho   = ones(rc) *    1.166     # density              [kg/m^3]
    mu    = ones(rc) *   16.04E-06  # viscosity            [Pa s]
    cp    = ones(rc) * 1005         # thermal capacity     [J/kg/K]
    kappa = ones(rc) *    0.0264    # thermal conductivity [W/m/K]
  elif (int(t_in) == 40):
    rho   = ones(rc) *    1.127
    mu    = ones(rc) *   16.97E-06
    cp    = ones(rc) * 1005
    kappa = ones(rc) *    0.0271
  elif (int(t_in) == 50): # interpolated!
    rho   = ones(rc) *    1.097
    mu    = ones(rc) *   17.935E-06
    cp    = ones(rc) * 1007
    kappa = ones(rc) *    0.0278
  elif (int(t_in) == 60):
    rho   = ones(rc) *    1.067
    mu    = ones(rc) *   18.90E-06
    cp    = ones(rc) * 1009
    kappa = ones(rc) *    0.0285
  elif (int(t_in) == 70): # interpolated!
    rho   = ones(rc) *    1.0335
    mu    = ones(rc) *   19.92E-06
    cp    = ones(rc) * 1009
    kappa = ones(rc) *    0.0292
  elif (int(t_in) == 80):
    rho   = ones(rc) *    1.000
    mu    = ones(rc) *   20.94E-06
    cp    = ones(rc) * 1009
    kappa = ones(rc) *    0.0299
  else: # values at 60C
    rho   = ones(rc) *    1.067
    mu    = ones(rc) *   18.90E-06
    cp    = ones(rc) * 1009
    kappa = ones(rc) *    0.0285
    
  print('air properties at ' + '%2.0f' %t_in + 'C')
    
  return rho, mu, cp, kappa  # end of function