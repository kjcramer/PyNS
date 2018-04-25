# Standard Python modules
from pyns.standard import *

# ScriNS modules
from pyns.constants          import *
from pyns.operators          import *

#==========================================================================
def latent_heat(t_in):
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
    h_d = 2454E3
  elif   (int(t_in) == 30):
    h_d = 2431E3
  elif   (int(t_in) == 40):
    h_d = 2407E3
  elif   (int(t_in) == 50):
    h_d = 2383E3
  elif   (int(t_in) == 60):
    h_d = 2359E3
  elif   (int(t_in) == 70):
    h_d = 2334E3
  elif   (int(t_in) == 80):
    h_d = 2309E3
  else: # values at 60C
    h_d = 2359E3
    
  print('latent heat at ' + '%2.0f' %t_in + 'C')
    
  return h_d  # end of function
