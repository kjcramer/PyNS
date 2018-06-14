"""
Preconditioner Incomplete Cholesky, 0 order

Source:
  http://www.netlib.org/templates/templates.pdf
"""

# Standard Python modules
from pyns.standard import *

# PyNS modules
from pyns.constants      import *
from pyns.display        import write

# =============================================================================
def preconditioner_form(a, x, verbose = False):
# -----------------------------------------------------------------------------
    """
    Args:
      a: ...... Object of the type "Matrix", holding the system matrix.
      x: ...... Array of solution
      verbose:  Logical variable setting if solver will be verbose (print
                info on Python console) or not.

    Returns:
      m: preconditioning matrix.
    """

    Mc = zeros(x.shape)
    Mw = zeros(x.shape)
    Ms = zeros(x.shape)
    Mb = zeros(x.shape)
    
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        for k in range(x.shape[2]):
          sum = a.C[i][j][k]
          sum = sum - Mw[i][j][k] * Mw[i][j][k]
          sum = sum - Ms[i][j][k] * Ms[i][j][k]
          sum = sum - Mb[i][j][k] * Mb[i][j][k]
          sum = sqrt(sum)
          Mc[i][j][k] = sum
          
          #Mw[i+1][j][k] = - a.W[i+1][j][k]/Mc[i][j][k]
          
          if verbose is True:
            write.at(__name__)
            print(i)
    
    return Mc, Mw, Ms, Mb
    
# =============================================================================
def preconditioner_solve(z, r, m,verbose = False):
# -----------------------------------------------------------------------------
    """
    Args:
      z: ...... scalar
      r: ...... scalar
      m: ...... tuple containing the preconditioned system matrix
      verbose:  Logical variable setting if solver will be verbose (print
                info on Python console) or not.

    Returns:
      z: preconditioning matrix.
    """      
    
    [Mc, Mw, Ms, Mb] = m
    
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        for k in range(x.shape[2]):
          sum = r[i][j][k]
          sum = 