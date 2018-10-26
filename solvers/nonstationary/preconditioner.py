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
    if verbose is True:
        write.at(__name__)
        print(x)
    Mc = zeros(x.shape)
    Mw = zeros(x.shape)
    Ms = zeros(x.shape)
    Mb = zeros(x.shape)
    
    return Mc, Mw, Ms, Mb