"""
Matrix-vector product, including booundary values, for PyNS matrix format.
"""

# Standard Python modules
from pyns.standard import *

# Specific Python modules
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from numpy import shape
from numpy import pad

# PyNS modules
from pyns.constants import W, E, S, N, B, T
from pyns.operators import cat_x, cat_y, cat_z

# =============================================================================
def mat_vec_bnd(a, phi):
# -----------------------------------------------------------------------------
    """
    Args:
      a:   Object of the type "Matrix", holding the matrix for multiplication.
      phi: Three-dimensional array holding a vector for multiplication.

    Returns:
      r: Result of the matrix-vector product, which is a vector stored
         in a three-dimensional array.
    """

    r = zeros(phi.val.shape)
    
    phi.exchange()

    r[:]  = a.C[:] * phi.val[:]
    
    r[:] -= a.W[:] * cat_x( (phi.bnd[W].val[ :1,:,:], 
                             phi.val       [:-1,:,:]) )
    r[:] -= a.E[:] * cat_x( (phi.val       [ 1:,:,:], 
                             phi.bnd[E].val[ :1,:,:]) )
    
    r[:] -= a.S[:] * cat_y( (phi.bnd[S].val[:, :1,:], 
                             phi.val       [:,:-1,:]) )
    r[:] -= a.N[:] * cat_y( (phi.val       [:, 1:,:], 
                             phi.bnd[N].val[:, :1,:]) )
    
    r[:] -= a.B[:] * cat_z( (phi.bnd[B].val[:,:, :1], 
                             phi.val       [:,:,:-1]) )
    r[:] -= a.T[:] * cat_z( (phi.val       [:,:, 1:], 
                             phi.bnd[T].val[:,:, :1]) )

    if True:
        # append x-boundary conditions to phi
        phi_gpu = cat_x(( phi.bnd[W].val[:1,:,:], phi.val[:,:,:], phi.bnd[E].val[:1,:,:] ))
        
        # append y-boundary conditions to phi
        y_bnd_N = pad( phi.bnd[N].val, ((1, 1), (0, 0), (0, 0)), 'constant' ) 
        y_bnd_S = pad( phi.bnd[S].val, ((1, 1), (0, 0), (0, 0)), 'constant' )
        phi_gpu = cat_y(( y_bnd_S, phi_gpu, y_bnd_N ))

        # append z-boundary condtiions to phi
        z_bnd_B = pad( phi.bnd[B].val, ((1, 1), (1, 1), (0, 0)), 'constant' )
        z_bnd_T = pad( phi.bnd[T].val, ((1, 1), (1, 1), (0, 0)), 'constant' )
        phi_gpu = cat_z(( z_bnd_B, phi_gpu, z_bnd_T ))
        
        # preparing a for gpu
        a_C_gpu = pad(a.C, ((1, 1), (1, 1), (1, 1)), 'constant')
        a_W_gpu = pad(a.W, ((0, 2), (1, 1), (1, 1)), 'constant')
        a_E_gpu = pad(a.E, ((2, 0), (1, 1), (1, 1)), 'constant')
        a_S_gpu = pad(a.S, ((1, 1), (0, 2), (1, 1)), 'constant')
        a_N_gpu = pad(a.N, ((1, 1), (2, 0), (1, 1)), 'constant')
        a_B_gpu = pad(a.B, ((1, 1), (1, 1), (0, 2)), 'constant')
        a_T_gpu = pad(a.T, ((1, 1), (1, 1), (2, 0)), 'constant')

       

    return r  # end of function
